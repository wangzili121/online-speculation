"""只在验收阶段回读 draft 边界；不进入正常推理或计时路径。"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def head_digest(head):
    """校验实际运行参数及布局，不包含共享缓冲的对齐空白字节。"""
    digest = hashlib.sha256()
    for layer in range(64):
        values = head.banks(layer)
        if layer % 4 != 3:
            values[5] = values[5].float()
        for index, value in enumerate(values):
            digest.update(json.dumps([layer, index, list(value.shape), str(value.dtype)]).encode())
            digest.update(memoryview(value.contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def read_tensor(native, handle):
    value = torch.empty(tuple(handle.shape), dtype=handle.dtype)
    native.worker.copy_from(value.data_ptr(), handle.data_ptr, value.numel() * value.element_size())
    return value


def save_history(native, request, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resident = native.requests[request.request_id]
    past = resident.pages.length
    conv, recurrent = native._read_state(resident.slot)
    values = dict(conv=conv, recurrent=recurrent)
    positions = torch.arange(past, past + 32)
    values.update({name: native.rope[name][positions].contiguous() for name in ("cos", "sin")})
    # 只读取已提交页的有效行，转成 HF 的 [batch, heads, tokens, dim]。
    for name, pool in zip(("key", "value"), native.resident[2:], strict=True):
        layers, heads, blocks, rows, dim = pool.shape
        for layer in range(layers):
            result = torch.empty((1, heads, past, dim), dtype=pool.dtype)
            for head in range(heads):
                for logical, physical in enumerate(resident.pages.pages):
                    count = min(rows, past - logical * rows)
                    if count <= 0:
                        break
                    value = torch.empty((count, dim), dtype=pool.dtype)
                    offset = ((layer * heads + head) * blocks + physical) * rows * dim * value.element_size()
                    native._copy_offset(value, pool, offset, False)
                    result[0, head, logical * rows:logical * rows + count].copy_(value)
            values[f"{name}.{layer}"] = result
    save_file({name: value.contiguous() for name, value in values.items()}, str(directory / "history.safetensors"))
    (directory / "history.json").write_text(json.dumps(dict(past=past, pages=list(resident.pages.pages))) + "\n")


@contextmanager
def capture_layers(draft, directory):
    """保存每层真实输入/输出；异常时恢复原方法，不影响后续服务。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = draft._run
    records = {}

    def observe(name, args):
        layer = args[-2] if name == "gdn" else args[-1] if name == "full_prepare" else args[-3] if name in {"full", "full_layer"} else None
        if name in {"gdn", "full_prepare", "full_layer"}:
            records[layer] = dict(input=read_tensor(draft.native, args[0]))
        result = original(name, args)
        if name == "gdn":
            records[layer]["output"] = read_tensor(draft.native, args[-3])
        elif name == "full_prepare":
            records[layer].update({key: read_tensor(draft.native, handle)
                                   for key, handle in zip(("query", "key", "value", "gate"), args[-5:-1], strict=True)})
        elif name in {"full", "full_layer"}:
            records[layer]["output"] = read_tensor(draft.native, args[-4])
        elif name == "finish":
            save_file(dict(hidden=read_tensor(draft.native, args[0]), logits=read_tensor(draft.native, args[-1])),
                      str(directory / "finish.safetensors"))
        return result

    draft._run = observe
    try:
        yield
        if set(records) != set(range(64)) or any("output" not in row for row in records.values()):
            raise RuntimeError("incomplete 64-layer draft capture")
        for layer, row in records.items():
            save_file(row, str(directory / f"layer{layer}.safetensors"))
    finally:
        draft._run = original
