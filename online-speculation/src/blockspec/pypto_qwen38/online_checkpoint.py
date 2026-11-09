"""在线状态的文件交换；数值转换与更新仍由设备 kernel 完成。"""

import hashlib
import json
from pathlib import Path


FORMAT = "orthrus-qwen38-pypto-last1-adam-v1"
STATES = ("master", "first", "second")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def identity(head):
    from .prepared_head import contract
    payload = (head.prepared_manifest["payload_sha256"] if head.prepared_manifest is not None
               else hashlib.sha256(memoryview(head.shared_weights.numpy())).hexdigest())
    return dict(contract=contract(head), payload_sha256=payload)


def save(optimizer, directory, head_identity):
    import torch
    from safetensors.torch import save_file
    if optimizer.steps < 1:
        raise ValueError("checkpoint requires a committed optimizer step")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for name, handles in optimizer.state.items():
        for label, handle in zip(STATES, handles, strict=True):
            value = torch.empty(tuple(handle.shape), dtype=torch.float32)
            optimizer.worker.copy_from(value.data_ptr(), handle.data_ptr, value.numel()*4)
            tensors[name+"."+label] = value
    path = directory / "optimizer.safetensors"
    save_file(tensors, str(path.with_suffix(".part")))
    path.with_suffix(".part").rename(path)
    manifest = dict(format=FORMAT, head=head_identity, step=optimizer.steps,
        beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0., payload_sha256=sha256(path))
    (directory/"manifest.json.part").write_text(json.dumps(manifest, indent=2)+"\n")
    (directory/"manifest.json.part").rename(directory/"manifest.json")
    return manifest


def read(directory, head_identity, shapes):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    expected = dict(format=FORMAT, head=head_identity, step=manifest.get("step"),
        beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0., payload_sha256=manifest.get("payload_sha256"))
    if manifest != expected or type(manifest["step"]) is not int or manifest["step"] < 1:
        raise ValueError("incompatible last-layer optimizer checkpoint")
    path = directory/"optimizer.safetensors"
    if sha256(path) != manifest["payload_sha256"]:
        raise ValueError("optimizer checkpoint SHA256 mismatch")
    values = load_file(str(path), device="cpu")
    schema = {name+"."+label: tuple(shape) for name, shape in shapes.items() for label in STATES}
    if set(values) != set(schema) or any(value.dtype != torch.float32 or tuple(value.shape) != schema[name]
                                        for name, value in values.items()):
        raise ValueError("optimizer tensor schema changed")
    return manifest["step"], values


def restore(optimizer, weights, directory, head_identity):
    """仅在启动时恢复；所有状态检查通过后才发布末层，随后保留 Adam 计数。"""
    if optimizer.steps or weights.version:
        raise RuntimeError("restore is only allowed before serving starts")
    step, values = read(directory, head_identity, optimizer.programs.shapes)
    for name, handles in optimizer.state.items():
        for label, handle in zip(STATES, handles, strict=True):
            value = values[name+"."+label]
            optimizer.worker.copy_to(handle.data_ptr, value.data_ptr(), value.numel()*4)
    for column in range(3):
        optimizer.statistics({name: handles[column] for name, handles in optimizer.state.items()})
    optimizer.render(0, optimizer.state)
    banks = tuple(optimizer.execution[name][0] for name in
                  ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight"))
    weights.publish(63, banks, expected_version=0)
    optimizer.steps, optimizer.index, weights.version = step, 0, step
    optimizer.zero_grad()
