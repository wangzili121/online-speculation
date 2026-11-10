"""保存并读取已由 PyPTO 校验、转换的 head；加载不再计算参数值。"""

import hashlib
import json
from pathlib import Path
import re
from dataclasses import dataclass

from .baseline import PYPTO_COMMIT


FORMAT = "orthrus-qwen38-pypto-banks-v1"


@dataclass(frozen=True)
class DeviceBank:
    """导出时保留分配 owner 和字节偏移，避免向 DMA 传入内部地址。"""
    owner: object
    shape: tuple
    dtype: object
    byte_offset: int

    @property
    def data_ptr(self):
        return self.owner.data_ptr


def contract(head):
    from .head import POLICY
    hashes = [hashlib.sha256((head.model_root / name).read_bytes()).hexdigest()
              for name in ("config.json", "model.safetensors.index.json")]
    return dict(format="orthrus-qwen38-native-tp-v1", base_config_sha256=hashes[0],
                base_index_sha256=hashes[1], tp_size=head.tp, block_size=head.block_size,
                mask_token_id=head.mask_token_id, hybrid_policy=POLICY)


def configure(head, manifest):
    required = {"format", "contract", "runtime_tp", "runtime_rank", "producer_pypto", "payload_bytes", "payload_sha256"}
    if (set(manifest) != required or manifest.get("format") != FORMAT
            or manifest.get("producer_pypto") != PYPTO_COMMIT):
        raise ValueError("incompatible prepared head manifest")
    source = manifest.get("contract")
    if not isinstance(source, dict) or type(source.get("tp_size")) is not int or source["tp_size"] not in {1, 2, 4}:
        raise ValueError("invalid prepared head source contract")
    head.tp = source["tp_size"]
    if (source != contract(head) or type(manifest["runtime_tp"]) is not int or type(manifest["runtime_rank"]) is not int
            or (manifest["runtime_tp"], manifest["runtime_rank"]) != (head.layout.size, head.layout.rank)):
        raise ValueError("prepared head must match base, block, mask and runtime TP layout")
    if (type(manifest["payload_bytes"]) is not int or manifest["payload_bytes"] < 1
            or not isinstance(manifest["payload_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest["payload_sha256"]) is None):
        raise ValueError("invalid prepared head payload metadata")
    head.prepared_manifest = manifest


def load(head):
    from .head_loading import allocate_host_banks, host_layout
    manifest = head.prepared_manifest
    path = head.directory / "weights.bin"
    expected = host_layout(head)[2]
    if expected != manifest["payload_bytes"] or path.stat().st_size != expected:
        raise ValueError("prepared head payload size does not match tensor schema")
    storage, banks = allocate_host_banks(head)
    data = memoryview(storage.numpy())
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for start in range(0, len(data), 16 << 20):
            chunk = data[start:start + (16 << 20)]
            done = 0
            while done < len(chunk):
                count = file.readinto(chunk[done:])
                if not count:
                    raise ValueError("truncated prepared head payload")
                done += count
            digest.update(chunk)
    if digest.hexdigest() != manifest["payload_sha256"]:
        raise ValueError("prepared head payload SHA256 mismatch")
    head.shared_weights, head._prepared = storage, banks


def save(head, directory, *, last_layer=None, layer_banks=None, worker=None):
    if head._prepared is None or getattr(head, "preparation_backend", None) != "pypto":
        raise ValueError("only a PyPTO-prepared head may be exported")
    patches = []
    if last_layer is not None:
        if layer_banks is not None:
            raise ValueError("choose last_layer or layer_banks")
        layer_banks = {63:last_layer}
    if layer_banks is not None:
        from math import prod
        if worker is None or not layer_banks or any(type(layer) is not int or not 0 <= layer < 64 for layer in layer_banks):
            raise ValueError("suffix device banks and their worker required")
        for layer,banks in layer_banks.items():
            expected = head.banks(layer)
            if len(banks) != len(expected):
                raise ValueError("all six Full Attention or nine GDN device banks required")
            for original, updated in zip(expected, banks, strict=True):
                if tuple(original.shape) != tuple(updated.shape) or original.dtype != updated.dtype:
                    raise ValueError("exported suffix schema changed")
                offset = original.data_ptr()-head.shared_weights.data_ptr()
                size = prod(original.shape)*original.element_size()
                if offset < 0 or offset+size > head.shared_weights.numel():
                    raise ValueError("suffix bank is outside the canonical payload")
                patches.append((offset, size, updated))
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    data = memoryview(head.shared_weights.numpy())
    digest = hashlib.sha256()
    # manifest 最后写入；未完成的目录不能被服务当作有效模型加载。
    with (directory / "weights.bin.part").open("xb") as file:
        for start in range(0, len(data), 16 << 20):
            chunk = data[start:start + (16 << 20)]
            file.write(chunk)
            digest.update(chunk)
    if patches:
        import torch
        scratch = torch.empty(16 << 20, dtype=torch.uint8)
        with (directory/"weights.bin.part").open("r+b") as file:
            for target, size, updated in patches:
                file.seek(target)
                for offset in range(0, size, scratch.numel()):
                    count = min(scratch.numel(), size-offset)
                    worker.copy_from(scratch.data_ptr(), updated.data_ptr, count,
                                     src_offset=getattr(updated,"byte_offset",0)+offset)
                    file.write(memoryview(scratch.numpy())[:count])
        digest = hashlib.sha256()
        with (directory/"weights.bin.part").open("rb") as file:
            while chunk := file.read(16 << 20):
                digest.update(chunk)
    (directory / "weights.bin.part").rename(directory / "weights.bin")
    manifest = dict(format=FORMAT, contract=contract(head), runtime_tp=head.layout.size,
                    runtime_rank=head.layout.rank, producer_pypto=PYPTO_COMMIT,
                    payload_bytes=len(data), payload_sha256=digest.hexdigest())
    (directory / "manifest.json.part").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "manifest.json.part").rename(directory / "manifest.json")
    return manifest
