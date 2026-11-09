"""保存并读取已由 PyPTO 校验、转换的 head；加载不再计算参数值。"""

import hashlib
import json
from pathlib import Path
import re

from .baseline import PYPTO_COMMIT


FORMAT = "orthrus-qwen38-pypto-banks-v1"


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


def save(head, directory):
    if head._prepared is None or getattr(head, "preparation_backend", None) != "pypto":
        raise ValueError("only a PyPTO-prepared head may be exported")
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
    (directory / "weights.bin.part").rename(directory / "weights.bin")
    manifest = dict(format=FORMAT, contract=contract(head), runtime_tp=head.layout.size,
                    runtime_rank=head.layout.rank, producer_pypto=PYPTO_COMMIT,
                    payload_bytes=len(data), payload_sha256=digest.hexdigest())
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
