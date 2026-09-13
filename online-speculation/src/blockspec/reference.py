"""Caller-selected integrity gate for optional external numerical references."""

import hashlib
import json
from pathlib import Path


def checked_reference(directory, manifest):
    """Validate caller-supplied local integrity metadata before importing code."""
    directory = Path(directory).resolve()
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))["models"]["base"]
    names = list(spec["reference_lf_sha256"]) + [spec["weight_filename"]]
    if not names or any((directory / name).resolve().parent != directory for name in names):
        raise ValueError("reference files must be direct children of the model directory")
    for name, expected in spec["reference_lf_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"reference source/config differs from reviewed pin: {name}")
    digest = hashlib.sha256()
    with (directory / spec["weight_filename"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["weight_sha256"]:
        raise ValueError("reference weights differ from reviewed pin")
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if set(index["weight_map"].values()) != {spec["weight_filename"]}:
            raise ValueError("reference index points outside the checked weight shard")
    return spec
