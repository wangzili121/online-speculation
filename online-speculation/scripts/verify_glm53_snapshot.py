#!/usr/bin/env python3
"""Verify a local GLM replica against a previously pinned source manifest."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from watch_glm53_validation import write_json


def load_manifest(path, expected_sha256):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("source manifest SHA256 mismatch")
    manifest = json.loads(content)
    if manifest["model_id"] != "Eco-Tech/GLM-5.3-Flash-w8a8" or not manifest["files"]:
        raise ValueError("expected a pinned GLM-Flash source manifest")
    names = set()
    for item in manifest["files"]:
        name = Path(item["path"])
        if (name.is_absolute() or ".." in name.parts or item["path"] in names
                or item["size"] < 0):
            raise ValueError("invalid source manifest entry")
        names.add(item["path"])
    return manifest


def verify_file(directory, item):
    path = directory / item["path"]
    if path.stat().st_size != item["size"]:
        raise ValueError(f"size mismatch: {item['path']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"SHA256 mismatch: {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--wait-hours", type=float, default=2)
    args = parser.parse_args()
    status_path = args.state_directory / "status.json"
    manifest = load_manifest(status_path.with_name("source_manifest.json"), args.manifest_sha256)
    files = manifest["files"]
    state = dict(state="waiting_files", directory=str(args.model.resolve()), model_id=manifest["model_id"],
                 files=len(files), total_bytes=sum(item["size"] for item in files),
                 verified_files=0, verified_bytes=0, source_manifest_sha256=args.manifest_sha256,
                 expected_git_head=manifest["git_head"])
    write_json(status_path, state)
    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        while not all((args.model / item["path"]).is_file()
                      and (args.model / item["path"]).stat().st_size == item["size"] for item in files):
            if time.monotonic() >= deadline:
                raise TimeoutError("replica did not finish arriving")
            time.sleep(30)
        state["state"] = "verifying"
        write_json(status_path, state)
        for item in files:
            verify_file(args.model, item)
            state["verified_files"] += 1
            state["verified_bytes"] += item["size"]
            write_json(status_path, state)
        state.update(state="complete", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(status_path, state)


if __name__ == "__main__":
    main()
