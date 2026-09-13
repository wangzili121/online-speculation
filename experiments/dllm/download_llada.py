"""Download the pinned LLaDA-MoE checkpoint and verify every weight shard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib

from huggingface_hub import snapshot_download as huggingface_snapshot_download

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.shared.artifacts import file_sha256 as sha256

MODEL_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
REVISION = "783d3467f108d28ac0a78d3e41af16ab05cabd8d"
MODELSCOPE_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
MODELSCOPE_REVISION = "master"
WEIGHTS = (
    (
        "model-00001-of-00003.safetensors",
        4_999_258_928,
        "84a7f34af2f3f14d767b0106b8fa7f0d7f9b95a0eeac74f2ab3f21bd69a03908",
    ),
    (
        "model-00002-of-00003.safetensors",
        4_997_188_984,
        "2ddb9174a03003263250c789942372382e2ce97f115377e28cb749c082f0b2d7",
    ),
    (
        "model-00003-of-00003.safetensors",
        4_717_712_520,
        "b1998424a021938681487ece097838f7131324eb4f776b8b0ce0c515526b31f4",
    ),
)
DEFAULT_OUTPUT = Path("models/LLaDA-MoE-7B-A1B-Instruct-783d3467")


def validate_checkpoint(directory: Path) -> dict[str, object]:
    validated: dict[str, dict[str, str | int]] = {}
    for name, expected_size, expected_digest in WEIGHTS:
        weight = directory / name
        if not weight.is_file():
            raise FileNotFoundError(f"missing LLaDA weight shard: {weight}")
        size = weight.stat().st_size
        if size != expected_size:
            raise RuntimeError(
                f"weight size mismatch for {name}: expected {expected_size}, got {size}"
            )
        digest = sha256(weight)
        if digest != expected_digest:
            raise RuntimeError(
                f"weight SHA-256 mismatch for {name}: expected {expected_digest}, got {digest}"
            )
        validated[name] = {"bytes": size, "sha256": digest}
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights": validated,
        "path": str(directory.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--endpoint",
        help="Optional Hugging Face-compatible endpoint; the official SHA is always checked.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Download transport. Both sources are accepted only after official SHA validation.",
    )
    parser.add_argument(
        "--modelscope-workers",
        type=int,
        default=8,
        help="Range streams within one ModelScope weight shard.",
    )
    parser.add_argument(
        "--modelscope-part-mb",
        type=int,
        default=32,
        help="Recoverable ModelScope byte-range size in MiB.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download code/tokenizer/config but omit the 14.7 GB weight shards.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not access the network; validate an existing local checkpoint.",
    )
    args = parser.parse_args()
    if args.output is None:
        if args.config is None:
            args.output = DEFAULT_OUTPUT
        else:
            with args.config.open("rb") as source:
                args.output = Path(str(tomllib.load(source)["model"]["path"]))
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    if not args.validate_only and not args.metadata_only and args.output.is_dir():
        try:
            existing = validate_checkpoint(args.output)
        except (FileNotFoundError, RuntimeError):
            pass
        else:
            print(existing)
            return

    if not args.validate_only:
        patterns = [
            "*.json",
            "*.py",
            "README.md",
        ]
        if not args.metadata_only:
            patterns.append("*.safetensors")
        if args.source == "huggingface":
            huggingface_snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_dir=args.output,
                allow_patterns=patterns,
            )
        else:
            if args.modelscope_workers <= 0 or args.modelscope_part_mb <= 0:
                raise ValueError("ModelScope worker count and part size must be positive")
            os.environ["MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS"] = str(
                args.modelscope_workers
            )
            os.environ["MODELSCOPE_DOWNLOAD_PART_SIZE_MB"] = str(
                args.modelscope_part_mb
            )
            os.environ["MODELSCOPE_DOWNLOAD_MAX_RETRIES"] = "20"
            os.environ["MODELSCOPE_DOWNLOAD_TIMEOUT"] = "120"
            from modelscope.hub.snapshot_download import (
                snapshot_download as modelscope_snapshot_download,
            )

            modelscope_snapshot_download(
                MODELSCOPE_ID,
                revision=MODELSCOPE_REVISION,
                local_dir=str(args.output),
                allow_patterns=patterns,
                max_workers=1,
            )
    if args.metadata_only:
        print(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "download_source": args.source,
                "path": str(args.output.resolve()),
                "weights": "not requested",
            }
        )
    else:
        print(validate_checkpoint(args.output))


if __name__ == "__main__":
    main()
