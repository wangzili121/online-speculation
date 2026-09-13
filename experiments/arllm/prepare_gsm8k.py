"""Download pinned public data and model revisions used by the reproduction."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import time
import tomllib
from pathlib import Path

from huggingface_hub import snapshot_download

from inference_scaling.shared.evaluation import download_gsm8k

from experiments.shared.artifacts import checkpoint_weight_hashes, weight_manifest_digest
from inference_scaling.shared.model_loading import model_loading_options, resolve_checkpoint_path


def _weight_digest(directory: str) -> str:
    return weight_manifest_digest(checkpoint_weight_hashes(Path(directory), cache_directory=Path('.cache/artifact_hashes')))


def _download_model(
    source: str,
    revision: str | None,
    destination: str,
    expected_weight_sha256: str | None,
    retries: int,
    *,
    modelscope_source: str | None = None,
) -> None:
    try:
        actual = _weight_digest(destination)
        if expected_weight_sha256 is None or actual == expected_weight_sha256:
            print(f"verified existing model: {destination}", flush=True)
            return
    except FileNotFoundError:
        pass
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"model={source} revision={revision} attempt={attempt}", flush=True)
            if modelscope_source is not None:
                from modelscope import snapshot_download as modelscope_download

                modelscope_download(
                    modelscope_source,
                    revision="master",
                    local_dir=destination,
                    max_workers=4,
                    ignore_file_pattern=["*.md"],
                )
            else:
                snapshot_download(
                    source,
                    revision=revision,
                    local_dir=destination,
                    max_workers=1,
                )
            actual = _weight_digest(destination)
            if expected_weight_sha256 is not None and actual != expected_weight_sha256:
                raise ValueError(
                    f"weight checksum mismatch for {source}: "
                    f"expected {expected_weight_sha256}, got {actual}"
                )
            return
        except Exception as caught:  # network retries are intentionally broad
            error = caught
            if attempt < retries:
                print(f"download retry after {type(caught).__name__}: {caught}", flush=True)
                time.sleep(2)
    assert error is not None
    raise error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_standard.toml"))
    parser.add_argument("--train-data", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--test-data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    train_dataset = download_gsm8k(args.train_data, split="train")
    test_dataset = download_gsm8k(args.test_data, split="test")
    print(f"verified GSM8K train: {train_dataset}")
    print(f"verified GSM8K test: {test_dataset}")
    if args.skip_models:
        return
    if args.allow_download is False:
        from experiments.arllm.runtime import validate_model_artifacts
        validate_model_artifacts(config, [role for role in ("base", "proposal") if role in config["models"]])
        return
    for role in ("base", "proposal"):
        if role not in config["models"]:
            continue
        options = model_loading_options(config, role)
        model = str(config["models"][role])
        source_name = config["models"].get(f"{role}_source")
        if source_name is None:
            resolved = resolve_checkpoint_path(
                model, revision=options.get("revision"), cache_dir=options.get("cache_dir"),
                local_files_only=args.allow_download is False,
            )
            print(f"verified {role}: {resolved} sha256={_weight_digest(str(resolved))}", flush=True)
        else:
            _download_model(
                str(source_name), options.get("revision"), model,
                config["models"].get(f"{role}_weight_sha256"), args.retries,
                modelscope_source=config["models"].get(f"{role}_modelscope_source"),
            )
    print(
        "The RL comparison is trained locally with experiments/arllm/train_gsm8k_grpo.py.",
        flush=True,
    )


if __name__ == "__main__":
    main()
