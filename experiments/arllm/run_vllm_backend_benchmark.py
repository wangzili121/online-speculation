"""Run matched Transformers and vLLM workloads in separate GPU processes."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, model_output_cli_arguments

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.shared.artifacts import file_sha256 as _sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

def _run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def _matches_requested_run(
    path: Path,
    *,
    backend: str,
    config_sha256: str,
    data_sha256: str,
    limit: int,
    workers: int,
    methods: set[str],
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return (
            report.get("runtime_backend") == backend
            and report.get("experiment_config", {}).get("sha256") == config_sha256
            and report.get("evaluation", {}).get("dataset_sha256") == data_sha256
            and int(report.get("examples", -1)) == limit
            and int(report.get("workers", -1)) == min(workers, limit)
            and set(report.get("methods", {})) == methods
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_3090_aligned.toml"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--methods",
        default="base,best_of_n,conditional_is,conditional_is_small_proposal",
    )
    parser.add_argument("--tag", default="matched")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/validation"),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="skip either backend run when its report already exists",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.limit <= 0 or args.workers <= 0:
        raise ValueError("--limit and --workers must be positive")
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    if not methods:
        raise ValueError("--methods must contain at least one method")
    config_sha256 = _sha256(args.config)
    data_sha256 = _sha256(args.data)

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{args.config.stem}_runtime_{args.tag}"
    transformers_output = args.output_root / f"{stem}_transformers.json"
    vllm_output = args.output_root / f"{stem}_vllm.json"
    comparison_output = args.output_root / f"{stem}_comparison.json"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT), existing or "")
    ).rstrip(os.pathsep)

    common = [*model_output_cli_arguments(args),
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--limit",
        str(args.limit),
        "--workers",
        str(args.workers),
        "--methods",
        args.methods,
    ]
    for backend, output in (
        ("transformers", transformers_output),
        ("vllm", vllm_output),
    ):
        if args.reuse_existing and _matches_requested_run(
            output,
            backend=backend,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            limit=args.limit,
            workers=args.workers,
            methods=methods,
        ):
            print(f"REUSE {output}", flush=True)
            continue
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_async_benchmark.py",
                *common,
                "--backend",
                backend,
                "--output",
                str(output),
            ],
            environment,
        )

    _run(
        [
            sys.executable,
            "experiments/arllm/summarize_vllm_backend.py",
            "--transformers",
            str(transformers_output),
            "--vllm",
            str(vllm_output),
            "--output",
            str(comparison_output),
        ],
        environment,
    )
    print(f"WROTE {comparison_output}", flush=True)


if __name__ == "__main__":
    main()
