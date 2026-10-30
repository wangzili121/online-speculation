#!/usr/bin/env python3
"""Run isolated native-engine acceptance cells and retain logs and evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def memory_growth(report, tolerance=64 * 1024**2):
    """Compare matching request boundaries after the first warm-up cycle."""
    samples = report.get("memory", [])
    groups = {}
    references = {item["case"]: item.get("reference_case", item["case"]) for item in report["outputs"]}
    for sample in samples:
        groups.setdefault(references[sample["case"]], []).append(sample["workers"])
    maximum = 0
    for runs in groups.values():
        if len(runs) < 3:
            continue
        for before, after in zip(runs[-2], runs[-1], strict=True):
            maximum = max(maximum, after["allocated"] - before["allocated"])
    if maximum > tolerance:
        raise AssertionError(f"live NPU allocation grew by {maximum} bytes after warm-up")
    return dict(max_live_growth_bytes=maximum, tolerance_bytes=tolerance,
                repeated_boundaries=sum(len(runs) >= 3 for runs in groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=12884901888)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", default=[2, 8, 17, 32])
    parser.add_argument("--sampling-trials", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (args.max_model_len < 2048 or args.repeats < 3 or any(k < 2 or k > 32 for k in args.blocks)
            or not 16 <= args.sampling_trials <= 4096):
        parser.error("require context >=2048, repeats >=3, blocks 2..32, sampling trials 16..4096")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).with_name("validate_qwen38_vllm.py")),
               "--model", args.model, "--mask-token-id", str(args.mask_token_id), "--tp", str(args.tp),
               "--max-model-len", str(args.max_model_len),
               "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]
    common = dict(ORTHRUS_VALIDATE_EXTENDED="1", ORTHRUS_VALIDATE_SOAK="1",
                  ORTHRUS_VALIDATE_COMPREHENSIVE="1")
    results = []

    def run(name, extra, flags):
        output = args.output_dir / f"{name}.json"
        log = args.output_dir / f"{name}.log"
        if output.exists() or log.exists():
            raise FileExistsError(f"refusing to overwrite evidence for {name}")
        env = {key: value for key, value in os.environ.items() if not key.startswith("ORTHRUS_VALIDATE_")}
        env.update(common | flags | {"ORTHRUS_VALIDATE_OUTPUT": str(output)})
        started = time.perf_counter()
        print(f"START {name}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(command + extra, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=2700, check=False)
        row = dict(case=name, returncode=completed.returncode, seconds=time.perf_counter() - started,
                   output=str(output), log=str(log))
        results.append(row)
        (args.output_dir / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"acceptance cell {name} failed; see {log}")
        report = json.loads(output.read_text())
        if int(flags.get("ORTHRUS_VALIDATE_REPEATS", "1")) >= 3:
            row["memory_check"] = memory_growth(report)
            (args.output_dir / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"PASS {name} ({row['seconds']:.1f}s)", flush=True)
        return report

    baseline = args.baseline or args.output_dir / "target.json"
    if args.baseline:
        reference = json.loads(baseline.read_text())
        if (not reference["target_only"] or reference["tp"] != args.tp
                or reference.get("completed") is False):
            raise ValueError("baseline must be target-only with matching TP")
    else:
        run("target", ["--block-size", "32"], dict(ORTHRUS_VALIDATE_TARGET_ONLY="1",
            ORTHRUS_VALIDATE_SAMPLING_TRIALS=str(args.sampling_trials)))
    candidate = dict(ORTHRUS_VALIDATE_BASELINE=str(baseline), ORTHRUS_VALIDATE_AUDIT="1",
                     ORTHRUS_VALIDATE_NUMERIC="0")
    for block in args.blocks:
        flags = candidate | (dict(ORTHRUS_VALIDATE_SAMPLING_TRIALS=str(args.sampling_trials),
                                  ORTHRUS_VALIDATE_FROZEN="1") if block == 32 else {})
        run(f"k{block}-fixed", ["--block-size", str(block)], flags)
    for last, stride in ((1, 16), (4, 1)):
        run(f"last{last}-online", ["--block-size", "32", "--online", "--last-layers", str(last),
            "--update-stride", str(stride), "--checkpoint-dir", str(args.output_dir / f"head-last{last}"),
            "--checkpoint-every", "1000000"], candidate | {"ORTHRUS_VALIDATE_REPEATS": str(args.repeats),
                                                          "ORTHRUS_VALIDATE_FROZEN": "1"})
    last4 = json.loads((args.output_dir / "last4-online.json").read_text())
    run("head-reload", ["--block-size", "32", "--online", "--last-layers", "1", "--head-dir",
        last4["checkpoint"][0]["directory"]], candidate)
    print(f"ACCEPTANCE PASSED: {args.output_dir / 'matrix.json'}", flush=True)


if __name__ == "__main__":
    main()
