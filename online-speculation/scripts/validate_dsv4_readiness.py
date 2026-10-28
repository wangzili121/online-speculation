#!/usr/bin/env python3
"""Run bounded checks without starting engines or reserving devices.

CPU checks, single-NPU checks and live-service checks are separate stages.
A successful small-model stage is not full-model or performance acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CPU_TESTS = (
    "dsv4_hybrid draft_sampling dsv4_deployment sampling_math sampling_execution "
    "vllm_ascend_proposer dsv4_custom_proposer vllm_trace vllm_artifacts "
    "vllm_results vllm_workload dsv4_readiness dsv4_prolog"
).split()


def commands(args):
    python = sys.executable

    def script(name, *options):
        return [python, f"scripts/{name}.py", *map(str, options)]
    if args.stage == "cpu":
        return [
            ("unit", [python, "-m", "pytest", "-q",
                      *[f"tests/test_{name}.py" for name in CPU_TESTS]]),
            ("head-contract", script("selftest_dsv4_contract")),
            ("evaluation-contract", script("selftest_dsv4_evaluation")),
        ]
    if args.stage == "npu":
        return [
            ("indexer", script("check_dsv4_indexer_npu")),
            ("rejection", script("check_dsv4_rejection_npu")),
            ("paged", script("check_dsv4_paged_npu")),
            *[(f"hybrid-h{heads}", script("smoke_dsv4_hybrid_npu",
                "--heads", heads, "--iterations", 10)) for heads in (8, 16, 32)],
            *[(f"attention-{case}", script("benchmark_dsv4_attention_paths",
                "--case", case, "--iterations", 10)) for case in ("joined", "gather", "sparse", "c128-sparse")],
        ]
    common = ["--base-url", args.base_url, "--model", args.model]
    snapshot = args.output / "snapshot.json"
    return [
        ("snapshot", script("validate_dsv4_service", "snapshot", *common,
                            "--cases", args.cases, "--output", snapshot,
                            *(["--include-long"] if args.include_long else []))),
        ("greedy-replay", script("check_dsv4_sampling_replay", *common,
             "--temperature", 0, "--output", args.output / "greedy-replay.json")),
        ("random-replay", script("check_dsv4_sampling_replay", *common,
             "--output", args.output / "random-replay.json")),
        ("lifecycle", script("check_dsv4_lifecycle", *common, "--baseline", snapshot,
             "--eos-token-id", args.eos_token_id, "--output", args.output / "lifecycle.json")),
    ]


def run_check(command, log_path, timeout, env):
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            timed_out, code = True, 124
    return {"command": command, "returncode": code, "timed_out": timed_out,
            "seconds": time.monotonic() - started, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "npu", "service"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per check")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/dsv4/correctness-ops-smoke.json")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.stage == "service" and (not args.base_url or not args.model or args.eos_token_id is None):
        parser.error("service requires --base-url, --model and --eos-token-id")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    report = {"stage": args.stage, "scope": "not full-model acceptance", "checks": {},
              "visible_devices": env.get("ASCEND_RT_VISIBLE_DEVICES"),
              "batch_invariant": env.get("VLLM_BATCH_INVARIANT"), "passed": False}
    sources = ROOT / "src/blockspec/vllm_ascend"
    report["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources.glob("*.py"))
    }
    patcher = ROOT / "runtime/vllm_ascend_v0_26/patch_runtime.py"
    report["source_sha256"]["patch_runtime.py"] = hashlib.sha256(patcher.read_bytes()).hexdigest()
    if args.stage == "service":
        report["service"] = {"base_url": args.base_url, "model": args.model,
                             "eos_token_id": args.eos_token_id, "cases": str(args.cases)}
    runtime_lock = ROOT / "runtime/vllm_ascend_v0_26/runtime-lock.local.json"
    if runtime_lock.is_file():
        report["runtime_lock"] = json.loads(runtime_lock.read_text())
    for name, command in commands(args):
        print(f"START {name}", flush=True)
        result = run_check(command, args.output / f"{name}.log", args.timeout, env)
        report["checks"][name] = result
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{'PASS' if result['returncode'] == 0 else 'FAIL'} {name}: {result['seconds']:.1f}s", flush=True)
    report["passed"] = all(item["returncode"] == 0 for item in report["checks"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
