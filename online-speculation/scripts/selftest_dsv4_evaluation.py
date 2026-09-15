#!/usr/bin/env python3
"""Exercise correctness and performance gates without a running model server."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.reporting import dumps  # noqa: E402
from blockspec.vllm_ascend.artifacts import write_private_json  # noqa: E402


def _write(path: Path, value: object) -> None:
    write_private_json(path, value)


def _record(token: str, suffix: str = "") -> dict[str, object]:
    return {"text": token + suffix, "tokens": [token, suffix], "usage": {}}


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orthrus-evaluation-") as temporary:
        base = Path(temporary)
        baseline_path = base / "baseline.json"
        candidate_path = base / "candidate.json"
        correctness_path = base / "correctness.json"

        greedy = [_record("greedy")]
        repeated = [_record("stable") for _ in range(3)]
        baseline_sampling = [_record("A", "0"), _record("B", "1")]
        candidate_sampling = [_record("B", "1"), _record("A", "0")]
        common = {
            "schema": "dsv4-service-snapshot-v1",
            "greedy": greedy,
            "cache_repetitions": repeated,
        }
        _write(
            baseline_path,
            {
                **common,
                "sampling": baseline_sampling,
                "sampling_replay": baseline_sampling,
            },
        )
        _write(
            candidate_path,
            {
                **common,
                "sampling": candidate_sampling,
                "sampling_replay": candidate_sampling,
            },
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_dsv4_service.py"),
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(correctness_path),
            ]
        )
        correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
        if not correctness["passed"]:
            raise RuntimeError("distributionally equivalent snapshots failed correctness")
        if correctness["cross_service_fixed_seed_match_rate_diagnostic"] != 0:
            raise RuntimeError("self-test did not exercise independent RNG consumption")

        target_dir = base / "target"
        orthrus_dir = base / "orthrus"
        benchmark = {
            "request_goodput": 100.0,
            "spec_decode_num_drafts": 10,
            "spec_decode_draft_tokens": 310,
            "spec_decode_accepted_tokens": 155,
            "spec_decode_acceptance_length": 15.5,
        }
        _write(target_dir / "target-2k-c1.json", {"request_goodput": 100.0})
        _write(orthrus_dir / "orthrus-2k-c1.json", {**benchmark, "request_goodput": 105.0})
        gate_path = base / "gate.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_dsv4_gate.py"),
                "--target-dir",
                str(target_dir),
                "--orthrus-dir",
                str(orthrus_dir),
                "--correctness",
                str(correctness_path),
                "--output",
                str(gate_path),
            ]
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate["passed"]:
            raise RuntimeError("exactly 5% Orthrus improvement failed the goodput gate")

    print(
        dumps(
            {
                "status": "ok",
                "greedy_exact": True,
                "internal_seed_replay": True,
                "distribution_test": True,
                "cross_service_rng_independent": True,
                "five_percent_goodput_gate": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
