#!/usr/bin/env python3
"""Validate device-to-device inactive Orthrus bank publication."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def compare_publication(runtime, ranks):
    import torch
    from blockspec.pypto_dsv41_runtime import MutableDraftHeadBank, ResidentHeadState

    shape = (8192, 1280)
    names = ("wq_a", "wq_b", "wo_b")
    rank_ids = {rank: rank for rank in range(ranks)}
    initial = {
        (rank, 39): {name: torch.ones(shape, dtype=torch.bfloat16) for name in names}
        for rank in rank_ids
    }
    bank = MutableDraftHeadBank(runtime, rank_ids, initial, (39,))
    operands = {
        (rank, 39): {name: runtime.workspace(shape, torch.bfloat16, rank) for name in names}
        for rank in rank_ids
    }
    state = ResidentHeadState(
        operands, tuple(f"layers.39.attn.{name}.weight" for name in names), "a" * 64,
    )
    expected_bytes = ranks * len(names) * 2 * shape[0] * shape[1]
    parallel = runtime.map_ranks

    def serial(function):
        return [function(rank) for rank in rank_ids]

    modes = {"serial": serial, "parallel": parallel}
    timings = {name: [] for name in modes}
    host_counters = ("h2d_bytes", "d2h_bytes")
    host_before = [runtime.stats[name] for name in host_counters]
    version = 0
    try:
        for _ in range(15):
            for name, dispatch in modes.items():
                runtime.map_ranks = dispatch
                version += 1
                copied = runtime.stats["d2d_bytes"]
                started = perf_counter()
                bank.stage_resident(state, version)
                timings[name].append((perf_counter() - started) * 1000)
                assert runtime.stats["d2d_bytes"] - copied == expected_bytes
                assert bank.active_version == version - 1
                bank.activate(version)
                bank.commit(version)
    finally:
        runtime.map_ranks = parallel
    assert [runtime.stats[name] for name in host_counters] == host_before
    for rank in rank_ids:
        for name in names:
            actual = runtime.read_output(bank.host[bank.active_slot, rank, 39, name], rank)
            assert actual.eq(0).all()
    return dict(active_version=version, d2d_bytes_per_version=expected_bytes, host_round_trip=False,
                serial_ms=median(timings["serial"][2:]), parallel_ms=median(timings["parallel"][2:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", type=int, default=0)
    devices.add_argument("--compare-ranks", type=int, choices=(2, 4), help="Time publication on this many visible cards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")

    import torch
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime
    from blockspec.pypto_dsv41_runtime import MutableDraftHeadBank, ResidentHeadState

    torch.manual_seed(411)
    shape = (64, 128)
    report = {
        "format": "orthrus-dsv41-resident-publication-v1",
        "passed": False,
        "device": args.device,
        "shape": list(shape),
    }
    if args.compare_ranks is not None:
        ranks = args.compare_ranks
        report = {"format": report["format"], "passed": False, "ranks": ranks}
        with ResidentNetworkRuntime(list(range(ranks)), dict(dp=ranks, tp=1, ep=ranks)) as runtime:
            report.update(compare_publication(runtime, ranks), passed=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.output.chmod(0o644)
        print(json.dumps(report), flush=True)
        return
    runtime = None
    try:
        runtime = ResidentNetworkRuntime(
            (args.device,), {"dp": 1, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1},
        )
        initial = {(0, 39): {"weight": torch.ones(shape, dtype=torch.bfloat16)}}
        bank = MutableDraftHeadBank(runtime, {0: 0}, initial, (39,))
        # ``workspace`` is device-authoritative.  Staging it therefore forces
        # ResidentNetworkRuntime.copy_tensor down the D2D path rather than its
        # host-only shortcut.
        candidate = runtime.workspace(shape, torch.bfloat16, 0)
        expected = torch.zeros(shape, dtype=torch.bfloat16)
        digest = hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest()
        state = ResidentHeadState(
            {(0, 39): {"weight": candidate}},
            ("layers.39.attn.weight",), digest,
        )
        before = runtime.stats["d2d_bytes"]
        receipt = bank.stage_resident(state, 1)
        if runtime.stats["d2d_bytes"] - before != expected.nbytes:
            raise AssertionError("staging did not copy the complete device candidate")
        if bank.active_version != 0:
            raise AssertionError("staging changed the active bank")
        bank.activate(1)
        actual = runtime.read_output(bank.host[bank.active_slot, 0, 39, "weight"], 0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        bank.commit(1)
        report.update(
            passed=True,
            digest=receipt.digest,
            active_version=bank.active_version,
            d2d_bytes=runtime.summary()["d2d_bytes"],
            host_round_trip=False,
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.output.chmod(0o644)


if __name__ == "__main__":
    main()
