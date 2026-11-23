#!/usr/bin/env python3
"""Validate device-to-device inactive Orthrus bank publication on one NPU."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
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
        receipt = bank.stage_resident(state, 1)
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
