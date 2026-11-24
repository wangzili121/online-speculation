"""Verify that C1A weight gradients overwrite every byte of dirty device scratch."""

import argparse
import json
from pathlib import Path
from threading import Lock

import torch

from blockspec.pypto_dsv41_online_programs import make_linear
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(41131)
    torch.set_num_threads(2)
    geometries = (
        ("wo_b", 2048, 5120, 1),
        ("wo_a", 1024, 4096, 2),
        ("wq_b", 1280, 8192, 1),
        ("wq_a", 5120, 1280, 1),
        ("wkv", 5120, 512, 1),
    )
    checked = {}
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(dp=1, tp=1, ep=1)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids, draft.native = runtime, {0: 0}, native
        draft.compile_lock, draft.programs = Lock(), {}
        draft.output = args.output
        draft.output.mkdir()
        for name, inputs, outputs, batches in geometries:
            a = (torch.randn(32 * batches, inputs) * 0.125).bfloat16()
            b = (torch.randn(32 * batches, outputs) * 0.125).bfloat16()
            shape = (batches * inputs, outputs)
            for label, value in (("a", a), ("b", b)):
                runtime.pin(value, (0,), readonly=True, name=f"{name}/{label}")

            def run(destination):
                draft.execute(
                    (name,), 0,
                    lambda: make_linear(
                        (32, inputs), (32, outputs), transpose_a=True,
                        transpose_b=False, gradient_weight=True, batches=batches,
                    ),
                    dict(a=a, b=b, output=destination),
                )
                return runtime.read_output(destination, 0)

            baseline = runtime.workspace(shape, torch.float32, 0)
            expected = run(baseline)
            assert expected.isfinite().all() and expected.count_nonzero()
            bytes_count = baseline.numel() * baseline.element_size()
            runtime.release_values()
            candidate = runtime.workspace(shape, torch.float32, 0, fill=None)
            allocation = runtime.tensor(candidate, 0)
            runtime._device_fill(allocation.data_ptr, bytes_count, 255, 0)
            runtime.mark_workspace_written(candidate, 0)
            assert runtime.read_output(candidate, 0).isnan().all(), name
            assert torch.equal(expected, run(candidate)), name
            checked[name] = bytes_count
            runtime.release_values()
    print(json.dumps(dict(passed=True, overwritten_bytes=checked, byte_exact=True)), flush=True)


if __name__ == "__main__":
    main()
