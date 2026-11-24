"""Compare exact C1A gradient statistics with/without fused device reduction."""

import argparse
import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter

import torch

from blockspec.pypto_dsv41_optimizer_programs import make_gradient_statistics, make_reduce, make_stats
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(41131)
    torch.set_num_threads(4)
    shapes = ((5120, 1280), (1280,), (1280, 8192), (5120, 512), (512,), (16,), (2, 1024, 4096), (2048, 5120))
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(dp=1, tp=1, ep=1)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids, draft.native = runtime, {0: 0}, native
        draft.compile_lock, draft.programs = Lock(), {}
        draft.output = args.output
        draft.output.mkdir()
        precreated = []
        for shape in shapes:
            values = torch.randn(shape) * .01
            runtime.pin(values, (0,), readonly=True, name=f"grad/{shape}")
            precreated.append(values)

        def run(mode, shape, value):
            if mode == "fused":
                sums = runtime.workspace((1, 8), torch.float32, 0, fill=None)
                flags = runtime.workspace((1, 8), torch.float32, 0, fill=None)
                draft.execute(("fused", shape), 0, lambda: make_gradient_statistics(shape),
                              dict(value=value, sums=sums, flags=flags))
                return sums, flags
            _, count = make_stats(shape)
            sums = runtime.workspace((count, 8), torch.float32, 0, fill=None)
            flags = runtime.workspace((count, 8), torch.float32, 0, fill=None)
            draft.execute(("stats", shape), 0, lambda: make_stats(shape)[0],
                          dict(value=value, sums=sums, flags=flags))
            while count > 1:
                _, next_count = make_reduce(count)
                out_sums = runtime.workspace((next_count, 8), torch.float32, 0, fill=None)
                out_flags = runtime.workspace((next_count, 8), torch.float32, 0, fill=None)
                draft.execute(("reduce", count), 0, lambda: make_reduce(count)[0],
                              dict(sums=sums, flags=flags, out_sums=out_sums, out_flags=out_flags))
                sums, flags, count = out_sums, out_flags, next_count
            return sums, flags

        def read(mode, shape, values):
            pair = tuple(runtime.read_output(tensor, 0) for tensor in run(mode, shape, values))
            runtime.release_values()
            return pair

        for shape, values in zip(shapes, precreated, strict=True):
            before, after = (read(mode, shape, values) for mode in ("old", "fused"))
            assert before[1].eq(1).all()
            assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True)), shape
        shape = shapes[0]
        bad = precreated[0].clone()
        bad[-1, -1] = float("inf")
        bad[-1, -2] = float("nan")
        runtime.pin(bad, (0,), readonly=True, name="grad/nonfinite")
        before, after = (read(mode, shape, bad) for mode in ("old", "fused"))
        assert not before[1].eq(1).all()
        assert torch.equal(before[1], after[1])
        torch.testing.assert_close(before[0], after[0], atol=0., rtol=0., equal_nan=True)

        samples = {"old": [], "fused": []}
        for _ in range(8):
            for mode in samples:
                started = perf_counter()
                for shape, values in zip(shapes, precreated, strict=True):
                    run(mode, shape, values)
                samples[mode].append((perf_counter() - started) * 1000)
                runtime.release_values()
    print(json.dumps(dict(passed=True, shapes=shapes, byte_exact=True, nonfinite="identical",
                          ms={mode: median(values) for mode, values in samples.items()})), flush=True)


if __name__ == "__main__":
    main()
