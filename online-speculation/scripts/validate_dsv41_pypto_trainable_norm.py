"""Check DSV4.1 K32 trainable Q/KV RMSNorm VJP on a resident device."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from threading import Lock

import torch

from blockspec.pypto_dsv41_online_programs import make_rms
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(41131)
    torch.set_num_threads(2)
    rows = 32
    cases = []
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(dp=1, tp=1, ep=1)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids = runtime, {0: 0}
        draft.native = native
        draft.compile_lock, draft.programs = Lock(), {}
        draft.output = args.output
        draft.output.mkdir()
        for dimension in (512, 1280):
            for gamma_dtype in (torch.float32, torch.bfloat16):
                options = dict(epsilon=1e-20, gamma_bf16=gamma_dtype == torch.bfloat16)
                x = torch.randn(rows, dimension).bfloat16()
                x[0].zero_()
                x[1] *= 2. ** -16
                dy = torch.randn_like(x) * 0.125
                gamma = (torch.ones(1, dimension) + 0.1 * torch.randn(1, dimension)).to(gamma_dtype)
                for name, value in dict(x=x, gamma=gamma, dy=dy).items():
                    runtime.pin(value, (0,), readonly=True, name=f"{dimension}/{gamma_dtype}/{name}")
                program = {mode: (lambda mode=mode: make_rms(rows, dimension, mode=mode, **options))
                           for mode in ("forward", "input", "weight", "trainable", "frozen_input")}

                def call(mode, **inputs):
                    draft.execute((dimension, str(gamma_dtype), mode), 0, program[mode], inputs)

                def run(mode):
                    input_gradient = runtime.workspace(x.shape, x.dtype, 0, fill=None)
                    weight_gradient = runtime.workspace(gamma.shape, torch.float32, 0)
                    if mode == "old":
                        inverse = runtime.workspace((rows, 8), torch.float32, 0, fill=None)
                        replay = runtime.workspace(x.shape, x.dtype, 0, fill=None)
                        call("forward", x=x, gamma=gamma, y=replay, inv=inverse)
                        call("input", x=x, gamma=gamma, inv=inverse, dy=dy, dx=input_gradient)
                        call("weight", x=x, inv=inverse, dy=dy, dg=weight_gradient)
                    else:
                        call("trainable", x=x, gamma=gamma, dy=dy, dx=input_gradient, dg=weight_gradient)
                    return input_gradient, weight_gradient

                def read(mode):
                    result = tuple(runtime.read_output(value, 0).clone() for value in run(mode))
                    runtime.release_values()
                    return result

                old, new = read("old"), read("new")
                for a, b in zip(old, new, strict=True):
                    assert a.isfinite().all() and a.count_nonzero()
                    assert torch.equal(a, b), (dimension, gamma_dtype, (a.float() - b.float()).abs().max().item())
                frozen = runtime.workspace(x.shape, x.dtype, 0, fill=None)
                call("frozen_input", x=x, gamma=gamma, dy=dy, dx=frozen)
                assert torch.equal(old[0], runtime.read_output(frozen, 0))
                runtime.release_values()
                samples = {"old": [], "new": []}
                for _ in range(15):
                    for mode in samples:
                        start = perf_counter()
                        run(mode)
                        samples[mode].append((perf_counter() - start) * 1e3)
                        runtime.release_values()
                cases.append(dict(width=dimension, gamma_dtype=str(gamma_dtype), byte_exact=True,
                                  ms={mode: median(values) for mode, values in samples.items()}))
    print(json.dumps(dict(passed=True, iterations=15, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
