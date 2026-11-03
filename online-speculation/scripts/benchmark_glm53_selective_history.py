#!/usr/bin/env python3
"""比较完整快照和按需读取；输入含真实的四行池展开及历史尾部。"""

import argparse
from functools import partial
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from blockspec.vllm_ascend.glm53_history import PagedLatent, gather_history, read_pages
from blockspec.vllm_ascend.glm53_ops import kda_native
from benchmark_glm53_attention_paths import measure


def selected_read(cache, table, length, indices, mode):
    history = PagedLatent(cache, table, length, block_size=16, mode=mode)
    try:
        return history.gather(indices)
    finally:
        history.release()


def snapshot_read(cache, table, length, indices):
    return gather_history(read_pages(cache, table, length, block_size=16), indices)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(5313)
    results = []
    for pitch in (512, 576):
        cache = torch.randn(512, 128, 1, pitch, device="npu", dtype=torch.bfloat16)[..., :512]
        table = torch.randperm(4096).to(device="npu", dtype=torch.int32)[None]
        for width in (2, 8):
            for length in (257, 2055, 8195, 32771):
                complete = length // 4
                pools = torch.stack([torch.randperm(complete)[:512] for _ in range(width)])
                indices = (pools[:, :, None] * 4 + torch.arange(4)).flatten(1)
                indices = torch.cat((indices, torch.arange(complete * 4, length).expand(width, -1)), 1).to("npu")
                baseline = partial(snapshot_read, cache, table, length, indices)
                expected = baseline()
                timings = {}
                for mode in ("snapshot", "paged", "auto"):
                    fn = partial(selected_read, cache, table, length, indices, mode)
                    assert torch.equal(fn(), expected)
                    timings[mode] = measure(fn, args.iterations)
                results.append(dict(case="history", width=width, length=length, pitch=pitch,
                                    selected_rows=indices.numel(), baseline=measure(baseline, args.iterations),
                                    paths=timings, exact=True))
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    for heads in (8, 64):
        packed = torch.randn(8, 3, heads, 128, device="npu", dtype=torch.bfloat16) * .1
        q, k, v = packed.unbind(1)
        gate = torch.randn(8, heads, 128, device="npu", dtype=torch.bfloat16)
        beta = torch.randn(8, heads, device="npu", dtype=torch.bfloat16)
        initial = torch.randn(heads, 128, 128, device="npu") * .01
        a_log = torch.zeros(heads, device="npu")
        bias = torch.full((heads * 128,), -3., device="npu")
        def contiguous(q=q, k=k, v=v, gate=gate, beta=beta, initial=initial, a_log=a_log, bias=bias):
            return kda_native(q.contiguous(), k.contiguous(), v.contiguous(), gate, beta, initial, a_log, bias)
        direct = partial(kda_native, q, k, v, gate, beta, initial, a_log, bias)
        expected, actual = contiguous(), direct()
        for x, y in zip(expected, actual, strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        results.append(dict(case="kda_strides", heads=heads, q_stride=list(q.stride()),
                            contiguous=measure(contiguous, args.iterations),
                            direct=measure(direct, args.iterations), exact=True))
    args.output.write_text(json.dumps(dict(scope="local synthetic calls, not model throughput", results=results), indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
