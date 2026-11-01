#!/usr/bin/env python3
"""GLM-5.3-Flash draft primitives on one NPU; no checkpoint required."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.vllm_ascend.glm53_ops import (
    finite_error, kda_native, kda_reference, select_history, sparse_draft_attention,
)


def initialize_npu_runtime():
    from vllm_ascend.utils import enable_custom_op

    # AscendC 算子延迟注册，必须先绑定可见设备，再加载原生算子库。
    torch.npu.set_device(0)
    if not enable_custom_op():
        raise RuntimeError("native Ascend operators are unavailable after Ascend initialization")
    from vllm_ascend.ops.kda import run_recurrent_kda

    if not callable(run_recurrent_kda):
        raise RuntimeError("native recurrent KDA interface is unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("mla", "kda", "all"), default="all")
    args = parser.parse_args()
    import torch_npu
    import vllm_ascend.ops  # noqa: F401

    initialize_npu_runtime()
    torch.manual_seed(154)
    results = {}
    def rand(*shape):
        return (torch.randn(*shape, device="npu", dtype=torch.bfloat16) * .15).contiguous()

    for width in ((2, 8, 32) if args.case != "kda" else ()):
        q, history, block = rand(width, 8, 256), rand(131, 512), rand(width, 512)
        iq, iw, pools = rand(width, 32, 128), rand(width, 32), rand(32, 128)
        indices = select_history(iq, iw, pools, 131, topk=128)
        weights = dict(ar_key=rand(8, 256, 512), ar_value=rand(8, 256, 512),
                       draft_key=rand(8, 256, 512), draft_value=rand(8, 256, 512))
        before = history.clone()
        actual = sparse_draft_attention(q, history, block, indices, **weights, backend="npu")
        expected = sparse_draft_attention(q.float(), history.float(), block.float(), indices,
                                         **{k: v.float() for k, v in weights.items()})
        torch.testing.assert_close(history, before, rtol=0, atol=0)
        error = finite_error(actual, expected)
        if error["relative_rms"] > .04:
            raise AssertionError(f"MLA BF16 error: {error}")
        results[f"mla_k{width}"] = error
    for width in ((1, 2, 8) if args.case != "mla" else ()):
        q, k, v, gate = [rand(width, 8, 128) for _ in range(4)]
        beta = rand(width, 8)
        initial = torch.randn(8, 128, 128, device="npu") * .02
        a_log = torch.zeros(8, device="npu")
        bias = torch.zeros(8 * 128, device="npu")
        before = initial.clone()
        expected, expected_states = kda_reference(q, k, v, gate, beta, initial, a_log, bias)
        actual, states = kda_native(q, k, v, gate, beta, initial, a_log, bias)
        torch.testing.assert_close(initial, before, rtol=0, atol=0)
        error = finite_error(actual, expected)
        state_error = finite_error(states, expected_states)
        if error["relative_rms"] > .04 or state_error["relative_rms"] > .04:
            raise AssertionError(f"KDA error: {error}, {state_error}")
        results[f"kda_k{width}"] = dict(output=error, checkpoints=state_error)
    results["scope"] = "synthetic primitive tests, not full-model Orthrus acceptance"
    results["torch_npu"] = torch_npu.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
