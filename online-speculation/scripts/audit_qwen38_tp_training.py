#!/usr/bin/env python3
"""Compare gathered/sharded losses and small-gradient collectives on real TP devices."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from audit_qwen38_execution import measure
from blockspec.losses import divergence
from blockspec.vllm_ascend.qwen38_loss import vocab_divergence
from blockspec.vllm_ascend.qwen38_weights import TensorParallel, _Gather


def metrics(actual, expected):
    delta = actual.detach().float() - expected.detach().float()
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=248320)
    args = parser.parse_args()
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        tp = TensorParallel(dist.group.WORLD, tuple(range(size)))
        width = (args.vocab_size + size - 1) // size
        start = min(rank * width, args.vocab_size)
        end = min((rank + 1) * width, args.vocab_size)
        results = []
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(731)
            full = torch.randn(31, width * size, device="npu", dtype=dtype)
            teacher = torch.randn(31, args.vocab_size, device="npu", dtype=dtype)
            local = full[:, rank * width:(rank + 1) * width].clone().requires_grad_()
            weights = torch.randn(31, device="npu")
            for kind in ("forward_kl", "tv"):
                def gathered(local=local, teacher=teacher, kind=kind, weights=weights):
                    local.grad = None
                    loss = divergence(_Gather.apply(local, tp)[:, :args.vocab_size], teacher, kind)
                    loss.backward(weights)
                    return loss

                def sharded(local=local, teacher=teacher, kind=kind, weights=weights):
                    local.grad = None
                    loss = vocab_divergence(local[:, :end - start], teacher, start=start, tp=tp, kind=kind)
                    loss.backward(weights)
                    return loss

                reference = gathered().detach()
                reference_grad = local.grad.clone()
                actual = sharded().detach()
                errors = dict(loss=metrics(actual, reference), gradient=metrics(local.grad, reference_grad))
                if errors["loss"]["relative_l2"] > 1e-4 or errors["gradient"]["relative_l2"] > .001:
                    raise AssertionError(errors)
                row = dict(dtype=str(dtype), kind=kind, errors=errors,
                    gathered=measure(gathered, args.repeats, torch.npu.synchronize),
                    sharded=measure(sharded, args.repeats, torch.npu.synchronize))
                row["peak_temporary_bytes"] = {}
                for name, operation in (("gathered", gathered), ("sharded", sharded)):
                    local.grad = None
                    torch.npu.synchronize()
                    baseline = torch.npu.memory_allocated()
                    torch.npu.reset_peak_memory_stats()
                    operation()
                    torch.npu.synchronize()
                    row["peak_temporary_bytes"][name] = torch.npu.max_memory_allocated() - baseline
                results.append(row)
        torch.manual_seed(731 + rank)
        gradients = [torch.randn(dim, device="npu") for dim in [256] * 32 + [128] * 48]

        def separate():
            return [tp.reduce(g.clone()) for g in gradients]

        def bucket():
            packed = tp.reduce(torch.cat(gradients))
            return list(packed.split([g.numel() for g in gradients]))

        expected, actual = separate(), bucket()
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        report = dict(rank=rank, tp=size, vocab_size=args.vocab_size, losses=results,
                      norm_gradients=dict(tensors=len(gradients), numerical_pass=True,
                        errors=metrics(torch.cat(actual), torch.cat(expected)),
                        separate=measure(separate, args.repeats, torch.npu.synchronize),
                        bucket=measure(bucket, args.repeats, torch.npu.synchronize)), passed=True)
        args.output.with_name(f"{args.output.stem}-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
