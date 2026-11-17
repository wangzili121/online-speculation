#!/usr/bin/env python3
"""Check distributed distillation through the frozen LM head and final RMSNorm."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.pypto_glm53_online_suffix import make_distillation_suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="use the deployed GLM LM head and final RMSNorm")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or args.output.exists():
        parser.error("use distinct devices and a fresh output path")

    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    torch.set_num_threads(4)
    from models.glm5_3_flash.config import FLASH

    ranks, rows, width, vocab, streams = len(args.devices), 16, FLASH.hidden_size, FLASH.vocab_size // 8, 4
    epsilon = FLASH.rms_norm_eps
    stride = (vocab + 63) // 64 * 64
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=args.devices, num_sub_workers=0))
    entry = make_distillation_suffix(ranks, rows, vocab, width, epsilon, streams)
    student = torch.empty((ranks, rows, stride))
    teacher = torch.empty_like(student)
    weight = torch.empty((ranks, stride, width), dtype=torch.bfloat16)
    activation = torch.empty((ranks, rows, width), dtype=torch.bfloat16)
    scale = torch.empty((ranks, width), dtype=torch.bfloat16)
    dlogit = torch.zeros_like(student)
    local = torch.zeros((ranks, rows, width))
    combined = torch.zeros_like(local)
    loss = torch.empty((ranks, rows, 8))
    gradient = torch.empty((ranks, rows * streams, width))
    tensors = (student, teacher, weight, activation, scale, dlogit, local, combined, loss, gradient)
    report = {"ranks": ranks, "checkpoint_weights": args.checkpoint is not None,
              "compiled": False, "executed": False, "cases": []}
    worker = None
    handles = []
    try:
        program = entry.compile(*tensors, 5, 1, config=config)
        report["compiled"] = True
        if args.compile_only:
            return
        rng = torch.Generator().manual_seed(29)
        activation[0].normal_(generator=rng)
        if args.checkpoint is None:
            weight.normal_(std=0.02, generator=rng)
            scale[0].uniform_(0.8, 1.2, generator=rng)
        else:
            from models.glm5_3_flash.utils import Checkpoint

            checkpoint = Checkpoint(args.checkpoint)
            weight.zero_()
            for rank in range(ranks):
                weight[rank, :vocab].copy_(checkpoint.read("lm_head.weight", slice(rank * vocab, (rank + 1) * vocab)))
            roots = ("model.language_model.", "model.")
            name = next((root + "norm.weight" for root in roots if root + "norm.weight" in checkpoint), None)
            if name is None:
                raise ValueError("GLM checkpoint does not contain the final RMSNorm")
            scale[0].copy_(checkpoint.read(name))
        activation.copy_(activation[0].clone().expand_as(activation))
        scale.copy_(scale[0].clone().expand_as(scale))
        student.zero_()
        teacher.zero_()
        loss.zero_()
        gradient.zero_()
        worker = DistributedWorker([program], config=config, persistent=True, reset_persistent_windows=False)
        for tensor in tensors:
            handles.append(worker.alloc_stacked_tensor(tensor.share_memory_()))
        for epoch, active in enumerate((5, 3), start=1):
            student.normal_(generator=rng)
            teacher.normal_(generator=rng)
            student[0, 0, 31] = 18
            teacher[-1, 0, 32] = 19
            student[..., vocab:] = 75
            teacher[..., vocab:] = 90
            for data, resident in zip(tensors[:2], handles[:2], strict=True):
                for rank, shard in enumerate(resident.shards):
                    worker.copy_to(shard.data_ptr, data[rank].data_ptr(), data[rank].nbytes, worker_id=rank)
            worker.run(program, *handles, active, epoch, config=config)
            for data, resident in zip(tensors[-2:], handles[-2:], strict=True):
                for rank, shard in enumerate(resident.shards):
                    worker.copy_from(data[rank].data_ptr(), shard.data_ptr, data[rank].nbytes, worker_id=rank)

            def logsoftmax(x):
                return x[:, :active, :vocab].transpose(0, 1).reshape(active, -1).double().log_softmax(-1)

            a, b = logsoftmax(student), logsoftmax(teacher)
            logits_grad = ((a.exp() - b.exp()) / active).reshape(active, ranks, vocab)
            upstream = torch.zeros((active, width), dtype=torch.float64)
            for rank in range(ranks):
                upstream.add_(logits_grad[:, rank] @ weight[rank, :vocab].double())
            x = activation[0, :active].double().requires_grad_()
            normalized = x * (x.square().mean(-1, keepdim=True) + epsilon).rsqrt() * scale[0].double()
            (dx,) = torch.autograd.grad(normalized, x, upstream)
            expected = dx.repeat_interleave(streams, dim=0) / streams
            expected = expected.float().expand(ranks, -1, -1)
            torch.testing.assert_close(gradient[:, :active * streams], expected, atol=2e-7, rtol=7e-4)
            torch.testing.assert_close(loss[:, :active, 0].sum(0), ((b - a) * b.exp()).sum(-1).float() / active,
                                       atol=3e-5, rtol=2e-5)
            if gradient[:, active * streams:].count_nonzero() or loss[:, active:].count_nonzero():
                raise AssertionError("inactive rows reached the activation gradient")
            report["cases"].append({"epoch": epoch, "rows": active, "max_abs_gradient":
                                   float((gradient[:, :active * streams] - expected).abs().max())})
        report["executed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                try:
                    for resident in reversed(handles):
                        worker.free_stacked_tensor(resident)
                finally:
                    worker.close()
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
