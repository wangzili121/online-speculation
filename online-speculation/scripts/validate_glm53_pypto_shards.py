#!/usr/bin/env python3
"""Verify full-vocabulary distillation across resident PyPTO vocabulary shards."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.pypto_glm53_online_shards import make_sharded_kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or args.output.exists():
        parser.error("use distinct devices and a fresh output path")

    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    torch.set_num_threads(4)
    ranks, rows, vocabulary = len(args.devices), 16, 19360
    extent = (vocabulary + 63) // 64 * 64
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=args.devices, num_sub_workers=0))
    kernel = make_sharded_kl(ranks, rows, vocabulary)
    report = {"ranks": ranks, "words_per_rank": vocabulary, "rows": rows, "compiled": False,
              "executed": False, "passed": False, "cases": []}
    generator = torch.Generator().manual_seed(148)
    logits_a = torch.empty((ranks, rows, extent), dtype=torch.float32).share_memory_()
    logits_b = torch.empty_like(logits_a).share_memory_()
    gradient = torch.empty_like(logits_a).share_memory_()
    losses = torch.empty((ranks, rows, 8), dtype=torch.float32).share_memory_()
    handles = []
    worker = None
    try:
        program = kernel.compile(logits_a, logits_b, gradient, losses, 5, 1, config=config)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        worker = DistributedWorker([program], config=config, persistent=True, reset_persistent_windows=False)
        for tensor in (logits_a, logits_b, gradient, losses):
            handles.append(worker.alloc_stacked_tensor(tensor.zero_()))
        for epoch, active in enumerate((5, 3, 5), 1):
            logits_a.normal_(generator=generator)
            logits_b.normal_(generator=generator)
            logits_a[..., vocabulary:].fill_(100)
            logits_b[..., vocabulary:].fill_(200)
            logits_a[0, 0, vocabulary - 1] = 62 + epoch
            logits_b[-1, 0, 0] = 58 + epoch
            logits_a[:, 1].zero_()
            logits_b[:, 1].zero_()
            for data, resident in zip((logits_a, logits_b), handles[:2], strict=True):
                for rank, shard in enumerate(resident.shards):
                    worker.copy_to(shard.data_ptr, data[rank].data_ptr(), data[rank].nbytes, worker_id=rank)
            worker.run(program, *handles, active, epoch, config=config)
            for data, resident in zip((gradient, losses), handles[2:], strict=True):
                for rank, shard in enumerate(resident.shards):
                    worker.copy_from(data[rank].data_ptr(), shard.data_ptr, data[rank].nbytes, worker_id=rank)

            def normalized(t):
                merged = t[:, :, :vocabulary].transpose(0, 1).reshape(rows, ranks * vocabulary).double()
                return merged.log_softmax(1).reshape(rows, ranks, vocabulary).transpose(0, 1)

            a, b = normalized(logits_a), normalized(logits_b)
            expected_grad = (a.exp() - b.exp()) / active
            expected_loss = (b.exp() * (b - a)).sum(-1) / active
            expected_grad[:, active:].zero_()
            expected_loss[:, active:].zero_()
            torch.testing.assert_close(gradient[:, :, :vocabulary], expected_grad.float(), atol=2e-7, rtol=2e-4)
            torch.testing.assert_close(losses[..., 0], expected_loss.float(), atol=3e-5, rtol=2e-5)
            if gradient[..., vocabulary:].count_nonzero():
                raise AssertionError("padding was included in distributed distillation")
            report["cases"].append({"epoch": epoch, "rows": active,
                                    "gradient_max_abs": float((gradient[:, :, :vocabulary] - expected_grad).abs().max()),
                                    "loss_max_abs": float((losses[..., 0] - expected_loss).abs().max())})
        report.update(passed=True, executed=True)
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
