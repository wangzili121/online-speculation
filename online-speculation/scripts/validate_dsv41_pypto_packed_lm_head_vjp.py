#!/usr/bin/env python3
"""Check the 1122 resident loss and its TP-sharded LM head input VJP."""

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    tp, world = args.tp, len(devices)
    if tp not in (2, 4) or world % tp or len(set(devices)) != world:
        parser.error("Use distinct device IDs with one or more full TP2 or TP4 groups")
    if args.repeats < 2:
        parser.error("Use at least two repetitions per case")

    import torch
    from blockspec.dsv41_packed_lm_head_vjp import PackedLMHeadVJP, make_packed_lm_head_vjp
    from blockspec.dsv41_packed_loss import PackedLoss, make_packed_loss_graph
    from blockspec.dsv41_packed_proposer import PackedStudentRows
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    rows, vocabulary, hidden = 16, 129280, 5120
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    vjp = make_packed_lm_head_vjp(rows, vocabulary, hidden, world // tp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, devices=devices, tp=tp)))
        return
    loss_graph = make_packed_loss_graph(rows, vocabulary, world // tp, tp, "tv").compile(config=config)
    generator = torch.Generator().manual_seed(5281)
    if args.checkpoint:
        from safetensors import safe_open

        index = json.loads((args.checkpoint / "quant_model_weights.safetensors.index.json").read_text())
        location = index["weight_map"]["head.weight"]
        with safe_open(args.checkpoint / location, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor("head.weight")
        assert weight.dtype == torch.bfloat16 and weight.shape == (vocabulary, hidden)
    else:
        weight = torch.randn(vocabulary, hidden, dtype=torch.bfloat16, generator=generator)

    cases = []
    with DistributedWorker([loss_graph, vjp], config, persistent=True, reset_persistent_windows=False) as runtime:
        student = torch.randn(rows, vocabulary, generator=generator)
        shards = weight.chunk(tp, dim=0)
        ranks = [
            dict(lm_head_weight=runtime.alloc_tensor(shards[rank % tp].shape, weight.dtype,
                                                     init=shards[rank % tp], worker_id=rank),
                 logits=runtime.alloc_tensor(student.shape, student.dtype, init=student, worker_id=rank))
            for rank in range(world)
        ]
        proposer = SimpleNamespace(runtime=runtime, config=config, world=world, tp=tp, ranks=ranks)
        print("Resident LM head shards uploaded", file=sys.stderr, flush=True)
        loss = PackedLoss(proposer, loss_graph)
        reverse = PackedLMHeadVJP(proposer, loss, vjp)
        for owner, valid in [(0, 1), (world - tp, 7), (0, 7)]:
            teacher = student[:valid] + .22 * torch.randn((valid, vocabulary), generator=generator)
            result = loss(PackedStudentRows(owner, 5, valid, ranks[owner]["logits"]), teacher)
            print(f"LM head VJP owner={owner}, rows={valid}", file=sys.stderr, flush=True)
            samples = [reverse(result) for _ in range(args.repeats)]
            actual = torch.empty((rows, hidden), dtype=torch.float32)
            runtime.copy_from(actual.data_ptr(), samples[-1].gradient.data_ptr, actual.nbytes, worker_id=owner)
            grad = torch.empty((rows, vocabulary), dtype=torch.bfloat16)
            runtime.copy_from(grad.data_ptr(), result.gradient.data_ptr, grad.nbytes, worker_id=owner)
            expected = torch.zeros((valid, hidden), dtype=torch.float64)
            for i in range(0, vocabulary, 1280):
                expected.add_(grad[:valid, i:i + 1280].double() @ weight[i:i + 1280].double())
            assert not torch.any(actual[valid:])
            delta = actual[:valid].double() - expected
            rel_l2 = float(delta.norm() / expected.norm())
            assert rel_l2 <= 1e-4, (owner, valid, rel_l2, float(delta.abs().max()))
            warm = [sample.seconds * 1000 for sample in samples[1:]]
            cases.append(dict(owner=owner, rows=valid, relative_l2=rel_l2, max_abs=float(delta.abs().max()),
                              warm_median_ms=median(warm), warm_max_ms=max(warm), warm_samples_ms=warm))
    print(json.dumps(dict(passed=True, compiled=True, executed=True, devices=devices, tp=tp, rows=rows,
                          vocabulary=vocabulary, hidden=hidden, checkpoint=str(args.checkpoint or "synthetic"),
                          cases=cases)), flush=True)


if __name__ == "__main__":
    main()
