#!/usr/bin/env python3
"""Compile the EP16 reverse stages or verify frozen W8A8 gradients over local HCCL."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter
from types import SimpleNamespace


def exercise(config, programs, model, block, last_layers, tp, checkpoint, *, host_exchange):
    import torch
    from blockspec.dsv41_online_profile import profile_online
    from blockspec.dsv41_packed_expert_vjp import packed_expert_vjp_reference
    from blockspec.dsv41_packed_ffn_tape import packed_ffn_tape_shapes
    from blockspec.dsv41_staged_ep_vjp import StagedExpertVJP
    from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.runtime import StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker

    rng = torch.Generator().manual_seed(2527)
    ranks = range(2)
    layer = 39
    local = block // tp
    hidden, topk = model.hidden_size, model.num_experts_per_tok
    experts = ResidentWeights(Checkpoint(checkpoint))
    actual_ids = (0, 3, 247, 381) if topk == 2 else (0, 3, 65, 129, 247, 280, 342, 381)
    experts_per_rank = len(actual_ids) // 2
    weights = []
    for expert in actual_ids:
        stem = f"layers.{layer}.ffn.experts.{expert}"
        projections = (experts.projection(f"{stem}.{name}", output_major=True) for name in ("w1", "w3", "w2"))
        weights.append(tuple(item for pair in projections for item in pair))
    matrices = tuple(
        torch.stack(parts).reshape(2, experts_per_rank, *parts[0].shape[-2:]) if index % 2 == 0
        else torch.stack(parts).reshape(2, experts_per_rank, -1)
        for index, parts in enumerate(zip(*weights, strict=True))
    )
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, model.hc_mult)
    slot = last_layers - 1
    counts = Counter()
    stages = Counter()
    profile = SimpleNamespace(current=dict(counts=counts, stage_seconds=stages), lock=Lock())

    def exchange_local(tag, payload):
        if payload is not None:
            return dict(collective=payload["collective"], group=payload["group"], packets=[])

    plan = SimpleNamespace(local_ranks=ranks, counts=(2,), tp=tp, node=0, groups=lambda _: ((0, 1),))
    link = SimpleNamespace(node=0, exchange=exchange_local)

    def transport_context(runtime, dispatcher):
        if not host_exchange:
            return ClusterCollectives(dispatcher, plan, link)

        def exchange(stage, transfers):
            for field in zip(*transfers, strict=True):
                partitions = []
                for rank, (send, _, _) in enumerate(field):
                    value = torch.empty(send.shape, dtype=send.dtype)
                    runtime.copy_from(value.data_ptr(), send.data_ptr, value.nbytes, worker_id=rank)
                    partitions.append(value.chunk(2))
                for rank, (_, receive, _) in enumerate(field):
                    value = torch.cat([source[rank] for source in partitions]).contiguous()
                    runtime.copy_to(receive.data_ptr, value.data_ptr(), value.nbytes, worker_id=rank)

        return nullcontext(SimpleNamespace(
            plan=plan, mappings=SimpleNamespace(runtime=runtime, register=lambda *_: None),
            exchange_resident=exchange,
        ))

    cases = []
    with (
        DistributedWorker(programs, config, persistent=True, reset_persistent_windows=False) as runtime,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        dispatcher = SimpleNamespace(devices=list(ranks), map_ranks=lambda fn: list(pool.map(fn, ranks)))
        with transport_context(runtime, dispatcher) as transport:
            def upload(value, rank):
                return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

            def read(handle, rank):
                value = torch.empty(handle.shape, dtype=handle.dtype)
                runtime.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes, worker_id=rank)
                return value

            banks = [{name: upload(torch.zeros(shape, dtype=getattr(torch, dtype)), rank)
                      for name, (shape, dtype) in shapes.items()} for rank in ranks]
            op = StagedExpertVJP(runtime, transport, programs, banks, model, block=block, last_layers=last_layers)
            resident = [StackedDeviceTensor(tuple(upload(value[r], r) for r in ranks), value.shape, ranks)
                        for value in matrices]
            with profile_online(profile):
                for owner, live in ((0, block - 3), (2 - tp, 1), (0, 7)):
                    x = torch.randn((block, hidden), dtype=torch.bfloat16, generator=rng)
                    dy = torch.randn((block, hidden), generator=rng) * .2
                    columns = [torch.full((block,), experts_per_rank), torch.arange(block) % 2]
                    if topk == 6:
                        columns.extend(torch.full((block,), expert) for expert in (2, 5, 6, 7))
                    ids = torch.stack(columns, dim=1).int()
                    factors = torch.rand(block, topk, generator=rng) * .6
                    factors[0, 0] = 0
                    saved = {}
                    for rank in range(owner, owner + tp):
                        lane = rank - owner
                        tape = {name: torch.zeros(shape, dtype=getattr(torch, dtype))
                                for name, (shape, dtype) in shapes.items()}
                        row = slot * (2 * block + local) + 2 * block
                        tape["ffn_tape_vectors"][row:row + local] = x[lane::tp]
                        span = slice(slot * local, (slot + 1) * local)
                        tape["ffn_tape_route_ids"][span, :topk] = ids[lane::tp]
                        tape["ffn_tape_route_weights"][span, :topk] = factors[lane::tp]
                        saved[rank] = {name: upload(value, rank) for name, value in tape.items()}
                    student = SimpleNamespace(owner=owner, ffn_tape=saved)
                    grad = SimpleNamespace(owner=owner, layer=layer, rows=live, value=upload(dy, owner))
                    expected_dx = torch.zeros_like(dy)
                    expected_route = torch.zeros(block, topk)
                    contributions = torch.zeros(block, topk, hidden)
                    for expert, matrices_ref in enumerate(weights):
                        rows, choices = (ids[:live] == expert).nonzero(as_tuple=True)
                        if rows.numel():
                            _, values, route = packed_expert_vjp_reference(
                                x[rows], dy[rows], factors[rows, choices, None], *matrices_ref,
                            )
                            contributions[rows, choices] = values
                            expected_route[rows, choices] = route[:, 0]
                    for choice in range(topk):
                        expected_dx += contributions[:, choice]
                    duration, prior = [], None
                    for _ in range(4):
                        before = perf_counter()
                        result = op.run_layer("fixture", owner=owner, layer=layer, rows=live,
                                              weights=resident, student=student, gradient=grad)
                        duration.append(perf_counter() - before)
                        device = tuple(
                            (read(result.local_gradient.shards[r], r), read(result.gradients.input.shards[r], r),
                             read(result.gradients.route.shards[r], r))
                            for r in range(owner, owner + tp)
                        )
                        if prior is not None:
                            for previous, current in zip(prior, device, strict=True):
                                for old, new in zip(previous, current, strict=True):
                                    torch.testing.assert_close(new, old, atol=0, rtol=0)
                        prior = device
                    errors = []
                    for lane, (seed, dx, dr) in enumerate(device):
                        want_seed = dy.clone()
                        want_seed[live:].zero_()
                        torch.testing.assert_close(seed, want_seed[lane::tp], atol=0, rtol=0)
                        assert not bool(dr[:, topk:].any())
                        for actual, expected in ((dx, expected_dx[lane::tp]), (dr[:, :topk], expected_route[lane::tp])):
                            valid = torch.arange(lane, block, tp) < live
                            assert not bool(actual[~valid].any())
                            rel = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
                            assert rel < .006, (owner, lane, live, rel)
                            errors.append(rel)
                    cases.append(dict(owner=owner, live_rows=live, max_relative_l2=max(errors),
                                      warm_median_ms=median(duration[1:]) * 1000, repeat_exact=True))
    return dict(cases=cases, checkpoint_experts=actual_ids, topk=topk,
                stage_seconds=stages, logical_hccl_counts=counts,
                exchange="host-permutation" if host_exchange else "resident-hccl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=16)
    parser.add_argument("--last-layers", type=int, choices=range(1, 27), default=4, metavar="N")
    parser.add_argument("--fixture-tp", type=int, choices=(1, 2), default=1, help="TP size for the two-card numerical check")
    parser.add_argument("--fixture-topk", type=int, choices=(2, 6), default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--host-exchange", action="store_true",
                        help="Run the kernels on NPU with the AllToAll permutation performed on CPU")
    args = parser.parse_args()
    if not args.compile_only and args.checkpoint is None:
        parser.error("Pass a checkpoint for the two-card validation")

    import torch
    from blockspec.dsv41_staged_ep_vjp import make_staged_expert_vjp
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    torch.set_num_threads(3)
    world, local, tp = (16, 8, 4) if args.compile_only else (2, 2, args.fixture_tp)
    model = FLASH if args.compile_only else replace(
        FLASH, n_routed_experts=4 if args.fixture_topk == 2 else 8, num_experts_per_tok=args.fixture_topk,
    )
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=list(range(local)), num_sub_workers=0))
    entries = make_staged_expert_vjp(model, args.block, args.last_layers, world=world, local_ranks=local, tp=tp)
    start = perf_counter()
    programs = [entry.compile(config=config) for entry in entries]
    report = dict(passed=True, block=args.block, last_layers=args.last_layers, tp=tp,
                  world=world, local_ranks=local, npu_execution=not args.compile_only,
                  compile_seconds=perf_counter() - start)
    if not args.compile_only:
        report.update(exercise(config, programs, model, args.block, args.last_layers, tp, args.checkpoint,
                               host_exchange=args.host_exchange))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
