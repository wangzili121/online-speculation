#!/usr/bin/env python3
"""Compare the packed Attention VJP to FP64 autograd on a shared latent KV block."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def reference(q, temporary, window, selected, history_bias, selected_bias, sink, cotangent, active):
    import torch

    block, heads, dim = q.shape
    q = q.double().requires_grad_(True)
    temporary = temporary.double().requires_grad_(True)
    sink = sink.double().requires_grad_(True)
    kv = torch.cat((window.double().expand(block, -1, -1),
                    selected.double(),
                    temporary.expand(block, -1, -1)), dim=1)
    bias = torch.cat((history_bias.expand(block, -1), selected_bias, torch.zeros(block, block)), dim=-1)
    logits = (q @ kv.transpose(-2, -1)) * dim ** -0.5 + bias.double().unsqueeze(1)
    scores = torch.cat((logits, sink.reshape(1, heads, 1).expand(block, -1, -1)), -1)
    normalized = scores.softmax(-1)
    context = normalized[..., :-1] @ kv
    lse = torch.logsumexp(scores, dim=-1)
    loss = (context[:active] * cotangent[:active].double()).sum()
    grads = torch.autograd.grad(loss, (q, temporary, sink))
    return context.detach().float().flatten(0, 1), lse.detach().float().reshape(-1, 1), grads


def make_fixture(block, heads, dim, args, lanes):
    import torch
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
    from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    stem = f"layers.{args.layer}.attn."
    generator = torch.Generator().manual_seed(624051)
    x = weights.lookup_rows(torch.arange(451, 451 + block)).contiguous()
    angles = torch.randn(block, FLASH.qk_rope_head_dim // 2, generator=generator)
    cosine, sine = angles.cos(), angles.sin()
    operands = {}
    for name in ("wq_a", "wkv"):
        payload, scale = weights.projection(stem + name)
        operands[name] = (payload.float() * scale).bfloat16()
    low = golden_rms_norm((x.float() @ operands["wq_a"].float()).bfloat16(),
                          checkpoint.tensor(stem + "q_norm.weight"))
    temporary = golden_rms_norm((x.float() @ operands["wkv"].float()).bfloat16(),
                                checkpoint.tensor(stem + "kv_norm.weight"))
    temporary = golden_rope_tail(temporary, cosine, sine)
    all_sinks = checkpoint.tensor(stem + "attn_sink").float()
    result = []
    for lane in range(lanes):
        shard = slice(lane * heads * dim, (lane + 1) * heads * dim)
        payload, scale = weights.projection(stem + "wq_b", output_slice=shard)
        qb = (payload.float() * scale).bfloat16()
        q = (low.float() @ qb.float()).bfloat16().reshape(block, heads, dim)
        q = golden_rope_tail(q, cosine, sine)
        sink = all_sinks[lane * heads:(lane + 1) * heads].reshape(1, heads)
        result.append((q, temporary, sink))
    return result


def make_forward(block, world, heads, dim):
    """Run the production SWA and compressed merge on resident, gathered KV."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from blockspec.pypto_dsv41_sparse import TOPK, make_sparse_attention
    from blockspec.pypto_dsv41_swa import WINDOW, make_swa_attention

    rows = block * heads
    swa = pl.jit.inline(make_swa_attention(heads, token_major_query=True, block_size=block)._func)
    sparse = pl.jit.inline(auto_scope=False)(make_sparse_attention(
        head_major_swa=True, block_size=block, gathered_selection=True,
    )._func)

    @pl.jit
    def attend(
        q: pl.Tensor[[rows, dim], pl.BF16],
        temp: pl.Tensor[[block, dim], pl.BF16],
        window: pl.Tensor[[WINDOW, dim], pl.BF16],
        selected: pl.Tensor[[block * TOPK, dim], pl.BF16],
        window_bias: pl.Tensor[[1, WINDOW], pl.FP32],
        selected_bias: pl.Tensor[[block, TOPK], pl.FP32],
        lengths: pl.Tensor[[block], pl.INT32],
        sink: pl.Tensor[[1, heads], pl.FP32],
        context: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
        lse: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
        spans: pl.Tensor[[2, 3], pl.INT32],
        zeros: pl.Tensor[[WINDOW, dim], pl.BF16],
        ids: pl.Tensor[[block, TOPK], pl.INT32],
    ):
        local = pl.create_tensor([rows, dim], dtype=pl.FP32)
        local_lse = pl.create_tensor([rows, 1], dtype=pl.FP32)
        query = pl.reshape(q, [block, heads * dim])
        sink_logits = pl.reshape(sink, [heads])
        local, local_lse = swa(query, window, spans, temp, window_bias, sink_logits, zeros, local, local_lse)
        return sparse(q, selected, ids, lengths, selected_bias, zeros, local, local_lse, context, lse)

    @pl.jit.host
    def dispatch(
        q: pl.Tensor[[world, rows, dim], pl.BF16],
        temp: pl.Tensor[[world, block, dim], pl.BF16],
        window: pl.Tensor[[world, WINDOW, dim], pl.BF16],
        selected: pl.Tensor[[world, block * TOPK, dim], pl.BF16],
        window_bias: pl.Tensor[[world, 1, WINDOW], pl.FP32],
        selected_bias: pl.Tensor[[world, block, TOPK], pl.FP32],
        lengths: pl.Tensor[[world, block], pl.INT32],
        sink: pl.Tensor[[world, 1, heads], pl.FP32],
        context: pl.Out[pl.Tensor[[world, rows, dim], pl.FP32]],
        lse: pl.Out[pl.Tensor[[world, rows, 1], pl.FP32]],
        spans: pl.Tensor[[world, 2, 3], pl.INT32],
        zeros: pl.Tensor[[world, WINDOW, dim], pl.BF16],
        ids: pl.Tensor[[world, block, TOPK], pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            attend(q[rank], temp[rank], window[rank], selected[rank], window_bias[rank], selected_bias[rank],
                   lengths[rank], sink[rank], context[rank], lse[rank], spans[rank], zeros[rank], ids[rank],
                   device=rank)
    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = list(map(int, args.devices.split(",")))
    if len(devices) not in (1, 4, 8, 16) or len(set(devices)) != len(devices):
        parser.error("Specify one rank or complete TP4 groups")
    if not args.compile_only and args.checkpoint is None:
        parser.error("--checkpoint is required on device")
    if args.iterations < 2:
        parser.error("--iterations must include a warm sample")

    import torch
    from blockspec.dsv41_packed_attention_vjp import make_packed_attention_vjp
    from blockspec.pypto_dsv41_sparse import HEAD_DIM, HEAD_TILE, TOPK, WINDOW
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    block, world, heads, dim = args.block, len(devices), HEAD_TILE, HEAD_DIM
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_attention_vjp(block, world=world).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, block=block, world=world, compiled=True)))
        return

    forward = make_forward(block, world, heads, dim).compile(config=config)
    rng = torch.Generator().manual_seed(523987)
    ranks = tuple(range(world))
    shared = make_fixture(block, heads, dim, args, min(world, 4))
    generator_geometry = (
        (block, 128, (0, 1, 127, 128, 129, 255, 256, 385, 511, 512)),
        (7, 73, (512, 1, 127, 128, 511, 0, 129)),
        (1, 0, (0,)),
    )
    cases = []
    with DistributedWorker([forward, graph], config, persistent=True) as runtime:
        def upload(values):
            handles = [runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                       for rank, value in enumerate(values)]
            return StackedDeviceTensor(handles, (world, *values[0].shape), ranks)

        def read(shards, rank):
            tensor = shards.shards[rank]
            host = torch.empty(tensor.shape, dtype=tensor.dtype)
            runtime.copy_from(host.data_ptr(), tensor.data_ptr, host.nbytes, worker_id=rank)
            return host

        metadata = [
            upload([torch.tensor([[0, 0, WINDOW], [0, 0, 0]], dtype=torch.int32)] * world),
            upload([torch.zeros(WINDOW, dim, dtype=torch.bfloat16)] * world),
            upload([torch.zeros(block, TOPK, dtype=torch.int32)] * world),
        ]
        for active, history, counts in generator_geometry:
            compressed_counts = torch.tensor(
                [counts[token % len(counts)] for token in range(block)], dtype=torch.int32,
            )
            history_mask = torch.where(torch.arange(WINDOW)[None, :] < history, 0., -1e30)
            selected_mask = torch.where(torch.arange(TOPK)[None, :] < compressed_counts[:, None], 0., -1e30)
            feeds, expected, forward_expected = [], [], []
            for rank in ranks:
                q, temp, sink = shared[rank % len(shared)]
                window = torch.randn(WINDOW, dim, generator=rng).bfloat16() * .5
                selected = torch.randn(block, TOPK, dim, generator=rng).bfloat16() * .5
                incoming = torch.randn(block, heads, dim, generator=rng) * .07
                context, lse, grads = reference(q, temp, window, selected, history_mask, selected_mask,
                                                sink, incoming, active)
                feeds.append((q.flatten(0, 1), temp, window, selected.flatten(0, 1), history_mask, selected_mask,
                              compressed_counts, torch.empty_like(context), torch.empty_like(lse),
                              sink, incoming.flatten(0, 1)))
                expected.append(grads)
                forward_expected.append((context, lse))
            inputs = [upload(tensors) for tensors in zip(*feeds, strict=True)]
            shapes = ((block * heads, dim), (block, dim), (1, heads))
            outputs = [upload([torch.empty(shape)] * world) for shape in shapes]
            owner = world - 4 if world > 4 and active == 7 else 0
            runtime.run(forward, *inputs[:7], inputs[9], *inputs[7:9], *metadata, config=config)
            times = []
            first = None
            live_ranks = range(owner, min(owner + 4, world))
            for _ in range(args.iterations):
                start = perf_counter()
                runtime.run(graph, *inputs, *outputs, owner, active, config=config)
                times.append((perf_counter() - start) * 1000)
                observed = [[read(value, rank) for value in outputs] for rank in live_ranks]
                if first is None:
                    first = observed
                else:
                    for rank, (before, after) in enumerate(zip(first, observed, strict=True)):
                        for label, a, b in zip(("query", "temporary_kv", "sink"), before, after, strict=True):
                            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                                label, active, rank, float((a - b).abs().max()),
                            )
            metrics = []
            for rank, observed in zip(live_ranks, first, strict=True):
                assert torch.equal(read(inputs[4], rank), history_mask)
                assert torch.equal(read(inputs[5], rank), selected_mask)
                errors = {}
                for label, flat, ref in zip(("query", "temporary_kv", "sink"), observed, expected[rank], strict=True):
                    actual = flat.reshape_as(ref)
                    error = actual.double() - ref
                    relative = float(error.norm() / ref.norm().clamp_min(1e-30))
                    assert relative < 3e-5, (label, rank, active, relative, float(error.abs().max()))
                    errors[label] = dict(relative_l2=relative, max_abs=float(error.abs().max()))
                    if label == "query":
                        assert torch.count_nonzero(actual[active:]) == 0
                if active < block:
                    assert torch.count_nonzero(observed[1][active:]) > 0
                ref_context, ref_lse = forward_expected[rank]
                ctx_l2 = float((read(inputs[7], rank) - ref_context).norm() / ref_context.norm())
                lse_max_abs = float((read(inputs[8], rank) - ref_lse).abs().max())
                assert ctx_l2 < 2e-5 and lse_max_abs < 1e-5, (rank, ctx_l2, lse_max_abs)
                errors["forward"] = dict(relative_l2=ctx_l2, lse_max_abs=lse_max_abs)
                metrics.append(errors)
            cases.append(dict(active_rows=active, history_rows=history, unique_selections=sorted(set(counts)),
                              owner=owner, repeated_byte_exact=args.iterations,
                              resident_call_median_ms=median(times[1:]), lanes=metrics))
            for stack in (*inputs, *outputs):
                for rank, allocation in enumerate(stack.shards):
                    runtime.free_tensor(allocation, worker_id=rank)
    print(json.dumps(dict(passed=True, block=block, world=world, layer=args.layer, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
