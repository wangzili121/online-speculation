#!/usr/bin/env python3
"""Check that retained draft Attention can replay the original packed history."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from validate_dsv41_pypto_packed_attention_vjp import make_fixture, reference


def make_capture(block, world, pages, model, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from blockspec.dsv41_packed_attention_tape import make_packed_attention_core_capture
    from blockspec.pypto_dsv41_packed_swa import make_packed_compressed_chain, make_packed_swa_chain

    heads, dim, topk, hidden = model.num_attention_heads // tp, model.head_dim, model.index_topk, model.hidden_size
    rows = block * heads
    fp8, fp8_scales = (pages, 128, 1, dim), (pages, 128, 1, dim // 32)
    fp4, fp4_scales = (pages, 128, 1, dim // 2), (pages, 128, 1, dim // 16)
    swa = pl.jit.inline(auto_scope=False)(make_packed_swa_chain(pages, heads, block)._func)
    sparse = pl.jit.inline(auto_scope=False)(make_packed_compressed_chain(pages, pages, block)._func)
    save = make_packed_attention_core_capture(block, 2, model, tp)

    @pl.jit
    def capture(
        normed: pl.Tensor[[block, hidden], pl.BF16], q: pl.Tensor[[block, heads * dim], pl.BF16],
        temp: pl.Tensor[[block, dim], pl.BF16], wp: pl.Tensor[fp8, pl.UINT8],
        ws: pl.Tensor[fp8_scales, pl.BF16], wl: pl.Tensor[[1, 256], pl.INT32],
        wr: pl.Tensor[[8, 16], pl.INT32], starts: pl.Tensor[[8], pl.INT32], spans: pl.Tensor[[2, 3], pl.INT32],
        bias: pl.Tensor[[1, 128], pl.FP32], cp: pl.Tensor[fp4, pl.UINT8], cs: pl.Tensor[fp4_scales, pl.BF16],
        cl: pl.Tensor[[1, 256], pl.INT32], ids: pl.Tensor[[block, topk], pl.INT32],
        lengths: pl.Tensor[[block], pl.INT32], mask: pl.Tensor[[block, topk], pl.FP32],
        sink: pl.Tensor[[heads], pl.FP32], zero: pl.Tensor[[128, dim], pl.BF16],
        saved_x: pl.InOut[pl.Tensor[[block * 2, hidden], pl.BF16]],
        saved_q: pl.InOut[pl.Tensor[[rows * 2, dim], pl.BF16]],
        saved_temp: pl.InOut[pl.Tensor[[block * 2, dim], pl.BF16]],
        saved_context: pl.InOut[pl.Tensor[[rows * 2, dim], pl.FP32]],
        saved_lse: pl.InOut[pl.Tensor[[block * 2, heads], pl.FP32]],
        saved_ids: pl.InOut[pl.Tensor[[block * 2, topk], pl.INT32]],
        saved_lengths: pl.InOut[pl.Tensor[[block * 2], pl.INT32]],
        sliding_only: pl.Scalar[pl.BOOL], slot: pl.Scalar[pl.INT32],
    ):
        context = pl.create_tensor([rows, dim], dtype=pl.FP32)
        lse = pl.create_tensor([rows, 1], dtype=pl.FP32)
        if sliding_only:
            swa(wp, ws, wl, wr, starts, spans, bias, q, temp, sink, zero, context, lse)
        else:
            sparse(wp, ws, wl, wr, starts, spans, bias, cp, cs, cl, ids, lengths, mask,
                   q, temp, sink, zero, context, lse)
        save(normed, q, temp, context, lse, ids, lengths, saved_x, saved_q, saved_temp, saved_context,
             saved_lse, saved_ids, saved_lengths, sliding_only, slot)

    @pl.jit.host
    def dispatch(
        normed: pl.Tensor[[world, block, hidden], pl.BF16], q: pl.Tensor[[world, block, heads * dim], pl.BF16],
        temp: pl.Tensor[[world, block, dim], pl.BF16], wp: pl.Tensor[[world, *fp8], pl.UINT8],
        ws: pl.Tensor[[world, *fp8_scales], pl.BF16], wl: pl.Tensor[[world, 1, 256], pl.INT32],
        wr: pl.Tensor[[world, 8, 16], pl.INT32], starts: pl.Tensor[[world, 8], pl.INT32],
        spans: pl.Tensor[[world, 2, 3], pl.INT32], bias: pl.Tensor[[world, 1, 128], pl.FP32],
        cp: pl.Tensor[[world, *fp4], pl.UINT8], cs: pl.Tensor[[world, *fp4_scales], pl.BF16],
        cl: pl.Tensor[[world, 1, 256], pl.INT32], ids: pl.Tensor[[world, block, topk], pl.INT32],
        lengths: pl.Tensor[[world, block], pl.INT32], mask: pl.Tensor[[world, block, topk], pl.FP32],
        sink: pl.Tensor[[world, heads], pl.FP32], zero: pl.Tensor[[world, 128, dim], pl.BF16],
        saved_x: pl.InOut[pl.Tensor[[world, block * 2, hidden], pl.BF16]],
        saved_q: pl.InOut[pl.Tensor[[world, rows * 2, dim], pl.BF16]],
        saved_temp: pl.InOut[pl.Tensor[[world, block * 2, dim], pl.BF16]],
        saved_context: pl.InOut[pl.Tensor[[world, rows * 2, dim], pl.FP32]],
        saved_lse: pl.InOut[pl.Tensor[[world, block * 2, heads], pl.FP32]],
        saved_ids: pl.InOut[pl.Tensor[[world, block * 2, topk], pl.INT32]],
        saved_lengths: pl.InOut[pl.Tensor[[world, block * 2], pl.INT32]],
        sliding_only: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            capture(normed[rank], q[rank], temp[rank], wp[rank], ws[rank], wl[rank], wr[rank], starts[rank],
                    spans[rank], bias[rank], cp[rank], cs[rank], cl[rank], ids[rank], lengths[rank], mask[rank],
                    sink[rank], zero[rank], saved_x[rank], saved_q[rank], saved_temp[rank], saved_context[rank],
                    saved_lse[rank], saved_ids[rank], saved_lengths[rank], sliding_only != 0, slot, device=rank)

    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=16)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only and args.checkpoint is None:
        parser.error("--checkpoint is required on device")

    import torch
    from blockspec.dsv41_packed_attention_tape import packed_core_tape_shapes
    from blockspec.dsv41_packed_attention_vjp import PackedAttentionVJP, make_packed_attention_vjp
    from blockspec.pypto_dsv41_packed_history import packed_history_lookup
    from blockspec.pypto_dsv41_packed_swa import packed_swa_controls
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE as tp
    from models.deepseek_v4_1_flash_a3.quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp4_cache, quantize_mxfp8_cache,
    )
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DeviceTensor, ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    devices = tuple(map(int, args.devices.split(",")))
    world, block, pages = len(devices), args.block, 5
    ranks = tuple(range(world))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    reverse = make_packed_attention_vjp(block, world=world, tp=tp, physical_pages=pages).compile(config=config)
    capture = make_capture(block, world, pages, model, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, world=world)))
        return
    dim, heads, topk = model.head_dim, model.num_attention_heads // tp, model.index_topk
    cases = []
    rng = torch.Generator().manual_seed(124853)
    with DistributedWorker([capture, reverse], config, persistent=True) as runtime:
        def stack(shards):
            return StackedDeviceTensor(shards, (world, *shards[0].shape), ranks)

        def upload(host):
            return stack([runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                          for rank, value in enumerate(host)])

        def shared(host):
            return upload([host] * world)

        def read(shards, rank):
            tensor = shards.shards[rank]
            host = torch.empty(tensor.shape, dtype=tensor.dtype)
            runtime.copy_from(host.data_ptr(), tensor.data_ptr, host.nbytes, worker_id=rank)
            return host

        page_table, anchor, active = (3, 0, 4, 1), 397, 7
        window_rows, starts, spans, bias = packed_swa_controls(page_table, anchor, pages)
        win_lookup, select_lookup = packed_history_lookup("fp8"), packed_history_lookup("fp4")
        old = torch.randn(pages, 128, 1, dim, generator=rng) * .5
        window_payload, window_scales = quantize_mxfp8_cache(old)
        selected_payload, selected_scales = quantize_mxfp4_cache(old, 16, "bf16")
        win_ref = dequantize_mxfp8_cache(window_payload, window_scales).reshape(-1, dim).bfloat16()
        select_ref = dequantize_mxfp4_cache(selected_payload, selected_scales, 16, "bf16").reshape(-1, dim).bfloat16()
        window_ref = win_ref[window_rows.flatten().long()]
        window_payload, selected_payload = window_payload.view(torch.uint8), selected_payload.view(torch.uint8)
        win_lookup, select_lookup = shared(win_lookup), shared(select_lookup)
        tape = {name: shared(torch.zeros(shape, dtype=getattr(torch, dtype)))
                for name, (shape, dtype) in packed_core_tape_shapes(block, 2, model, tp).items()}
        args.layer = 39
        fixture = make_fixture(block, heads, dim, args, min(world, tp))
        normed = torch.randn(block, model.hidden_size, generator=rng).bfloat16()
        sink = upload([fixture[rank % tp][2].flatten() for rank in ranks])
        outputs_seed = [torch.randn(block, heads, dim, generator=rng) * .07 for _ in ranks]
        incoming = upload([seed.flatten(0, 1) for seed in outputs_seed])
        q = upload([fixture[rank % tp][0].flatten(1) for rank in ranks])
        temp = upload([fixture[rank % tp][1] for rank in ranks])
        for ratio, layer, slot in ((0, 1, 0), (2, 14, 1), (1, 39, 1)):
            swa = ratio == 0
            selected_prefix = anchor // max(1, ratio)
            lengths = torch.tensor([0 if swa else (token * 23 + 1) for token in range(block)],
                                   dtype=torch.int32).clamp(max=selected_prefix)
            ids = torch.full((block, topk), -1, dtype=torch.int32)
            for token, n in enumerate(lengths.tolist()):
                positions = torch.randperm(selected_prefix, generator=rng)[:n]
                ids[token, :n] = torch.tensor(page_table, dtype=torch.int32)[positions // 128] * 128 + positions % 128
            mask = torch.where(ids >= 0, 0., -1e30)
            fields = [shared(val) for val in (window_rows, starts, spans, bias)]
            sources = (
                ("window_cache_pool", window_payload, layer),
                ("window_cache_scale_pool", window_scales, layer),
                ("compressed_cache_pool", selected_payload, 0 if swa else min(3, (layer - 2) // 6)),
                ("compressed_cache_scale_pool", selected_scales, 0 if swa else min(3, (layer - 2) // 6)),
            )
            views, banks = [[] for _ in sources], []
            for rank in ranks:
                bank = {name: value.shards[rank] for name, value in tape.items()}
                for (name, host, source), by_rank in zip(sources, views, strict=True):
                    shape = ((source + 1) * pages, *host.shape[1:])
                    slab = runtime.alloc_tensor(shape, host.dtype, worker_id=rank)
                    offset = source * host.nbytes
                    runtime.copy_to(slab.data_ptr, host.data_ptr(), host.nbytes, dst_offset=offset, worker_id=rank)
                    by_rank.append(DeviceTensor(slab.data_ptr + offset, tuple(host.shape), slab.dtype,
                                               buffer=slab.buffer, byte_offset=offset))
                    bank[name] = slab
                sink_host = fixture[rank % tp][2]
                slab = runtime.alloc_tensor((layer + 1, heads), sink_host.dtype, worker_id=rank)
                runtime.copy_to(slab.data_ptr, sink_host.data_ptr(), sink_host.nbytes,
                                dst_offset=layer * sink_host.nbytes, worker_id=rank)
                bank.update(draft_sink=slab, window_lookup=win_lookup.shards[rank],
                            compressed_lookup=select_lookup.shards[rank])
                banks.append(bank)
            wp, ws, cp, cs = map(stack, views)
            runtime.run(capture, shared(normed), q, temp, wp, ws, win_lookup, *fields,
                        cp, cs, select_lookup, shared(ids), shared(lengths), shared(mask), sink,
                        shared(torch.zeros(128, dim, dtype=torch.bfloat16)), *tape.values(), int(swa), slot, config=config)
            for name, host, source in sources:
                position = selected_prefix if name.startswith("compressed_") else anchor
                prefix_row = page_table[position // 128] * 128 + position % 128
                replacement = torch.zeros_like(host.flatten(0, 1)[prefix_row])
                offset = source * host.nbytes + prefix_row * replacement.nbytes
                for rank in ranks:
                    runtime.copy_to(banks[rank][name].data_ptr, replacement.data_ptr(), replacement.nbytes,
                                    dst_offset=offset, worker_id=rank)
            proposer = SimpleNamespace(runtime=runtime, config=config, ranks=banks, world=world, block=block,
                                       tp=tp, pages=pages, last_layers=2)
            vjp = PackedAttentionVJP(proposer, reverse, SimpleNamespace(num_hidden_layers=layer - slot + 2))
            student = SimpleNamespace(owner=0, rows=active, position=anchor, pages=page_table,
                                      ffn_tape={rank: {name: value.shards[rank] for name, value in tape.items()}
                                                for rank in ranks})
            seed = SimpleNamespace(owner=0, layer=layer, rows=active, context=incoming, parameter_gradients=None,
                                   residual=None)
            timings, first = [], None
            for _ in range(8):
                start = perf_counter()
                result = vjp(student, seed)
                timings.append((perf_counter() - start) * 1e3)
                gradients = (result.query, result.temporary, result.parameter_gradients["draft_sink"])
                observation = [[read(g, rank) for g in gradients] for rank in ranks]
                if first is None:
                    first = observation
                else:
                    for before, after in zip(first, observation, strict=True):
                        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                   for a, b in zip(before, after, strict=True))
            errors = []
            for rank in ranks:
                q_ref, temp_ref, sink_ref = fixture[rank % tp]
                context, lse, expected = reference(q_ref, temp_ref, window_ref, select_ref[ids.clamp(min=0).long()],
                                                   bias, mask, sink_ref, outputs_seed[rank], active)
                saved = slice(slot * block, (slot + 1) * block)
                assert torch.equal(read(tape["attention_tape_normed"], rank)[saved], normed)
                assert torch.equal(read(tape["attention_tape_lengths"], rank)[saved], lengths)
                if not swa:
                    assert torch.equal(read(tape["attention_tape_selection"], rank)[saved], ids)
                offset = slice(slot * block * heads, (slot + 1) * block * heads)
                assert torch.equal(read(tape["attention_tape_query"], rank)[offset], q_ref.flatten(0, 1))
                assert torch.equal(read(tape["attention_tape_temporary"], rank)[saved], temp_ref)
                torch.testing.assert_close(read(tape["attention_tape_context"], rank)[offset], context,
                                           atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(read(tape["attention_tape_lse"], rank)[saved].reshape(-1, 1), lse,
                                           atol=1e-5, rtol=0)
                relative = [float((g.reshape_as(ref).double() - ref).norm() / ref.norm())
                            for g, ref in zip(first[rank], expected, strict=True)]
                assert max(relative) < 3e-5, (swa, rank, relative)
                errors.append(relative)
            cases.append(dict(layer=layer, slot=slot, ratio=ratio, active=active, max_relative_l2=max(map(max, errors)),
                              post_capture_append=True, repeated_byte_exact=len(timings),
                              first_ms=timings[0], warm_median_ms=median(timings[1:])))
    print(json.dumps(dict(passed=True, block=block, world=world, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
