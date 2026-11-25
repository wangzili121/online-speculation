#!/usr/bin/env python3
"""Validate the fused, rank-local SWA, Indexer and Reuse draft sublayers."""

import argparse
from dataclasses import asdict
import json
from math import frexp
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_draft import (
    pack_draft_input_projection, pack_draft_output_projection, prepare_output,
)
from blockspec.pypto_dsv41_packed_draft_attention import (
    make_packed_reuse_draft_layer, make_packed_scored_draft_layer, make_packed_swa_draft_layer,
)
from blockspec.pypto_dsv41_packed_indexer import packed_index_controls
from validate_dsv41_pypto_packed_indexer import attention_fixture, attention_reference, fixtures
from validate_dsv41_pypto_packed_qkv import checkpoint_operands


def compare_bf16_contribution(actual, expected):
    """Gate the complete sublayer with CANN BF16 and report the stricter Wo-only comparison."""
    actual, expected = actual.float(), expected.float()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Draft output or golden contains NaN/Inf")
    error = (actual - expected).abs()
    max_abs, worst = error.flatten().max(dim=0)
    # CANN uses the ULP of the narrowed golden where the largest error occurs.
    rounded = expected.bfloat16().float()
    narrowed = rounded.flatten()[worst].abs().item()
    _, exponent = frexp(max(narrowed, 2.0 ** -126))
    ulp = 2.0 ** (exponent - 8)
    cann_limit = max(0.1, 32 * ulp)
    cann_matches = (error <= (1 + expected.abs()) / 128).float().mean().item()
    wo_error = (actual - rounded).abs()
    wo_tolerance = 0.003 + rounded.abs() / 128
    wo_outliers = wo_error > wo_tolerance
    relative_l2 = (wo_error.norm() / rounded.norm().clamp_min(1e-30)).item()
    return dict(
        passed=relative_l2 <= 0.005 and cann_matches >= 0.99 and max_abs.item() <= cann_limit,
        relative_l2=relative_l2, max_abs=max_abs.item(),
        native_wo_tolerance_passed=not bool(wo_outliers.any()),
        native_wo_outliers=wo_outliers.sum().item(), native_wo_worst_ratio=(wo_error / wo_tolerance).max().item(),
        elements=error.numel(), cann_bf16_matched_ratio=cann_matches, cann_bf16_abs_limit=cann_limit,
    )


def reference(hidden_rows, norm, operands, wo, rope, history, selected, sink, anchor):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import golden_o_proj
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
    from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm

    block = selected.shape[0]
    heads = FLASH.num_attention_heads // TP_SIZE
    cosine, sine = rope
    window, compressed = history
    x = golden_rms_norm(hidden_rows[:block, 0].bfloat16(), norm)
    qr = golden_rms_norm((x.float() @ operands["wq_a"].float()).bfloat16(), operands["q_norm"])
    query = (qr.float() @ operands["wq_b"].float()).bfloat16().view(block, heads, FLASH.head_dim)
    query = golden_rope_tail(query, cosine[:block], sine[:block])
    kv = golden_rms_norm((x.float() @ operands["wkv"].float()).bfloat16(), operands["kv_norm"])
    kv = golden_rope_tail(kv, cosine[:block], sine[:block])
    context, _ = attention_reference(query, kv, window, compressed, selected, sink, anchor)
    return golden_o_proj(context.bfloat16(), cosine[:block], sine[:block],
                         wo["wo_a"].transpose(-2, -1), wo["wo_b"], output_dtype=torch.float32)


def validate(args):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    torch.set_num_threads(3)
    config = RunConfig(
        backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device, distributed_config=DistributedConfig(device_ids=[args.device]),
    )
    embedding, operands = checkpoint_operands(args.checkpoint, args.layer, args.tp_rank)
    checkpoint = Checkpoint(args.checkpoint)
    prefix = f"layers.{args.layer}."
    wo_weights = {name: checkpoint.tensor(prefix + "attn." + name) for name in ("wo_a.weight", "wo_b.weight")}
    wo = prepare_output(wo_weights, asdict(FLASH), tp_rank=args.tp_rank, tp=TP_SIZE)
    packed_wo = pack_draft_output_projection(wo)
    packed_qkv = pack_draft_input_projection(operands)
    norm = checkpoint.tensor(prefix + "attn_norm.weight")
    pages = 16
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(5319)
    cases = []
    for block, ratio, anchor in ((8, 2, 257), (16, 1, 643), (32, 2, 1091)):
        capacity = max(16, block)
        common, _ = fixtures(block, pages, generator)
        _, _, idx_wq, idx_weight, cosine, sine, index_payload, index_scales = common
        banks, (_, _, sink, zeros), _, (_, window, compressed) = attention_fixture(
            block, pages, table, anchor, generator,
        )
        x_hc = embedding[:capacity].float()[:, None].expand(-1, FLASH.hc_mult, -1).contiguous()
        pre_mix = torch.full((capacity, FLASH.hc_mult), 1 / FLASH.hc_mult)
        bucket, index_table, request_ids, visible = packed_index_controls(table, anchor, ratio, block)
        rows = torch.empty((block, FLASH.index_topk), dtype=torch.int32)
        lengths = torch.empty((block,), dtype=torch.int32)
        bias = torch.empty_like(rows, dtype=torch.float32)
        contribution = torch.empty((block, FLASH.hidden_size), dtype=torch.bfloat16)
        projection = (x_hc, pre_mix, norm, packed_qkv, operands["q_norm"], operands["wq_b"], operands["kv_norm"])
        output_weights = (sink, zeros, packed_wo["wo_a"], packed_wo["wo_b"])
        selection = (rows, lengths, bias)
        reused = torch.empty_like(contribution)
        next_contribution = torch.empty_like(contribution)
        swa_contribution = torch.empty_like(contribution)
        next_hidden = x_hc.roll(1, dims=0)
        calls = dict(
            scored=(
                make_packed_scored_draft_layer(pages, pages, bucket, ratio, block),
                (*projection, idx_wq, idx_weight, cosine, sine, index_payload, index_scales, index_table,
                 request_ids, visible, *banks, *output_weights),
                (*selection, contribution),
            ),
            reuse=(
                make_packed_reuse_draft_layer(pages, pages, block),
                (*projection, cosine, sine, *banks, *selection, *output_weights),
                (reused,),
            ),
            swa=(
                make_packed_swa_draft_layer(pages, block),
                (*projection, cosine, sine, *banks[:7], *output_weights),
                (swa_contribution,),
            ),
        )
        programs = {
            role: entry.compile(*inputs, *outputs, config=config)
            for role, (entry, inputs, outputs) in calls.items()
        }
        with ChipWorker(config) as worker:
            handles = {}
            for _, inputs, outputs in calls.values():
                for tensor in (*inputs, *outputs):
                    if id(tensor) not in handles:
                        handles[id(tensor)] = worker.alloc_tensor(tensor.shape, tensor.dtype)
            for tensor in calls["scored"][1]:
                worker.copy_to(handles[id(tensor)].data_ptr, tensor.data_ptr(), tensor.nbytes)

            timings = {}
            for role, (_, inputs, outputs) in calls.items():
                arguments = [handles[id(tensor)] for tensor in (*inputs, *outputs)]
                worker.run(programs[role], *arguments, config=config)
                samples = []
                for _ in range(args.iterations):
                    start = perf_counter()
                    worker.run(programs[role], *arguments, config=config)
                    samples.append(perf_counter() - start)
                timings[role] = median(samples) * 1e6

            # Keep the selected rows on the device across both calls.
            next_input = worker.alloc_tensor(next_hidden.shape, next_hidden.dtype)
            next_output = worker.alloc_tensor(next_contribution.shape, next_contribution.dtype)
            worker.copy_to(next_input.data_ptr, next_hidden.data_ptr(), next_hidden.nbytes)
            worker.run(programs["reuse"], next_input, *(
                handles[id(tensor)] for tensor in calls["reuse"][1][1:]
            ), next_output, config=config)

            for _, _, outputs in calls.values():
                for tensor in outputs:
                    worker.copy_from(tensor.data_ptr(), handles[id(tensor)].data_ptr, tensor.nbytes)
            worker.copy_from(next_contribution.data_ptr(), next_output.data_ptr, next_contribution.nbytes)

            for original in (banks[0], banks[1], banks[7], banks[8], index_payload, index_scales):
                observed = torch.empty_like(original)
                worker.copy_from(observed.data_ptr(), handles[id(original)].data_ptr, observed.nbytes)
                torch.testing.assert_close(observed.view(torch.uint8), original.view(torch.uint8), rtol=0, atol=0)

        torch.testing.assert_close(contribution, reused, rtol=0, atol=0)
        committed = anchor // ratio
        count = min(committed, FLASH.index_topk)
        torch.testing.assert_close(lengths, torch.full_like(lengths, count), atol=0, rtol=0)
        logical = torch.argsort(table)[rows[:, :count] // 128] * 128 + rows[:, :count] % 128
        assert bool(((logical >= 0) & (logical < committed)).all())
        assert bool((rows[:, count:] == -1).all())
        # The packed Indexer fixture independently tests exact selected sets.
        # Use the selected rows here to isolate the connected projection path.
        precision = {}
        for role, result, hidden_rows, selected in (
            ("scored", contribution, x_hc, rows),
            ("reuse_next", next_contribution, next_hidden, rows),
            ("swa", swa_contribution, x_hc, rows[:, :0]),
        ):
            expected = reference(
                hidden_rows, norm, operands, wo, (cosine, sine), (window, compressed), selected, sink, anchor,
            )
            precision[role] = compare_bf16_contribution(result, expected)
        case = dict(block=block, ratio=ratio, anchor=anchor, precision=precision,
                    relative_l2={role: stats["relative_l2"] for role, stats in precision.items()},
                    resident_call_median_us=timings, same_input_reuse_byte_exact=True, caches_byte_exact=True)
        print(json.dumps(case), flush=True)
        cases.append(case)
    passed = all(stats["passed"] for case in cases for stats in case["precision"].values())
    return dict(passed=passed, npu_execution=True, layer=args.layer, tp_rank=args.tp_rank, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args)
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
