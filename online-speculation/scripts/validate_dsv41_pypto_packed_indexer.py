#!/usr/bin/env python3
"""Check draft Top-K against Torch using read-only packed A3 index pages."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
import json

import torch

from blockspec.pypto_dsv41_packed_indexer import make_packed_draft_indexer, packed_index_controls
from blockspec.pypto_dsv41_sparse import TOPK


def fixtures(block, physical_pages, generator):
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
    from models.deepseek_v4_1_flash_a3.decode_indexer import golden_index_query_fp4
    from models.deepseek_v4_1_flash_a3.quant import dequantize_mxfp4_cache, quantize_mxfp4_cache

    def sample(*shape):
        # Binary fractions keep the projection comparison independent of
        # differences in GEMM accumulation order at an FP4 rounding threshold.
        return (torch.randint(-2, 3, shape, generator=generator).float() / 16).bfloat16()

    capacity = max(16, block)
    heads, dim = FLASH.index_n_heads, FLASH.index_head_dim
    hidden, rank = FLASH.hidden_size, FLASH.q_lora_rank
    x, latent = sample(capacity, hidden), sample(capacity, rank)
    wq, ww = sample(rank, heads * dim), sample(hidden, heads)
    cosine = torch.zeros(capacity, FLASH.qk_rope_head_dim // 2)
    sine = torch.ones_like(cosine)
    keys = torch.randn(physical_pages, 128, 1, dim, generator=generator)
    payload, scales = quantize_mxfp4_cache(keys, 32, "e8m0", round_to_even=True)
    decoded = dequantize_mxfp4_cache(payload, scales, 32, "e8m0").bfloat16().flatten(0, 1)[:, 0]
    raw = (latent[:block].float() @ wq.float()).bfloat16().view(block, heads, dim)
    q = golden_index_query_fp4(golden_rope_tail(raw, cosine[:block], sine[:block]))
    weights = ((x[:block].float() @ ww.float()).bfloat16().float() * (heads * dim) ** -0.5).bfloat16()
    return (x, latent, wq, ww, cosine, sine, payload.view(torch.uint8), scales), (q, weights, decoded)


def attention_fixture(block, pages, table, anchor, generator):
    from models.deepseek_v4_1_flash_a3.quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp4_cache, quantize_mxfp8_cache,
    )

    from blockspec.pypto_dsv41_packed_history import packed_history_lookup
    from blockspec.pypto_dsv41_packed_swa import packed_swa_controls
    from blockspec.pypto_dsv41_sparse import HEAD_TILE
    from blockspec.pypto_dsv41_swa import HEAD_DIM, WINDOW
    from blockspec.vllm_ascend.dsv41_attention import PagedRows

    history = torch.randn((pages, WINDOW, 1, HEAD_DIM), generator=generator)
    code, scale = quantize_mxfp8_cache(history)
    long_code, long_scale = quantize_mxfp4_cache(history.flip(0), 16, "bf16", round_to_even=True)
    window_table = table.flip(0)
    window = dequantize_mxfp8_cache(code, scale).squeeze(-2).bfloat16()
    compressed = dequantize_mxfp4_cache(long_code, long_scale, 16, "bf16").squeeze(-2).bfloat16()
    query = torch.randn(block, HEAD_TILE, HEAD_DIM, generator=generator).bfloat16()
    temporary = torch.zeros(max(16, block), HEAD_DIM, dtype=torch.bfloat16)
    temporary[:block] = torch.randn(block, HEAD_DIM, generator=generator).bfloat16()
    sink = torch.linspace(-4, 4, HEAD_TILE)
    zeros = torch.zeros(WINDOW, HEAD_DIM, dtype=torch.bfloat16)
    fields = (code.view(torch.uint8), scale, packed_history_lookup("fp8"),
              *packed_swa_controls(window_table, anchor, pages),
              long_code.view(torch.uint8), long_scale, packed_history_lookup("fp4"))
    banks = (PagedRows(window, window_table[None]), PagedRows(compressed, torch.arange(pages)[None]))
    output = torch.empty(block * HEAD_TILE, HEAD_DIM, dtype=torch.float32)
    lse = torch.empty(block * HEAD_TILE, 1, dtype=torch.float32)
    return fields, (query.flatten(1), temporary, sink, zeros), (output, lse), (query, *banks)


def attention_reference(query, temporary, window, compressed, rows, sink, anchor):
    """Use FP64 accumulation across the same history, block and sink."""
    from blockspec.vllm_ascend.dsv4_hybrid import tail_logical_indices

    block, heads, dim = query.shape
    logical = tail_logical_indices(torch.tensor([anchor]), 128)
    local = window.gather(logical)[0].expand(block, -1, -1)
    long = compressed.gather(rows[None])[0]
    current = temporary[:block].expand(block, -1, -1)
    kv = torch.cat((local, long, current), dim=1).double()
    valid = torch.cat(((logical >= 0).expand(block, -1), rows >= 0,
                       torch.ones((block, block), dtype=torch.bool)), dim=1)
    scores = torch.einsum("khd,krd->khr", query.double(), kv) * dim ** -0.5
    scores = scores.masked_fill(~valid[:, None], -torch.inf)
    scores = torch.cat((scores, sink.double().reshape(1, heads, 1).expand(block, -1, -1)), dim=-1)
    probabilities = scores.softmax(dim=-1)[..., :-1]
    result = torch.einsum("khr,krd->khd", probabilities, kv)
    return result.float(), scores.logsumexp(dim=-1).float()


def validate(args):
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    from blockspec.pypto_dsv41_packed_draft_attention import make_packed_indexed_attention
    from blockspec.pypto_dsv41_packed_swa import make_packed_compressed_chain

    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device, distributed_config=DistributedConfig(device_ids=[args.device]))
    pages = 128 if args.extended_context else 16
    scenarios = ((1, (8192, 16384)), (2, (8193, 16384))) if args.extended_context else (
        (1, (0, 129, 643)), (2, (257, 1091)),
    )
    generator = torch.Generator().manual_seed(41401)
    table = torch.randperm(pages, generator=generator, dtype=torch.int32)
    zero_page = int(torch.where(table == 0)[0][0])
    table[zero_page] = table[0]
    table[0] = 0
    torch.set_num_threads(3)
    reports = []
    index_programs, attention_programs, fused_programs = {}, {}, {}
    for block in args.blocks:
        common, (queries, weights, decoded) = fixtures(block, pages, generator)
        for ratio, anchors in scenarios:
            for anchor in anchors:
                bucket, page_table, request_ids, visible = packed_index_controls(table, anchor, ratio, block)
                rows = torch.empty((block, TOPK), dtype=torch.int32)
                lengths = torch.empty((block,), dtype=torch.int32)
                bias = torch.empty((block, TOPK), dtype=torch.float32)
                inputs = (*common, page_table, request_ids, visible)
                selection = (rows, lengths, bias)
                key = block, bucket, ratio if bucket > TOPK else 0
                if key not in index_programs:
                    index_programs[key] = make_packed_draft_indexer(pages, bucket, ratio, block).compile(
                        *inputs, *selection, config=config,
                    )
                indexer = index_programs[key]
                if args.fused_attention:
                    banks, projections, output, reference = attention_fixture(block, pages, table, anchor, generator)
                    if key not in fused_programs:
                        fused_programs[key] = make_packed_indexed_attention(pages, pages, bucket, ratio, block).compile(
                            *inputs, *banks, *projections, *selection, *output, config=config,
                        )
                    if block not in attention_programs:
                        attention_programs[block] = make_packed_compressed_chain(pages, pages, block).compile(
                            *banks, *selection, *projections, *output, config=config,
                        )
                    program, attention = fused_programs[key], attention_programs[block]
                    input_values, output_values = (*inputs, *banks, *projections), (*selection, *output)
                else:
                    program = indexer
                    input_values, output_values = inputs, selection
                values = (*input_values, *output_values)
                if args.compile_only:
                    reports.append(dict(block=block, ratio=ratio, anchor=anchor, compiled=True,
                                        fused_attention=args.fused_attention))
                    continue
                split_times = []
                with ChipWorker(config) as worker:
                    handles = [worker.alloc_tensor(tuple(value.shape), value.dtype) for value in values]
                    device_inputs = handles[:len(input_values)]
                    device_outputs = handles[len(input_values):]
                    for value, handle in zip(input_values, device_inputs, strict=True):
                        worker.copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)
                    if args.fused_attention:
                        index_handles = device_inputs[:len(inputs)]
                        cut = len(inputs) + len(banks)
                        attention_handles = (*device_inputs[len(inputs):cut], *device_outputs[:3],
                                             *device_inputs[cut:], *device_outputs[3:])
                        worker.run(indexer, *index_handles, *device_outputs[:3], config=config)
                        worker.run(attention, *attention_handles, config=config)
                        separate = [torch.empty_like(value) for value in output_values]
                        for value, handle in zip(separate, device_outputs, strict=True):
                            worker.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
                    worker.run(program, *handles, config=config)
                    timings = []
                    for _ in range(args.iterations):
                        if args.fused_attention:
                            start = perf_counter()
                            worker.run(indexer, *index_handles, *device_outputs[:3], config=config)
                            worker.run(attention, *attention_handles, config=config)
                            split_times.append(perf_counter() - start)
                        start = perf_counter()
                        worker.run(program, *handles, config=config)
                        timings.append(perf_counter() - start)
                    for value, handle in zip(output_values, device_outputs, strict=True):
                        worker.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
                    for index in (6, 7):
                        original = values[index]
                        retained = torch.empty_like(original)
                        worker.copy_from(retained.data_ptr(), handles[index].data_ptr, retained.nbytes)
                        torch.testing.assert_close(retained, original, atol=0, rtol=0)

                count = anchor // ratio
                logical = torch.arange(count)
                physical = table[logical // 128].long() * 128 + logical % 128
                if count > TOPK:
                    dots = (queries.float() @ decoded[physical].float().T).bfloat16().float()
                    scores = (dots.relu() * weights.float().unsqueeze(-1)).bfloat16().float().sum(1).bfloat16()
                    winners = scores.argsort(descending=True, stable=True)[:, :TOPK]
                    expected = physical[winners]
                else:
                    expected = physical.expand(block, -1)
                keep = min(TOPK, count)
                torch.testing.assert_close(lengths, torch.full_like(lengths, keep), atol=0, rtol=0)
                torch.testing.assert_close(rows[:, :keep].sort().values, expected.int().sort().values, atol=0, rtol=0)
                assert bool((rows[:, keep:] == -1).all())
                assert bool((bias == 0).eq(rows >= 0).all())
                report = dict(block=block, ratio=ratio, anchor=anchor, exact_topk_rows=block,
                              resident_call_median_us=median(timings) * 1e6)
                if args.fused_attention:
                    for value, separate_value in zip(output_values, separate, strict=True):
                        torch.testing.assert_close(value, separate_value, atol=0, rtol=0)
                    query, window, compressed = reference
                    # Selection was checked independently above. Preserve its
                    # physical order for the floating-point reduction oracle.
                    expected_output, expected_lse = attention_reference(
                        query, projections[1], window, compressed, rows, projections[2], anchor,
                    )
                    expected_output, expected_lse = expected_output.reshape_as(output[0]), expected_lse.reshape_as(output[1])
                    report.update(split_call_median_us=median(split_times) * 1e6,
                                  attention_max_abs=(output[0] - expected_output).abs().max().item(),
                                  lse_max_abs=(output[1] - expected_lse).abs().max().item())
                    error = str(report)
                    torch.testing.assert_close(output[0], expected_output, atol=1e-5, rtol=1.3e-6,
                                               msg=lambda message: f"{error}\n{message}")
                    torch.testing.assert_close(output[1], expected_lse, atol=2e-5, rtol=2e-6, msg=error)
                reports.append(report)
                print(json.dumps(report), flush=True)
    return dict(passed=True, npu_execution=not args.compile_only, cases=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--blocks", type=int, choices=(8, 16, 32), nargs="+", default=(8, 16, 32))
    parser.add_argument("--fused-attention", action="store_true",
                        help="Compare one dispatch with separate device Indexer and Attention dispatches")
    parser.add_argument("--extended-context", action="store_true", help="Use 8K and 16K committed prefixes")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(validate(args), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
