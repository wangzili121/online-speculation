#!/usr/bin/env python3
"""Verify native BF16 diffusion projections and K32 Attention with resident tensors."""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import torch

from blockspec.pypto_dsv41_draft import (
    load_draft_layer,
    make_output_program,
    make_qkv_program,
    prepare_output,
    prepare_qkv,
)
from blockspec.pypto_dsv41_indexer import INDEX_DIM, INDEX_HEADS, make_indexer_program
from blockspec.pypto_dsv41_sparse import make_sparse_attention
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_attention
from blockspec.vllm_ascend.dsv41_attention import PagedRows, draft_attention
from dsv41_pypto_indexer_fixture import check as check_indexer
from dsv41_pypto_indexer_fixture import fixture as indexer_fixture
from validate_dsv41_pypto_swa import compressed_fixture, fixture, swa_table


def synthetic_weights(config, rank, *, indexer=False):
    generator = torch.Generator().manual_seed(492)
    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads = config["num_attention_heads"]
    groups, out_rank = config["o_groups"], config["o_lora_rank"]

    def projection(rows, cols):
        return (torch.randn(rows, cols, generator=generator) * cols ** -0.5).to(torch.bfloat16)

    weights = {
        "wq_a.weight": projection(latent, hidden),
        "q_norm.weight": torch.ones(latent, dtype=torch.bfloat16),
        "wq_b.weight": projection(heads * HEAD_DIM, latent),
        "wkv.weight": projection(HEAD_DIM, hidden),
        "kv_norm.weight": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
        "attn_sink": torch.linspace(-4, 4, heads),
        "wo_a.weight": projection(groups * out_rank, heads * HEAD_DIM // groups),
        "wo_b.weight": projection(hidden, groups * out_rank),
    }
    packed = prepare_qkv(weights, config, tp_rank=rank)
    packed.update(prepare_output(weights, config, tp_rank=rank))
    if indexer:
        packed.update(idx_wq=projection(latent, INDEX_HEADS * INDEX_DIM),
                      idx_weight=projection(hidden, INDEX_HEADS))
    width = heads // 4 * HEAD_DIM
    torch.testing.assert_close(packed["wq_b"].T, weights["wq_b.weight"][rank * width:(rank + 1) * width],
                               atol=0, rtol=0)
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, choices=range(4), default=2)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--ratio", type=int, choices=(1, 2), help="include the corresponding compressed KV path")
    parser.add_argument("--indexer", action="store_true", help="select the compressed history with draft QLI")
    parser.add_argument("--anchor", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if bool(args.model) != bool(args.head) or args.output.exists():
        parser.error("provide both model and head or neither, and choose a new output file")
    if args.indexer and not args.ratio:
        parser.error("--indexer requires --ratio")
    sys.path.insert(0, str(args.lib.resolve()))

    def native(name):
        return importlib.import_module(f"models.deepseek_v4_1_flash_a3.{name}")

    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    torch.set_num_threads(4)
    artifact_role = None
    if args.head is not None:
        config, manifest, operands = load_draft_layer(args.model, args.head, args.layer, tp_rank=args.tp_rank)
        if (args.ratio or 0) != config["compress_ratios"][args.layer]:
            parser.error("--ratio differs from the selected checkpoint layer's cache geometry")
        artifact_role = manifest["artifact_role"]
    else:
        config = asdict(native("config").FLASH)
        operands = synthetic_weights(config, args.tp_rank, indexer=args.indexer)
    if args.indexer and "idx_wq" not in operands:
        parser.error("--indexer needs a layer owning diffusion index projections")
    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, rope_dim = config["num_attention_heads"] // 4, config["qk_rope_head_dim"]
    width = heads * HEAD_DIM
    max_context = native("decode_network").MAX_CONTEXT
    anchor = args.anchor if args.anchor is not None else (1025 if args.ratio else 257)
    if not 0 <= anchor <= max_context:
        parser.error(f"--anchor must be within the A3 context limit of {max_context}")
    x = torch.randn((BLOCK, hidden), generator=torch.Generator().manual_seed(49)).bfloat16()
    phase = torch.arange(anchor, anchor + BLOCK)[:, None] * torch.linspace(0.001, 0.11, rope_dim // 2)[None]
    cos, sin = phase.cos(), phase.sin()
    buffers = dict(operands, hidden=x, cos=cos, sin=sin, active=BLOCK)
    qkv_shapes = (
        ("qa", latent), ("latent", latent), ("qb", width), ("query", width),
        ("kv", HEAD_DIM), ("normalized_kv", HEAD_DIM), ("temporary", HEAD_DIM),
    )
    for name, columns in qkv_shapes:
        buffers[name] = torch.empty((BLOCK, columns), dtype=torch.bfloat16)
    qkv = make_qkv_program(native, config)
    swa = make_swa_attention(heads, token_major_query=True)
    local, _ = fixture(anchor, heads, 75)
    attention_args = [buffers["query"], *local[1:3], buffers["temporary"], local[4], operands["sink"], *local[6:]]
    execution = ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD
    options = RunConfig(arch=BackendType.Ascend910, execution_mode=execution, device_id=args.device)
    qkv_args = [buffers[name] for name in qkv.param_names]
    compiled_qkv = qkv.compile(config=options)
    compiled_swa = swa.compile(*attention_args, config=options)
    if args.ratio:
        if args.indexer:
            geometry, index_buffers, index_outputs, kv_pages, zeros, gold_kv = indexer_fixture(
                config, anchor, args.ratio, max_context=max_context)
            buffers.update(index_buffers)
            indexer_program = make_indexer_program(native, config, geometry)
            indexer_args = [buffers[name] for name in indexer_program.param_names]
            compiled_indexer = indexer_program.compile(config=options)
            compressed = [buffers["physical_rows"], buffers["lengths"], buffers["bias"], zeros]
            outputs = [torch.empty_like(value) for value in local[-2:]]
        else:
            compressed_fixture_args, _ = compressed_fixture(anchor, args.ratio, 75, local)
            kv_pages, *compressed = compressed_fixture_args[1:6]
            outputs = compressed_fixture_args[-2:]
        sparse_args = [kv_pages, *compressed, local[-2], local[-1], *outputs]
        sparse = make_sparse_attention(head_major_swa=True)
        compiled_sparse = sparse.compile(buffers["query"].view(-1, HEAD_DIM), *sparse_args, config=options)
    attention = sparse_args[-2] if args.ratio else local[-2]
    output_projection = make_output_program(native, config, head_major_input=not args.ratio)
    output_shapes = (
        ("context", width), ("unrotated", width), ("grouped", operands["wo_b"].shape[0]),
        ("contribution", hidden),
    )
    buffers["attention"] = attention
    for name, columns in output_shapes:
        buffers[name] = torch.empty((BLOCK, columns), dtype=torch.bfloat16)
    output_args = [buffers[name] for name in output_projection.param_names]
    compiled_output = output_projection.compile(config=options)
    if args.compile_only:
        args.output.write_text(json.dumps({"compiled": True, "executed": False}) + "\n")
        return
    with ChipWorker(config=options) as worker:
        handles = {}

        def bind(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if id(tensor) not in handles:
                handles[id(tensor)] = worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor)
            return handles[id(tensor)]

        def read(tensor):
            result = torch.empty_like(tensor)
            worker.copy_from(result.data_ptr(), bind(tensor).data_ptr, result.nbytes)
            return result

        try:
            worker.run(compiled_qkv, *(bind(tensor) for tensor in qkv_args))
            worker.run(compiled_swa, *(bind(tensor) for tensor in attention_args))
            if args.indexer:
                worker.run(compiled_indexer, *(bind(tensor) for tensor in indexer_args))
            if args.ratio:
                flat_query = worker.view_tensor(bind(buffers["query"]), (BLOCK * heads, HEAD_DIM))
                worker.run(compiled_sparse, flat_query, *(bind(tensor) for tensor in sparse_args))
            worker.run(compiled_output, *(bind(tensor) for tensor in output_args))
            output_buffers = {name: read(buffers[name]) for name, _ in output_shapes}
            if args.ratio:
                compressed_output, compressed_lse, compressed_pages = map(read, (
                    sparse_args[-2], sparse_args[-1], sparse_args[0],
                ))
            if args.indexer:
                index_results = {name: read(buffers[name]) for name in index_outputs}
                for name in ("index_codes", "index_scales"):
                    assert torch.equal(read(buffers[name]).view(torch.uint8), buffers[name].view(torch.uint8))
            projected = {name: read(buffers[name]) for name, _ in qkv_shapes}
            output, lse, pages_after = map(read, (local[-2], local[-1], local[1]))
        finally:
            for handle in reversed(tuple(handles.values())):
                worker.free_tensor(handle)

    def linear(value, weight):
        return (value.float() @ weight.float()).bfloat16()

    rms = native("rmsnorm").rms_norm_reference
    rope = native("rope").rope_reference
    eps = config["rms_norm_eps"]
    stage_reference = {
        "qa": linear(x, operands["wq_a"]),
        "latent": rms(projected["qa"], operands["q_norm"], eps),
        "qb": linear(projected["latent"], operands["wq_b"]),
        "query": rope(projected["qb"], cos, sin, heads, HEAD_DIM, rope_dim),
        "kv": linear(x, operands["wkv"]),
        "normalized_kv": rms(projected["kv"], operands["kv_norm"], eps),
        "temporary": rope(projected["normalized_kv"], cos, sin, 1, HEAD_DIM, rope_dim),
    }
    stage_errors = {}
    for name, expected in stage_reference.items():
        actual = projected[name]
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-3, msg=lambda msg, label=name: f"{label}\n{msg}")
        stage_errors[name] = float((actual.float() - expected.float()).abs().max())
    actual_qr, actual_q, actual_kv = (projected[name] for name in ("latent", "query", "temporary"))
    qr = rms(linear(x, operands["wq_a"]), operands["q_norm"], config["rms_norm_eps"])
    query = rope(linear(qr, operands["wq_b"]), cos, sin, heads, HEAD_DIM, rope_dim)
    kv = rope(rms(linear(x, operands["wkv"]), operands["kv_norm"], config["rms_norm_eps"]),
              cos, sin, 1, HEAD_DIM, rope_dim)
    errors = {}
    for name, actual, expected in (("latent", actual_qr, qr), ("query", actual_q, query), ("temporary_kv", actual_kv, kv)):
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.015, msg=lambda msg, label=name: f"{label}\n{msg}")
        errors[name] = float((actual.float() - expected.float()).abs().max())
    torch.testing.assert_close(pages_after.view(torch.uint8), local[1].view(torch.uint8), rtol=0, atol=0)
    expected_output, expected_lse = draft_attention(
        actual_q.view(1, BLOCK, heads, HEAD_DIM), actual_kv[None],
        PagedRows(local[1].unflatten(0, (-1, WINDOW)), swa_table(anchor)),
        torch.tensor([anchor]), operands["sink"], window=WINDOW,
    )
    for name, actual, expected in (("attention", output, expected_output), ("lse", lse, expected_lse)):
        expected = expected.transpose(1, 2).reshape_as(actual)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-6)
        errors[name] = float((actual - expected).abs().max())
    index_report = None
    if args.indexer:
        index_report = check_indexer(native, index_results, buffers, geometry, gold_kv, anchor=anchor)
        index_query_gold = rope(linear(actual_qr, operands["idx_wq"]), cos, sin, INDEX_HEADS, INDEX_DIM, rope_dim)
        torch.testing.assert_close(index_results["idx_query"].flatten(1), index_query_gold, atol=2e-3, rtol=8e-3)
        weights_gold = linear(x, operands["idx_weight"]).float() * (INDEX_DIM * INDEX_HEADS) ** -0.5
        torch.testing.assert_close(index_results["idx_scaled_weights"], weights_gold.half(), atol=5e-5, rtol=8e-3)
        compressed_gold = draft_attention(
            actual_q.view(1, BLOCK, heads, HEAD_DIM), actual_kv[None],
            PagedRows(local[1].unflatten(0, (-1, WINDOW)), swa_table(anchor)), torch.tensor([anchor]),
            operands["sink"], window=WINDOW, global_kv=gold_kv, global_indices=index_results["logical_rows"][None])
    elif args.ratio:
        reference_inputs = list(local)
        reference_inputs[0] = actual_q.view(BLOCK, heads, HEAD_DIM).transpose(0, 1).reshape(-1, HEAD_DIM)
        reference_inputs[3] = actual_kv
        reference_inputs[5] = operands["sink"]
        _, compressed_gold = compressed_fixture(anchor, args.ratio, 75, reference_inputs)
    if args.ratio:
        torch.testing.assert_close(compressed_pages.view(torch.uint8), kv_pages.view(torch.uint8), rtol=0, atol=0)
        for name, actual, expected in zip(
            ("compressed_attention", "compressed_lse"), (compressed_output, compressed_lse), compressed_gold, strict=True,
        ):
            expected = expected.reshape_as(actual)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-6)
            errors[name] = float((actual - expected).abs().max())
    active_attention = compressed_output if args.ratio else output
    if not args.ratio:
        active_attention = active_attention.view(heads, BLOCK, HEAD_DIM).transpose(0, 1)
    context = active_attention.reshape(BLOCK, width).bfloat16()
    groups, _, grouped_input = operands["wo_a"].shape
    output_reference = {
        "context": context,
        "unrotated": rope(output_buffers["context"], cos, -sin, heads, HEAD_DIM, rope_dim),
        "grouped": torch.einsum("tgd,god->tgo", output_buffers["unrotated"].float().view(
            BLOCK, groups, grouped_input), operands["wo_a"].float()).flatten(1).bfloat16(),
        "contribution": linear(output_buffers["grouped"], operands["wo_b"]),
    }
    for name, expected in output_reference.items():
        actual = output_buffers[name]
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-3, msg=lambda msg, label=name: f"{label}\n{msg}")
        stage_errors[name] = float((actual.float() - expected.float()).abs().max())
    report = dict(block=BLOCK, anchor=anchor, local_heads=heads, layer=args.layer, tp_rank=args.tp_rank,
                  synthetic_weights=args.head is None, artifact_role=artifact_role, cache_byte_exact=True,
                  execution="simulator" if args.simulate else "npu",
                  query_kv_device_handoff=True, compressed_ratio=args.ratio,
                  swa_lse_device_handoff=bool(args.ratio), max_absolute_error=errors,
                  attention_output_device_handoff=True, projection_stage_max_absolute_error=stage_errors)
    if args.indexer:
        report["indexer"] = index_report
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
