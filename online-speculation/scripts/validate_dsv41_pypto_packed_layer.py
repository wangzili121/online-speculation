#!/usr/bin/env python3
"""Validate the fused, rank-local scored draft attention sublayer."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_draft import (
    pack_draft_input_projection, pack_draft_output_projection, prepare_output,
)
from blockspec.pypto_dsv41_packed_draft_attention import make_packed_scored_draft_layer
from blockspec.pypto_dsv41_packed_indexer import packed_index_controls
from validate_dsv41_pypto_packed_indexer import attention_fixture, attention_reference, fixtures
from validate_dsv41_pypto_packed_qkv import checkpoint_operands


def validate(args):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import golden_o_proj
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
    from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm
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
    heads = FLASH.num_attention_heads // TP_SIZE
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
        inputs = (x_hc, pre_mix, norm, packed_qkv, operands["q_norm"], operands["wq_b"], operands["kv_norm"],
                  idx_wq, idx_weight, cosine, sine, index_payload, index_scales, index_table, request_ids,
                  visible, *banks, sink, zeros, packed_wo["wo_a"], packed_wo["wo_b"])
        outputs = (rows, lengths, bias, contribution)
        program = make_packed_scored_draft_layer(pages, pages, bucket, ratio, block).compile(
            *inputs, *outputs, config=config,
        )
        with ChipWorker(config) as worker:
            handles = [worker.alloc_tensor(t.shape, t.dtype) for t in (*inputs, *outputs)]
            for tensor, handle in zip(inputs, handles[:len(inputs)], strict=True):
                worker.copy_to(handle.data_ptr, tensor.data_ptr(), tensor.nbytes)
            worker.run(program, *handles, config=config)
            timings = []
            for _ in range(args.iterations):
                start = perf_counter()
                worker.run(program, *handles, config=config)
                timings.append(perf_counter() - start)
            for tensor, handle in zip(outputs, handles[len(inputs):], strict=True):
                worker.copy_from(tensor.data_ptr(), handle.data_ptr, tensor.nbytes)

        committed = anchor // ratio
        count = min(committed, FLASH.index_topk)
        torch.testing.assert_close(lengths, torch.full_like(lengths, count), atol=0, rtol=0)
        logical = torch.argsort(table)[rows[:, :count] // 128] * 128 + rows[:, :count] % 128
        assert bool(((logical >= 0) & (logical < committed)).all())
        assert bool((rows[:, count:] == -1).all())
        # The packed Indexer fixture independently tests exact selected sets.
        # Use the selected rows here to isolate the connected projection path.
        x = golden_rms_norm(embedding[:block], norm)
        qr = golden_rms_norm((x.float() @ operands["wq_a"].float()).bfloat16(), operands["q_norm"])
        query = (qr.float() @ operands["wq_b"].float()).bfloat16().view(block, heads, FLASH.head_dim)
        query = golden_rope_tail(query, cosine[:block], sine[:block])
        kv = golden_rms_norm((x.float() @ operands["wkv"].float()).bfloat16(), operands["kv_norm"])
        kv = golden_rope_tail(kv, cosine[:block], sine[:block])
        context, _ = attention_reference(query, kv, window, compressed, rows, sink, anchor)
        expected = golden_o_proj(context.bfloat16(), cosine[:block], sine[:block],
                                 wo["wo_a"].transpose(-2, -1), wo["wo_b"]).float()
        relative_l2 = ((contribution.float() - expected).norm() / expected.norm().clamp_min(1e-30)).item()
        case = dict(block=block, ratio=ratio, anchor=anchor, relative_l2=relative_l2,
                    resident_call_median_us=median(timings) * 1e6)
        print(json.dumps(case), flush=True)
        if relative_l2 > 0.005:
            raise RuntimeError("scored draft attention sublayer differs from the Torch reference")
        cases.append(case)
    return dict(passed=True, npu_execution=True, layer=args.layer, tp_rank=args.tp_rank, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(validate(args), indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
