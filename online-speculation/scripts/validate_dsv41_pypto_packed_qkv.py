#!/usr/bin/env python3
"""Check the 1122 draft QKV kernel with one resident TP4 rank."""

import argparse
from dataclasses import asdict
import json
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_draft import (
    make_packed_draft_output, make_packed_draft_qkv, pack_draft_input_projection,
    pack_draft_output_projection, prepare_output,
)


def checkpoint_operands(path, layer, rank):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    ckpt = Checkpoint(path)
    weights = ResidentWeights(ckpt)
    prefix = f"layers.{layer}.attn."
    width = FLASH.num_attention_heads * FLASH.head_dim // TP_SIZE
    q_shard = slice(rank * width, (rank + 1) * width)
    result = {}
    for key in ("wq_a", "wq_b", "wkv"):
        payload, scale = weights.projection(prefix + key, output_slice=q_shard if key == "wq_b" else None)
        result[key] = (payload.float() * scale).bfloat16().contiguous()
    result["q_norm"] = ckpt.tensor(prefix + "q_norm.weight")
    result["kv_norm"] = ckpt.tensor(prefix + "kv_norm.weight")
    return weights.lookup_rows(torch.arange(150, 182)), result


def validate_output_projection(args, config, generator):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import golden_o_proj
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.runtime import ChipWorker

    checkpoint = Checkpoint(args.checkpoint)
    prefix = f"layers.{args.layer}.attn."
    weights = {key: checkpoint.tensor(prefix + key) for key in ("wo_a.weight", "wo_b.weight")}
    operands = prepare_output(weights, asdict(FLASH), tp_rank=args.tp_rank, tp=TP_SIZE)
    packed = pack_draft_output_projection(operands)
    heads = FLASH.num_attention_heads // TP_SIZE
    for block in (8, 16, 32):
        attention = torch.randn(block, heads, FLASH.head_dim, generator=generator)
        phase = torch.randn(block, FLASH.qk_rope_head_dim // 2, generator=generator)
        cosine, sine = phase.cos(), phase.sin()
        reference = golden_o_proj(attention.bfloat16(), cosine, sine,
                                  operands["wo_a"].transpose(-2, -1), operands["wo_b"]).float()
        layouts = []
        for head_major in (False, True):
            local = attention.transpose(0, 1).contiguous() if head_major else attention
            contribution = torch.empty_like(reference, dtype=torch.bfloat16)
            values = (local.flatten(0, 1), packed["wo_a"], packed["wo_b"], cosine, sine, contribution)
            program = make_packed_draft_output(block_size=block, head_major=head_major).compile(
                *values, config=config,
            )
            with ChipWorker(config) as worker:
                handles = [worker.alloc_tensor(value.shape, value.dtype) for value in values]
                for value, handle in zip(values[:-1], handles[:-1], strict=True):
                    worker.copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)
                worker.run(program, *handles, config=config)
                worker.copy_from(contribution.data_ptr(), handles[-1].data_ptr, contribution.nbytes)
            layouts.append(contribution)
        torch.testing.assert_close(*layouts, atol=0, rtol=0)
        difference = (layouts[0].float() - reference).abs()
        relative_l2 = (difference.norm() / reference.norm().clamp_min(1e-30)).item()
        exceeds = (difference > 0.005 + 0.02 * reference.abs()).float().mean().item()
        # Match the upstream A3 Wo gate with the same input context on both paths.
        native_wo_outliers = (difference > 0.003 + reference.abs() / 128).sum().item()
        print(json.dumps(dict(stage="output_projection", layer=args.layer, tp_rank=args.tp_rank,
                              block=block, max_abs=difference.max().item(), relative_l2=relative_l2,
                              rtol_atol_failure_ratio=exceeds, native_wo_outliers=native_wo_outliers)), flush=True)
        if not bool(torch.isfinite(difference).all()) or relative_l2 > 0.005 or native_wo_outliers:
            raise RuntimeError("draft output projection differs from the BF16 Torch reference")


def main():
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
    from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, choices=range(FLASH.num_hidden_layers), default=0)
    parser.add_argument("--tp-rank", type=int, choices=range(TP_SIZE), default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output-projection", action="store_true", help="Also check the packed Wo projection")
    args = parser.parse_args()
    torch.set_num_threads(3)
    geometry = asdict(FLASH)
    heads, dim = FLASH.num_attention_heads // TP_SIZE, FLASH.head_dim
    x, operands = checkpoint_operands(args.checkpoint, args.layer, args.tp_rank)
    wa, wb, wk = (operands[key] for key in ("wq_a", "wq_b", "wkv"))
    q_gamma, kv_gamma = (operands[key] for key in ("q_norm", "kv_norm"))
    packed = pack_draft_input_projection(operands)
    generator = torch.Generator().manual_seed(41131)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device, distributed_config=DistributedConfig(device_ids=[args.device]))
    for capacity, blocks in ((16, (8, 16)), (32, (32,))):
        phase = torch.randn(capacity, FLASH.qk_rope_head_dim // 2, generator=generator)
        cosine, sine = phase.cos(), phase.sin()
        inputs = (x[:capacity], packed, q_gamma, wb, kv_gamma, cosine, sine)
        outputs = tuple(torch.zeros(capacity, width, dtype=torch.bfloat16)
                        for width in (FLASH.q_lora_rank, heads * dim, dim))
        values = (*inputs, *outputs)
        program = make_packed_draft_qkv(geometry, tp=TP_SIZE, capacity=capacity).compile(
            *values, capacity, config=config,
        )
        with ChipWorker(config) as worker:
            handles = [worker.alloc_tensor(value.shape, value.dtype) for value in values]
            for value, handle in zip(values, handles, strict=True):
                worker.copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)
            for block in blocks:
                outputs[-1].fill_(float("nan"))
                worker.copy_to(handles[-1].data_ptr, outputs[-1].data_ptr(), outputs[-1].nbytes)
                worker.run(program, *handles, block, config=config)
                durations = []
                for _ in range(args.iterations):
                    start = perf_counter()
                    worker.run(program, *handles, block, config=config)
                    durations.append(perf_counter() - start)
                for value, handle in zip(outputs, handles[len(inputs):], strict=True):
                    worker.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
                torch.testing.assert_close(outputs[-1][block:], torch.zeros_like(outputs[-1][block:]),
                                           atol=0, rtol=0)
                source = x[:block].float()
                query_latent = golden_rms_norm((source @ wa.float()).bfloat16(), q_gamma)
                raw_query = (query_latent.float() @ wb.float()).bfloat16().reshape(block, heads, dim)
                query = golden_rope_tail(raw_query, cosine[:block], sine[:block]).flatten(1)
                normed_kv = golden_rms_norm((source @ wk.float()).bfloat16(), kv_gamma)
                kv = golden_rope_tail(normed_kv, cosine[:block], sine[:block])
                errors = {}
                for name, value, reference in zip(("q_latent", "query", "kv"), outputs,
                                                  (query_latent, query, kv), strict=True):
                    reference = reference.float()
                    difference = (value[:block].float() - reference).abs()
                    exceeds = (difference > 0.002 + 0.01 * reference.abs()).float().mean().item()
                    relative_l2 = (difference.norm() / reference.norm().clamp_min(1e-30)).item()
                    errors[name] = dict(max_abs=difference.max().item(), rtol_atol_failure_ratio=exceeds,
                                        relative_l2=relative_l2)
                print(json.dumps(dict(layer=args.layer, tp_rank=args.tp_rank, block=block,
                                      median_us=median(durations) * 1e6, errors=errors)), flush=True)
                if any(stat["rtol_atol_failure_ratio"] > 0.005 or stat["relative_l2"] > 0.005
                       for stat in errors.values()):
                    raise RuntimeError("draft QKV differs from the BF16 Torch reference")
    if args.output_projection:
        validate_output_projection(args, config, generator)


if __name__ == "__main__":
    main()
