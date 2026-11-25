#!/usr/bin/env python3
"""Check the native TP4 and mHC boundary of the packed draft graph."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_packed_draft_attention import make_packed_draft_tp_post


def make_tp_probe(block):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import retire_prefill_signals

    capacity = max(16, block)
    hidden, mix, mix_coeffs, hc_dim = FLASH.hidden_size, FLASH.hc_mult, FLASH.mix_hc, FLASH.hc_dim
    post = make_packed_draft_tp_post(block)

    @pl.jit(auto_scope=False)
    def run_rank(
        residual: pl.Tensor[[capacity, mix, hidden], pl.FP32],
        delta: pl.Tensor[[block, hidden], pl.BF16],
        function: pl.Tensor[[mix_coeffs, hc_dim], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[mix_coeffs], pl.FP32],
        output: pl.Out[pl.Tensor[[block, mix, hidden], pl.FP32]],
        pre_mix: pl.Out[pl.Tensor[[block, mix], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[block, hidden], pl.BF16]],
        window: pld.DistributedTensor[[block, hidden], pl.BF16],
        arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        output, pre_mix, ffn_input = post(
            residual, delta, function, scale, base, window, arrived, rank, 1,
            output, pre_mix, ffn_input,
        )
        retire_prefill_signals(output, arrived, 0, rank)
        return output, pre_mix, ffn_input

    @pl.jit.host
    def probe(
        residual: pl.Tensor[[TP_SIZE, capacity, mix, hidden], pl.FP32],
        delta: pl.Tensor[[TP_SIZE, block, hidden], pl.BF16],
        function: pl.Tensor[[TP_SIZE, mix_coeffs, hc_dim], pl.FP32],
        scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
        base: pl.Tensor[[TP_SIZE, mix_coeffs], pl.FP32],
        output: pl.Out[pl.Tensor[[TP_SIZE, block, mix, hidden], pl.FP32]],
        pre_mix: pl.Out[pl.Tensor[[TP_SIZE, block, mix], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[TP_SIZE, block, hidden], pl.BF16]],
    ):
        window_buffer = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(window_buffer, [block, hidden], dtype=pl.BF16)
            arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            run_rank(
                residual[rank], delta[rank], function[rank], scale[rank], base[rank],
                output[rank], pre_mix[rank], ffn_input[rank], window, arrived, rank, device=rank,
            )
        return output, pre_mix, ffn_input

    return probe



def make_tp_swa_probe(block, pages):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import retire_prefill_signals

    from blockspec.pypto_dsv41_packed_draft_attention import make_packed_swa_draft_layer

    capacity = max(16, block)
    hidden, mix, mix_coeffs, hc_dim = FLASH.hidden_size, FLASH.hc_mult, FLASH.mix_hc, FLASH.hc_dim
    dim, latent = FLASH.head_dim, FLASH.q_lora_rank
    heads = FLASH.num_attention_heads // TP_SIZE
    groups, output_rank = FLASH.o_groups // TP_SIZE, FLASH.o_lora_rank
    group_input = heads * dim // groups
    entry = make_packed_swa_draft_layer(pages, block)
    attend = pl.jit.inline(auto_scope=False)(entry._func)
    post = make_packed_draft_tp_post(block)

    @pl.jit(auto_scope=False)
    def run_rank(
        residual: pl.Tensor, input_mix: pl.Tensor, function: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
        norm: pl.Tensor, q_a_kv: pl.Tensor, q_gamma: pl.Tensor, q_b: pl.Tensor, kv_gamma: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor,
        window_payload: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        sink: pl.Tensor, zeros: pl.Tensor, wo_a: pl.Tensor, wo_b: pl.Tensor,
        contribution: pl.Out[pl.Tensor[[block, hidden], pl.BF16]],
        output: pl.Out[pl.Tensor[[block, mix, hidden], pl.FP32]],
        pre_mix: pl.Out[pl.Tensor[[block, mix], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[block, hidden], pl.BF16]],
        window: pld.DistributedTensor[[block, hidden], pl.BF16],
        arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        contribution = attend(
            residual, input_mix, norm, q_a_kv, q_gamma, q_b, kv_gamma, cosine, sine,
            window_payload, window_scales, window_lookup, window_rows, window_starts, spans,
            window_bias, sink, zeros, wo_a, wo_b, contribution,
        )
        output, pre_mix, ffn_input = post(
            residual, contribution, function, scale, base, window, arrived, rank, 1, output, pre_mix, ffn_input,
        )
        retire_prefill_signals(output, arrived, 0, rank)
        return contribution, output, pre_mix, ffn_input

    @pl.jit.host
    def probe(
        residual: pl.Tensor[[TP_SIZE, capacity, mix, hidden], pl.FP32],
        input_mix: pl.Tensor[[TP_SIZE, capacity, mix], pl.FP32],
        function: pl.Tensor[[TP_SIZE, mix_coeffs, hc_dim], pl.FP32],
        scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
        base: pl.Tensor[[TP_SIZE, mix_coeffs], pl.FP32],
        norm: pl.Tensor[[TP_SIZE, hidden], pl.BF16],
        q_a_kv: pl.Tensor[[TP_SIZE, hidden, latent + dim], pl.BF16],
        q_gamma: pl.Tensor[[TP_SIZE, latent], pl.BF16],
        q_b: pl.Tensor[[TP_SIZE, latent, heads * dim], pl.BF16],
        kv_gamma: pl.Tensor[[TP_SIZE, dim], pl.BF16],
        cosine: pl.Tensor[[TP_SIZE, capacity, 32], pl.FP32],
        sine: pl.Tensor[[TP_SIZE, capacity, 32], pl.FP32],
        window_payload: pl.Tensor[[TP_SIZE, pages, 128, 1, dim], pl.UINT8],
        window_scales: pl.Tensor[[TP_SIZE, pages, 128, 1, dim // 32], pl.BF16],
        window_lookup: pl.Tensor[[TP_SIZE, 1, 256], pl.INT32],
        window_rows: pl.Tensor[[TP_SIZE, 8, 16], pl.INT32],
        window_starts: pl.Tensor[[TP_SIZE, 8], pl.INT32],
        spans: pl.Tensor[[TP_SIZE, 2, 3], pl.INT32],
        window_bias: pl.Tensor[[TP_SIZE, 1, 128], pl.FP32],
        sink: pl.Tensor[[TP_SIZE, heads], pl.FP32],
        zeros: pl.Tensor[[TP_SIZE, 128, dim], pl.BF16],
        wo_a: pl.Tensor[[TP_SIZE, groups, group_input, output_rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[TP_SIZE, groups * output_rank, hidden], pl.BF16, pl.NZ],
        contribution: pl.Out[pl.Tensor[[TP_SIZE, block, hidden], pl.BF16]],
        output: pl.Out[pl.Tensor[[TP_SIZE, block, mix, hidden], pl.FP32]],
        pre_mix: pl.Out[pl.Tensor[[TP_SIZE, block, mix], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[TP_SIZE, block, hidden], pl.BF16]],
    ):
        window_buffer = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(window_buffer, [block, hidden], dtype=pl.BF16)
            arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            run_rank(
                residual[rank], input_mix[rank], function[rank], scale[rank], base[rank],
                norm[rank], q_a_kv[rank], q_gamma[rank], q_b[rank], kv_gamma[rank], cosine[rank], sine[rank],
                window_payload[rank], window_scales[rank], window_lookup[rank], window_rows[rank],
                window_starts[rank], spans[rank], window_bias[rank], sink[rank], zeros[rank], wo_a[rank], wo_b[rank],
                contribution[rank], output[rank], pre_mix[rank], ffn_input[rank], window, arrived, rank, device=rank,
            )
        return contribution, output, pre_mix, ffn_input

    return probe


def swa_projection_operands(checkpoint, layer):
    from dataclasses import asdict

    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.weights import bf16_nz_shape
    from validate_dsv41_pypto_packed_qkv import checkpoint_operands

    from blockspec.pypto_dsv41_draft import pack_draft_input_projection, pack_draft_output_projection, prepare_output

    prefix = f"layers.{layer}."
    wo = {name: checkpoint.tensor(prefix + "attn." + name) for name in ("wo_a.weight", "wo_b.weight")}
    qkv_and_wo = []
    reference_weights = []
    for rank in range(TP_SIZE):
        _, qkv = checkpoint_operands(checkpoint.root, layer, rank)
        local_wo = prepare_output(wo, asdict(FLASH), tp_rank=rank, tp=TP_SIZE)
        packed_wo = pack_draft_output_projection(local_wo)
        reference_weights.append((qkv, local_wo))
        wo_a, wo_b = packed_wo["wo_a"], packed_wo["wo_b"]
        a_physical = bf16_nz_shape(*wo_a.shape)
        # The host lowering retains one matrix axis for each rank's Wo-B.
        b_physical = bf16_nz_shape(1, *wo_b.shape)
        qkv_and_wo.append((pack_draft_input_projection(qkv), qkv["q_norm"], qkv["wq_b"], qkv["kv_norm"],
                          wo_a.reshape(a_physical), wo_b.reshape(b_physical)))
    packed = tuple(torch.stack(values) for values in zip(*qkv_and_wo, strict=True))
    return packed, reference_weights


def replicate(tensor):
    from models.deepseek_v4_1_flash_a3.config import TP_SIZE

    return torch.stack([tensor] * TP_SIZE)


def validate(args):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import golden_hc_collapse, golden_hc_post, golden_mhc_mixes
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig, StackedDeviceTensor

    torch.set_num_threads(3)
    dd = DistributedConfig(device_ids=args.devices, num_sub_workers=0)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.devices[0], distributed_config=dd)
    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    prefix = f"layers.{args.layer}.hc_attn_"
    function, scale, base = (checkpoint.tensor(prefix + name).float() for name in ("fn", "scale", "base"))
    embedding = weights.lookup_rows(torch.arange(150, 182)).float()
    generator = torch.Generator().manual_seed(2104)
    if args.swa:
        projections, reference_weights = swa_projection_operands(checkpoint, args.layer)
    cases = []
    for block in args.blocks:
        capacity = max(16, block)
        hidden, mix = FLASH.hidden_size, FLASH.hc_mult
        residual = embedding[:capacity, None, :] * torch.linspace(.75, 1.25, mix)[None, :, None]
        if capacity > block:
            residual[block:].fill_(1234.0)
        outputs = (
            torch.empty((TP_SIZE, block, mix, hidden), dtype=torch.float32),
            torch.empty((TP_SIZE, block, mix), dtype=torch.float32),
            torch.empty((TP_SIZE, block, hidden), dtype=torch.bfloat16),
        )
        if args.swa:
            from validate_dsv41_pypto_packed_indexer import attention_fixture

            pages = 4
            anchor = 193
            banks, (_, _, sink, zeros), _, (_, *history) = attention_fixture(
                block, pages, torch.tensor([2, 1, 0, 3], dtype=torch.int32), anchor, generator,
            )
            norm = checkpoint.tensor(f"layers.{args.layer}.attn_norm.weight")
            phase = torch.randn(capacity, 32, generator=generator)
            mix_input = torch.full((capacity, mix), 1 / mix)
            rope = phase.cos(), phase.sin()
            q_a_kv, q_gamma, q_b, kv_gamma, wo_a, wo_b = projections
            contributions = torch.empty((TP_SIZE, block, hidden), dtype=torch.bfloat16)
            inputs = (*map(replicate, (residual, mix_input, function, scale, base, norm)),
                      q_a_kv, q_gamma, q_b, kv_gamma,
                      *map(replicate, (*rope, *banks[:7], sink, zeros)), wo_a, wo_b)
            result_tensors = (contributions, *outputs)
            graph = make_tp_swa_probe(block, pages)
        else:
            # Distinct BF16 inputs exercise the TP reduction, including cancellation.
            contributions = (torch.randn(TP_SIZE, block, hidden, generator=generator) * .0625).bfloat16()
            inputs = replicate(residual), contributions, *map(replicate, (function, scale, base))
            result_tensors = outputs
            graph = make_tp_probe(block)
        compile_inputs = inputs
        if args.swa:
            groups = FLASH.o_groups // TP_SIZE
            group_input = FLASH.num_attention_heads * FLASH.head_dim // FLASH.o_groups
            output_rank = FLASH.o_lora_rank
            # The DSL describes logical matrices while the runtime binds their
            # packed bytes, including the retained rank-local matrix axis.
            compile_inputs = (*inputs[:-2], wo_a.view(TP_SIZE, groups, group_input, output_rank),
                              wo_b.view(TP_SIZE, groups * output_rank, hidden))
        program = graph.compile(*compile_inputs, *result_tensors, config=config)
        if args.compile_only:
            cases.append(dict(block=block, carrier_rows=capacity, compiled=True))
            continue
        with DistributedWorker([program], config=config, persistent=True, reset_persistent_windows=False) as worker:
            ranks = tuple(range(TP_SIZE))
            arguments, bindings = [], []
            for index, tensor in enumerate((*inputs, *result_tensors)):
                buffers = [worker.alloc_tensor(tensor.shape[1:], tensor.dtype,
                           worker_id=rank) for rank in ranks]
                if index < len(inputs):
                    for rank, buffer in enumerate(buffers):
                        item = tensor[rank]
                        worker.copy_to(buffer.data_ptr, item.data_ptr(), item.nbytes, worker_id=rank)
                bindings.append(buffers)
                arguments.append(StackedDeviceTensor(buffers, tensor.shape, ranks))
            worker.run(program, *arguments, config=config)
            timings = []
            for _ in range(args.iterations):
                start = perf_counter()
                worker.run(program, *arguments, config=config)
                timings.append(perf_counter() - start)
            for output, buffers in zip(result_tensors, bindings[len(inputs):], strict=True):
                for rank, buffer in enumerate(buffers):
                    item = output[rank]
                    worker.copy_from(item.data_ptr(), buffer.data_ptr, item.nbytes, worker_id=rank)

        summed = torch.zeros_like(contributions[0], dtype=torch.float32)
        for partial in contributions:
            summed.add_(partial.float())
        if not torch.isfinite(contributions).all() or not torch.count_nonzero(contributions):
            raise RuntimeError("Invalid Attention contribution before TP")
        expected_pre, expected_post, expected_skip = golden_mhc_mixes(residual[:block], function, scale, base)
        expected_hidden = golden_hc_post(summed.bfloat16().float(), residual[:block], expected_post, expected_skip)
        expected_ffn = golden_hc_collapse(expected_hidden, expected_pre).bfloat16()
        errors = {}
        for name, actual, expected in zip(
            ("hidden", "pre_mix", "ffn_input"), outputs, (expected_hidden, expected_pre, expected_ffn), strict=True,
        ):
            torch.testing.assert_close(actual, actual[0].expand_as(actual), rtol=0, atol=0)
            diff = actual[0].float() - expected.float()
            errors[name] = dict(max_abs=diff.abs().max().item(),
                                relative_l2=(diff.norm() / expected.float().norm().clamp_min(1e-30)).item())
            if errors[name]["relative_l2"] > 0.005:
                raise RuntimeError(f"TP4 {name} differs from the native Torch reference")
        case = dict(block=block, carrier_rows=capacity, identical_across_ranks=True, errors=errors)
        case["resident_call_median_us"] = median(timings) * 1e6
        if args.swa:
            from validate_dsv41_pypto_packed_layer import compare_bf16_contribution, reference

            collapsed = golden_hc_collapse(residual, mix_input).bfloat16()[:, None, :]
            empty_selection = torch.empty((block, 0), dtype=torch.int32)
            per_rank = []
            for rank, (qkv, wo) in enumerate(reference_weights):
                expected = reference(collapsed, norm, qkv, wo, rope, history, empty_selection, sink, anchor)
                per_rank.append(compare_bf16_contribution(contributions[rank], expected))
            case["swa_contribution_precision"] = per_rank
            if not all(stats["passed"] for stats in per_rank):
                raise RuntimeError(f"SWA contributions differ from the Torch reference: {per_rank}")
        print(json.dumps(case), flush=True)
        cases.append(case)
    return dict(passed=True, npu_execution=not args.compile_only, layer=args.layer,
                source="packed-swa" if args.swa else "tp-fixture", cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--devices", type=int, nargs=4, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--swa", action="store_true", help="Compute draft contributions in the same device dispatch")
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 16, 32], choices=(8, 16, 32))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(validate(args), indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
