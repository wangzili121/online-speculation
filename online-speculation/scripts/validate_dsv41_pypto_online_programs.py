#!/usr/bin/env python3
"""Validate reusable DSV4.1 online loss, linear-gradient and Adam PyPTO kernels."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocabulary", type=int, default=129280)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-full-shape-compile", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.vocabulary < 64 or args.vocabulary % 64:
        parser.error("fresh output and a 64-aligned vocabulary required")

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_online_programs import (
        make_adam,
        make_element,
        make_kl,
        make_linear,
        make_rms,
        make_tv,
    )
    from blockspec.pypto_dsv41_optimizer_programs import make_publish
    from blockspec.pypto_dsv41_attention_vjp import (
        C1AAttentionVJPGeometry,
        make_block_extract,
        make_head_reduce,
        make_sink_reduce,
        make_softmax_vjp,
    )
    from blockspec.pypto_dsv41_tail_vjp import (
        make_hc_post_data_vjp,
        make_hc_post_gate_vjp,
        make_hc_pre_gate_vjp,
    )
    from blockspec.pypto_dsv41_moe_vjp import (
        make_clipped_swiglu_vjp,
        make_dequantize_blocked_weight,
        make_moe_combine_vjp,
        make_normalized_route_vjp,
    )

    torch.set_num_threads(4)
    torch.manual_seed(41)
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    vocabulary = args.vocabulary
    attention_fixture = C1AAttentionVJPGeometry(
        local_heads=1, swa=0, compressed=0, padded_columns=64,
    )
    programs = {
        "tv": make_tv(32, vocabulary).compile(
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "kl": make_kl(32, vocabulary).compile(
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "adam": make_adam((64, 128)).compile(
            *[empty((64, 128), torch.float32) for _ in range(7)],
            1e-4, .1, .001, 1., config=config,
        ),
        "linear": make_linear((32, 128), (64, 128)).compile(
            empty((32, 128)), empty((64, 128)), empty((32, 64)), config=config,
        ),
        "linear_input_grad": make_linear((32, 64), (128, 64)).compile(
            empty((32, 64)), empty((128, 64)), empty((32, 128)), config=config,
        ),
        "linear_weight_grad": make_linear(
            (32, 64), (32, 128), transpose_a=True, transpose_b=False, gradient_weight=True,
        ).compile(empty((32, 64)), empty((32, 128)), empty((64, 128), torch.float32), config=config),
        "rms_forward": make_rms(32, 128).compile(
            empty((32, 128)), empty((1, 128), torch.float32), empty((32, 128)),
            empty((32, 8), torch.float32), config=config,
        ),
        "rms_input_grad": make_rms(32, 128, mode="input").compile(
            empty((32, 128)), empty((1, 128), torch.float32), empty((32, 8), torch.float32),
            empty((32, 128)), empty((32, 128)), config=config,
        ),
        "rms_weight_grad": make_rms(32, 128, mode="weight").compile(
            empty((32, 128)), empty((32, 8), torch.float32), empty((32, 128)),
            empty((1, 128), torch.float32), config=config,
        ),
        "silu_grad": make_element((32, 128), "silu_grad").compile(
            empty((32, 128)), empty((32, 128)), empty((32, 128)), config=config,
        ),
        "publish": make_publish((64, 128)).compile(
            empty((64, 128), torch.float32), empty((64, 128)), config=config,
        ),
        "c1a_softmax_vjp": make_softmax_vjp(attention_fixture).compile(
            empty((attention_fixture.rows, attention_fixture.padded_columns), torch.float32),
            empty((attention_fixture.rows, attention_fixture.padded_columns)),
            empty((attention_fixture.rows, attention_fixture.padded_columns)),
            empty((attention_fixture.rows, 16), torch.float32), config=config,
        ),
        "c1a_sink_reduce": make_sink_reduce(attention_fixture).compile(
            empty((attention_fixture.rows, 16), torch.float32),
            empty((attention_fixture.local_heads, 16), torch.float32), config=config,
        ),
        "c1a_block_extract": make_block_extract(attention_fixture).compile(
            empty((attention_fixture.token_columns, attention_fixture.dimension)),
            empty((attention_fixture.block, attention_fixture.dimension)), config=config,
        ),
        "c1a_head_reduce": make_head_reduce(attention_fixture).compile(
            empty((attention_fixture.local_heads * attention_fixture.token_columns,
                   attention_fixture.dimension), torch.float32),
            empty((attention_fixture.token_columns, attention_fixture.dimension), torch.float32),
            config=config,
        ),
        "hc_post_data_vjp": make_hc_post_data_vjp(32, 128).compile(
            empty((32 * 4, 128)), empty((32, 4), torch.float32),
            empty((32 * 4, 4), torch.float32), empty((32, 128)),
            empty((32 * 4, 128)), config=config,
        ),
        "hc_post_gate_vjp": make_hc_post_gate_vjp(32, 128).compile(
            empty((32, 128)), empty((32 * 4, 128)), empty((32 * 4, 128)),
            empty((32, 16), torch.float32), empty((32 * 4, 16), torch.float32),
            config=config,
        ),
        "hc_pre_gate_vjp": make_hc_pre_gate_vjp().compile(
            empty((32, 24), torch.float32), empty((3,), torch.float32),
            empty((24,), torch.float32), empty((32, 4), torch.float32),
            empty((32, 4), torch.float32), empty((32 * 4, 4), torch.float32),
            empty((32, 24), torch.float32), config=config,
        ),
        "moe_combine_vjp": make_moe_combine_vjp(4, 128).compile(
            empty((4, 128)), empty((4 * 6, 128)), empty((4, 6), torch.float32),
            empty((4, 128)), empty((4 * 6, 128)), empty((4, 16), torch.float32),
            config=config,
        ),
        "normalized_route_vjp": make_normalized_route_vjp(4).compile(
            empty((4, 6), torch.float32), empty((4, 6), torch.float32),
            empty((4, 16), torch.float32), 2.5, config=config,
        ),
        "w8a8_vjp_weight": make_dequantize_blocked_weight(256, 128).compile(
            empty((256 * 2, 64), torch.int8), empty((128,), torch.float32),
            empty((256, 128)), config=config,
        ),
        "clipped_swiglu_vjp": make_clipped_swiglu_vjp(4, 128).compile(
            empty((4, 256), torch.float32), empty((4, 128)),
            empty((4, 256)), config=config,
        ),
    }
    if not args.skip_full_shape_compile:
        production_attention = C1AAttentionVJPGeometry()
        programs["c1a_softmax_vjp_tp4_k32"] = make_softmax_vjp(production_attention).compile(
            empty((production_attention.rows, production_attention.padded_columns), torch.float32),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.rows, 16), torch.float32), config=config,
        )
        programs["c1a_sink_reduce_tp4_k32"] = make_sink_reduce(production_attention).compile(
            empty((production_attention.rows, 16), torch.float32),
            empty((production_attention.local_heads, 16), torch.float32), config=config,
        )
        programs["c1a_block_extract_tp4_k32"] = make_block_extract(production_attention).compile(
            empty((production_attention.token_columns, production_attention.dimension)),
            empty((production_attention.block, production_attention.dimension)), config=config,
        )
        programs["c1a_head_reduce_tp4_k32"] = make_head_reduce(production_attention).compile(
            empty((production_attention.local_heads * production_attention.token_columns,
                   production_attention.dimension), torch.float32),
            empty((production_attention.token_columns, production_attention.dimension), torch.float32),
            config=config,
        )
        programs["hc_post_data_vjp_k32x5120"] = make_hc_post_data_vjp().compile(
            empty((32 * 4, 5120)), empty((32, 4), torch.float32),
            empty((32 * 4, 4), torch.float32), empty((32, 5120)),
            empty((32 * 4, 5120)), config=config,
        )
        programs["hc_post_gate_vjp_k32x5120"] = make_hc_post_gate_vjp().compile(
            empty((32, 5120)), empty((32 * 4, 5120)), empty((32 * 4, 5120)),
            empty((32, 16), torch.float32), empty((32 * 4, 16), torch.float32),
            config=config,
        )
        programs["moe_combine_vjp_k32x5120"] = make_moe_combine_vjp().compile(
            empty((32, 5120)), empty((32 * 6, 5120)), empty((32, 6), torch.float32),
            empty((32, 5120)), empty((32 * 6, 5120)), empty((32, 16), torch.float32),
            config=config,
        )
        programs["normalized_route_vjp_k32"] = make_normalized_route_vjp().compile(
            empty((32, 6), torch.float32), empty((32, 6), torch.float32),
            empty((32, 16), torch.float32), 2.5, config=config,
        )
        programs["moe_down_vjp_weight_2304x5120"] = make_dequantize_blocked_weight(
            2304, 5120,
        ).compile(
            empty((2304 * (5120 // 64), 64), torch.int8),
            empty((5120,), torch.float32), empty((2304, 5120)), config=config,
        )
        programs["moe_gate_up_vjp_weight_5120x4608"] = make_dequantize_blocked_weight(
            5120, 4608,
        ).compile(
            empty((5120 * (4608 // 64), 64), torch.int8),
            empty((4608,), torch.float32), empty((5120, 4608)), config=config,
        )
        programs["clipped_swiglu_vjp_k32x2304"] = make_clipped_swiglu_vjp().compile(
            empty((32, 4608), torch.float32), empty((32, 2304)),
            empty((32, 4608)), config=config,
        )
        # The frozen LM head is not updated, but its transpose is required to
        # propagate dLogits into the final trainable draft layer. Compile the
        # real V4.1 shape so a small fixture cannot hide an ABI/tile failure.
        programs["lm_head_input_grad_129280x5120"] = make_linear(
            (32, vocabulary), (5120, vocabulary), reduction_limit=64,
        ).compile(
            empty((32, vocabulary)), empty((5120, vocabulary)), empty((32, 5120)), config=config,
        )
        full_shape_linears = {
            # name: (upstream columns, input columns). Each pair compiles the
            # activation and parameter VJPs for one local TP4 projection.
            "wq_a": (1280, 5120),
            "wq_b": (8192, 1280),
            "wkv": (512, 5120),
            "wo_a_group": (1024, 4096),
            "wo_b": (5120, 2048),
        }
        for name, (outputs, inputs) in full_shape_linears.items():
            programs[f"{name}_input_grad"] = make_linear(
                (32, outputs), (inputs, outputs), reduction_limit=64,
            ).compile(
                empty((32, outputs)), empty((inputs, outputs)), empty((32, inputs)), config=config,
            )
            programs[f"{name}_weight_grad"] = make_linear(
                (32, outputs), (32, inputs), transpose_a=True, transpose_b=False,
                gradient_weight=True, reduction_limit=32,
            ).compile(
                empty((32, outputs)), empty((32, inputs)), empty((outputs, inputs), torch.float32), config=config,
            )
        # Frozen-tail input VJPs.  The target weights are never optimized, but
        # their transposes are required to reach an earlier selected draft
        # layer in a last-N replay.  mHC has 24 logical outputs padded to 32 in
        # the native resident layout.
        frozen_tail_linears = {
            "mhc_projection_input_grad_32x20480": (32, 20480),
            "moe_down_input_grad_5120x2304": (5120, 2304),
            "moe_gate_input_grad_2304x5120": (2304, 5120),
            "moe_up_input_grad_2304x5120": (2304, 5120),
            "router_input_grad_384x5120": (384, 5120),
        }
        for name, (outputs, inputs) in frozen_tail_linears.items():
            programs[name] = make_linear(
                (32, outputs), (inputs, outputs), reduction_limit=64,
            ).compile(
                empty((32, outputs)), empty((inputs, outputs)), empty((32, inputs)), config=config,
            )
    report = {
        "format": "orthrus-dsv41-pypto-online-programs-v1",
        "passed": False,
        "compiled": sorted(programs),
        "executed": False,
        "vocabulary": vocabulary,
        "cases": [],
    }
    if args.compile_only:
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    worker, handles = None, []
    try:
        worker = DistributedWorker(list(programs.values()), config=config)

        def allocate(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.detach())
            handles.append(handle)
            return handle

        def read(handle, target):
            worker.copy_from(target.data_ptr(), handle.data_ptr, target.nbytes)
            return target

        student = torch.randn(32, vocabulary, dtype=torch.bfloat16).requires_grad_()
        teacher = torch.randn_like(student)
        student_handle, teacher_handle = allocate(student), allocate(teacher)
        gradient_handle = allocate(torch.zeros_like(student))
        losses_handle = allocate(torch.zeros(32, 8, dtype=torch.float32))
        for objective in ("tv", "kl"):
            for valid in (1, 7, 31):
                worker.run(
                    programs[objective], student_handle, teacher_handle,
                    gradient_handle, losses_handle, valid, valid, config=config,
                )
                student_log = student[:valid].float().log_softmax(-1)
                teacher_log = teacher[:valid].float().log_softmax(-1)
                expected = (
                    .5 * (student_log.exp() - teacher_log.exp()).abs().sum(-1).mean()
                    if objective == "tv"
                    else (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()
                )
                expected_gradient, = torch.autograd.grad(expected, student, retain_graph=True)
                actual = read(gradient_handle, torch.empty_like(student))
                losses = read(losses_handle, torch.empty(32, 8, dtype=torch.float32))
                torch.testing.assert_close(actual, expected_gradient, atol=2e-5, rtol=.03)
                torch.testing.assert_close(losses[:, 0].sum(), expected.detach(), atol=2e-5, rtol=.002)
                assert torch.count_nonzero(actual[valid:]) == 0
                report["cases"].append({"operator": objective, "valid": valid, "loss": float(expected)})

        left, weight = torch.randn(32, 128).bfloat16(), torch.randn(64, 128).bfloat16()
        left_handle, weight_handle = allocate(left), allocate(weight)
        output_handle = allocate(torch.zeros(32, 64, dtype=torch.bfloat16))
        worker.run(programs["linear"], left_handle, weight_handle, output_handle, config=config)
        torch.testing.assert_close(
            read(output_handle, torch.empty(32, 64, dtype=torch.bfloat16)),
            left @ weight.T, atol=.125, rtol=.015,
        )
        report["cases"].append({"operator": "linear", "shape": [32, 128, 64]})

        dy, full_weight = torch.randn(32, 64).bfloat16(), torch.randn(128, 64).bfloat16()
        dx_handle = allocate(torch.zeros(32, 128, dtype=torch.bfloat16))
        worker.run(programs["linear_input_grad"], allocate(dy), allocate(full_weight), dx_handle, config=config)
        torch.testing.assert_close(
            read(dx_handle, torch.empty(32, 128, dtype=torch.bfloat16)),
            dy @ full_weight.T, atol=.125, rtol=.015,
        )
        x = torch.randn(32, 128).bfloat16()
        dw_handle = allocate(torch.zeros(64, 128, dtype=torch.float32))
        worker.run(programs["linear_weight_grad"], allocate(dy), allocate(x), dw_handle, config=config)
        # Parameter gradients preserve the BF16 reduction boundary before the FP32 master update.
        expected_dw = (dy.T @ x).bfloat16().float()
        torch.testing.assert_close(read(dw_handle, torch.empty_like(expected_dw)), expected_dw, atol=.125, rtol=.015)
        report["cases"].append({"operator": "linear_vjp", "activation": [32, 128], "weight": [64, 128]})

        rms_x = torch.randn(32, 128).bfloat16()
        gamma = torch.randn(1, 128)
        rms_y_handle = allocate(torch.zeros_like(rms_x))
        inv_handle = allocate(torch.zeros(32, 8, dtype=torch.float32))
        worker.run(programs["rms_forward"], allocate(rms_x), allocate(gamma), rms_y_handle, inv_handle, config=config)
        rms_x32 = rms_x.float()
        inverse = (rms_x32.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        expected_y = (rms_x32 * inverse * gamma).bfloat16()
        torch.testing.assert_close(read(rms_y_handle, torch.empty_like(rms_x)), expected_y, atol=.016, rtol=.016)
        dy_rms = torch.randn_like(rms_x)
        dx_rms_handle = allocate(torch.zeros_like(rms_x))
        worker.run(
            programs["rms_input_grad"], allocate(rms_x), allocate(gamma), inv_handle,
            allocate(dy_rms), dx_rms_handle, config=config,
        )
        reference_x = rms_x.float().requires_grad_()
        reference_gamma = gamma.detach().clone().requires_grad_()
        reference_y = reference_x * (reference_x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * reference_gamma
        reference_dx, reference_dg = torch.autograd.grad(reference_y, (reference_x, reference_gamma), dy_rms.float())
        torch.testing.assert_close(
            read(dx_rms_handle, torch.empty_like(rms_x)), reference_dx.bfloat16(), atol=.03125, rtol=.02,
        )
        dg_handle = allocate(torch.zeros(1, 128, dtype=torch.float32))
        worker.run(programs["rms_weight_grad"], allocate(rms_x), inv_handle, allocate(dy_rms), dg_handle, config=config)
        torch.testing.assert_close(
            read(dg_handle, torch.empty_like(reference_dg)), reference_dg.bfloat16().float(), atol=.125, rtol=.02,
        )
        report["cases"].append({"operator": "rmsnorm_forward_vjp", "shape": [32, 128]})

        silu_x, silu_dy = torch.randn(32, 128).bfloat16(), torch.randn(32, 128).bfloat16()
        silu_dx_handle = allocate(torch.zeros_like(silu_x))
        worker.run(programs["silu_grad"], allocate(silu_x), allocate(silu_dy), silu_dx_handle, config=config)
        sigmoid = silu_x.float().sigmoid()
        expected_silu_dx = (silu_dy.float() * sigmoid * (1 + silu_x.float() * (1 - sigmoid))).bfloat16()
        torch.testing.assert_close(read(silu_dx_handle, torch.empty_like(silu_x)), expected_silu_dx, atol=.016, rtol=.016)
        report["cases"].append({"operator": "silu_vjp", "shape": [32, 128]})

        gate_up = (torch.randn(4, 256) * 7).float()
        expert_dy = torch.randn(4, 128).bfloat16()
        gate_up_gradient_handle = allocate(torch.zeros(4, 256, dtype=torch.bfloat16))
        worker.run(
            programs["clipped_swiglu_vjp"], allocate(gate_up), allocate(expert_dy),
            gate_up_gradient_handle, config=config,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        gate_clamped = gate.clamp(max=10.)
        up_clamped = up.clamp(-10., 10.)
        sigmoid = gate_clamped.sigmoid()
        expected_gate = (
            expert_dy.float() * up_clamped * sigmoid
            * (1 + gate_clamped * (1 - sigmoid)) * (gate <= 10.)
        )
        expected_up = (
            expert_dy.float() * gate_clamped * sigmoid
            * ((up >= -10.) & (up <= 10.))
        )
        expected_gate_up = torch.cat((expected_gate, expected_up), dim=-1).bfloat16()
        torch.testing.assert_close(
            read(gate_up_gradient_handle, torch.empty_like(expected_gate_up)),
            expected_gate_up, atol=.016, rtol=.016,
        )
        report["cases"].append({"operator": "clipped_swiglu_vjp", "shape": [4, 128]})

        # C1A-specific softmax VJP. The fixture has 32 temporary token
        # columns, one sink column and zero-probability padding to 64.
        probability = torch.zeros(attention_fixture.rows, attention_fixture.padded_columns, dtype=torch.float32)
        scores = torch.randn(attention_fixture.rows, attention_fixture.block + 1)
        probability[:, :attention_fixture.block + 1] = scores.softmax(-1)
        probability_gradient = torch.randn_like(probability).bfloat16()
        score_gradient_handle = allocate(torch.zeros_like(probability_gradient))
        sink_rows_handle = allocate(torch.zeros(attention_fixture.rows, 16, dtype=torch.float32))
        worker.run(
            programs["c1a_softmax_vjp"], allocate(probability), allocate(probability_gradient),
            score_gradient_handle, sink_rows_handle, config=config,
        )
        expected_score_gradient = probability * (
            probability_gradient.float()
            - (probability * probability_gradient.float()).sum(-1, keepdim=True)
        )
        actual_score_gradient = read(score_gradient_handle, torch.empty_like(probability_gradient))
        torch.testing.assert_close(
            actual_score_gradient, expected_score_gradient.bfloat16(), atol=.002, rtol=.01,
        )
        sink_gradient_handle = allocate(torch.zeros(1, 16, dtype=torch.float32))
        worker.run(
            programs["c1a_sink_reduce"], sink_rows_handle, sink_gradient_handle, config=config,
        )
        torch.testing.assert_close(
            read(sink_gradient_handle, torch.empty(1, 16))[:, :1],
            expected_score_gradient[:, attention_fixture.sink_column].sum().reshape(1, 1),
            atol=.01, rtol=.02,
        )
        joined_gradient = torch.randn(attention_fixture.token_columns, attention_fixture.dimension).bfloat16()
        block_gradient_handle = allocate(torch.zeros(attention_fixture.block, attention_fixture.dimension,
                                                     dtype=torch.bfloat16))
        worker.run(
            programs["c1a_block_extract"], allocate(joined_gradient), block_gradient_handle, config=config,
        )
        torch.testing.assert_close(
            read(block_gradient_handle, torch.empty_like(joined_gradient[-attention_fixture.block:])),
            joined_gradient[-attention_fixture.block:], atol=0, rtol=0,
        )
        per_head = torch.randn(
            attention_fixture.local_heads * attention_fixture.token_columns,
            attention_fixture.dimension,
            dtype=torch.float32,
        )
        shared_handle = allocate(torch.zeros(
            attention_fixture.token_columns, attention_fixture.dimension, dtype=torch.float32,
        ))
        worker.run(programs["c1a_head_reduce"], allocate(per_head), shared_handle, config=config)
        expected_shared = per_head.view(
            attention_fixture.local_heads, attention_fixture.token_columns,
            attention_fixture.dimension,
        ).sum(0)
        torch.testing.assert_close(
            read(shared_handle, torch.empty_like(expected_shared)), expected_shared, atol=1e-6, rtol=1e-6,
        )
        report["cases"].append({
            "operator": "c1a_attention_vjp",
            "block": attention_fixture.block,
            "sink_column": attention_fixture.sink_column,
            "padded_columns": attention_fixture.padded_columns,
        })

        hc_dy = torch.randn(32, 4, 128).bfloat16()
        hc_value = torch.randn(32, 128).bfloat16()
        hc_residual = torch.randn(32, 4, 128).bfloat16()
        hc_post = torch.randn(32, 4, dtype=torch.float32)
        hc_comb = torch.randn(32, 4, 4, dtype=torch.float32)
        hc_value_grad_handle = allocate(torch.zeros(32, 128, dtype=torch.bfloat16))
        hc_residual_grad_handle = allocate(torch.zeros(32 * 4, 128, dtype=torch.bfloat16))
        worker.run(
            programs["hc_post_data_vjp"], allocate(hc_dy.flatten(0, 1)), allocate(hc_post),
            allocate(hc_comb.flatten(0, 1)), hc_value_grad_handle, hc_residual_grad_handle,
            config=config,
        )
        expected_value_grad = torch.einsum("tjd,tj->td", hc_dy.float(), hc_post).bfloat16()
        expected_residual_grad = torch.einsum("tjd,tij->tid", hc_dy.float(), hc_comb).bfloat16()
        torch.testing.assert_close(
            read(hc_value_grad_handle, torch.empty_like(expected_value_grad)),
            expected_value_grad, atol=.016, rtol=.016,
        )
        torch.testing.assert_close(
            read(hc_residual_grad_handle, torch.empty_like(expected_residual_grad.flatten(0, 1))),
            expected_residual_grad.flatten(0, 1), atol=.016, rtol=.016,
        )
        hc_post_grad_handle = allocate(torch.zeros(32, 16, dtype=torch.float32))
        hc_comb_grad_handle = allocate(torch.zeros(32 * 4, 16, dtype=torch.float32))
        worker.run(
            programs["hc_post_gate_vjp"], allocate(hc_value), allocate(hc_residual.flatten(0, 1)),
            allocate(hc_dy.flatten(0, 1)), hc_post_grad_handle, hc_comb_grad_handle, config=config,
        )
        actual_post_grad = read(hc_post_grad_handle, torch.empty(32, 16))[:, :4]
        actual_comb_grad = read(hc_comb_grad_handle, torch.empty(32 * 4, 16))[:, :4]
        expected_post_grad = torch.einsum("tjd,td->tj", hc_dy.float(), hc_value.float())
        expected_comb_grad = torch.einsum("tid,tjd->tij", hc_residual.float(), hc_dy.float())
        torch.testing.assert_close(actual_post_grad, expected_post_grad, atol=.02, rtol=.002)
        torch.testing.assert_close(
            actual_comb_grad, expected_comb_grad.flatten(0, 1), atol=.02, rtol=.002,
        )
        report["cases"].extend((
            {"operator": "hc_post_data_vjp", "shape": [32, 4, 128]},
            {"operator": "hc_post_gate_vjp", "shape": [32, 4, 128]},
        ))

        projection = torch.randn(32, 24, dtype=torch.float32, requires_grad=True)
        hc_scale = torch.tensor([.7, 1.1, .9], dtype=torch.float32)
        hc_base = torch.randn(24, dtype=torch.float32)
        pre = torch.sigmoid(projection[:, :4] * hc_scale[0] + hc_base[:4]) + 1e-6
        post = 2 * torch.sigmoid(projection[:, 4:8] * hc_scale[1] + hc_base[4:8])
        probability = torch.softmax(
            projection[:, 8:].reshape(32, 4, 4) * hc_scale[2] + hc_base[8:].reshape(4, 4), -1,
        ) + 1e-6
        probability = probability / (probability.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            probability = probability / (probability.sum(-1, keepdim=True) + 1e-6)
            probability = probability / (probability.sum(-2, keepdim=True) + 1e-6)
        pre_gradient = torch.randn_like(pre)
        post_gradient = torch.randn_like(post)
        combination_gradient = torch.randn_like(probability)
        expected_projection_gradient, = torch.autograd.grad(
            (pre, post, probability), projection,
            (pre_gradient, post_gradient, combination_gradient),
        )
        projection_gradient_handle = allocate(torch.zeros(32, 24, dtype=torch.float32))
        worker.run(
            programs["hc_pre_gate_vjp"], allocate(projection.detach()), allocate(hc_scale),
            allocate(hc_base), allocate(pre_gradient), allocate(post_gradient),
            allocate(combination_gradient.flatten(0, 1)), projection_gradient_handle, config=config,
        )
        actual_projection_gradient = read(
            projection_gradient_handle, torch.empty_like(expected_projection_gradient),
        )
        # Twenty Sinkhorn rounds accumulate small device-exp differences; the
        # gradient remains FP32 and the observed worst case is below 6e-5.
        torch.testing.assert_close(
            actual_projection_gradient, expected_projection_gradient, atol=7e-5, rtol=2e-3,
        )
        report["cases"].append({"operator": "hc_pre_gate_vjp", "shape": [32, 4, 4]})

        moe_dy = torch.randn(4, 128).bfloat16()
        expert_outputs = torch.randn(4, 6, 128).bfloat16()
        selected_scores = torch.rand(4, 6, dtype=torch.float32) + .1
        route_weights = 2.5 * selected_scores / selected_scores.sum(-1, keepdim=True)
        shared_grad_handle = allocate(torch.zeros(4, 128, dtype=torch.bfloat16))
        expert_grad_handle = allocate(torch.zeros(4 * 6, 128, dtype=torch.bfloat16))
        weight_grad_handle = allocate(torch.zeros(4, 16, dtype=torch.float32))
        worker.run(
            programs["moe_combine_vjp"], allocate(moe_dy), allocate(expert_outputs.flatten(0, 1)),
            allocate(route_weights), shared_grad_handle, expert_grad_handle, weight_grad_handle,
            config=config,
        )
        expected_expert_grad = route_weights.unsqueeze(-1) * moe_dy.float().unsqueeze(1)
        expected_weight_grad = torch.einsum("td,trd->tr", moe_dy.float(), expert_outputs.float())
        torch.testing.assert_close(
            read(shared_grad_handle, torch.empty_like(moe_dy)), moe_dy, atol=0, rtol=0,
        )
        torch.testing.assert_close(
            read(expert_grad_handle, torch.empty(4 * 6, 128, dtype=torch.bfloat16)),
            expected_expert_grad.flatten(0, 1).bfloat16(), atol=.016, rtol=.016,
        )
        actual_weight_grad = read(weight_grad_handle, torch.empty(4, 16))[:, :6]
        torch.testing.assert_close(actual_weight_grad, expected_weight_grad, atol=.02, rtol=.002)
        score_grad_handle = allocate(torch.zeros(4, 16, dtype=torch.float32))
        worker.run(
            programs["normalized_route_vjp"], allocate(selected_scores), allocate(actual_weight_grad.contiguous()),
            score_grad_handle, 2.5, config=config,
        )
        probability = selected_scores / selected_scores.sum(-1, keepdim=True)
        expected_score_grad = 2.5 / selected_scores.sum(-1, keepdim=True) * (
            expected_weight_grad - (expected_weight_grad * probability).sum(-1, keepdim=True)
        )
        torch.testing.assert_close(
            read(score_grad_handle, torch.empty(4, 16))[:, :6],
            expected_score_grad, atol=.01, rtol=.002,
        )
        report["cases"].append({"operator": "fixed_route_moe_vjp", "shape": [4, 6, 128]})

        logical_weight = torch.randint(-127, 128, (256, 128), dtype=torch.int8)
        packed_weight = logical_weight.reshape(256, 2, 64).permute(1, 0, 2).contiguous().view(512, 64)
        channel_scale = torch.rand(128, dtype=torch.float32) * .02
        logical_handle = allocate(torch.zeros(256, 128, dtype=torch.bfloat16))
        worker.run(
            programs["w8a8_vjp_weight"], allocate(packed_weight), allocate(channel_scale),
            logical_handle, config=config,
        )
        expected_weight = (logical_weight.float() * channel_scale).bfloat16()
        torch.testing.assert_close(
            read(logical_handle, torch.empty_like(expected_weight)), expected_weight,
            atol=0, rtol=0,
        )
        report["cases"].append({
            "operator": "w8a8_frozen_input_vjp_weight", "shape": [256, 128],
        })

        master = torch.nn.Parameter(torch.randn(64, 128))
        optimizer = torch.optim.AdamW([master], lr=1e-4, weight_decay=0, foreach=False)
        state = [allocate(master.detach()), allocate(torch.zeros_like(master)), allocate(torch.zeros_like(master))]
        next_state = [allocate(torch.zeros_like(master)) for _ in range(3)]
        gradient = allocate(torch.zeros_like(master))
        for step in range(1, 5):
            value = torch.randn_like(master) * .02
            master.grad = value
            worker.copy_to(gradient.data_ptr, value.data_ptr(), value.nbytes)
            worker.run(
                programs["adam"], state[0], gradient, state[1], state[2], *next_state,
                1e-4, 1 - .9**step, 1 - .999**step, 1., config=config,
            )
            optimizer.step()
            expected = (master.detach(), optimizer.state[master]["exp_avg"], optimizer.state[master]["exp_avg_sq"])
            for reference, actual in zip(expected, next_state, strict=True):
                torch.testing.assert_close(read(actual, torch.empty_like(reference)), reference, atol=2e-6, rtol=.003)
            state, next_state = next_state, state
        report["cases"].append({"operator": "adam", "steps": 4})
        execution_handle = allocate(torch.zeros(64, 128, dtype=torch.bfloat16))
        worker.run(programs["publish"], state[0], execution_handle, config=config)
        published = read(execution_handle, torch.empty(64, 128, dtype=torch.bfloat16))
        torch.testing.assert_close(published, expected[0].bfloat16(), atol=0, rtol=0)
        report["cases"].append({"operator": "publish", "shape": [64, 128]})
        report["executed"] = report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
