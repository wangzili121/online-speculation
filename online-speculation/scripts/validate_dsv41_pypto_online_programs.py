#!/usr/bin/env python3
"""Validate reusable DSV4.1 online loss, linear-gradient and Adam PyPTO kernels."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def compile_resident_update_extras(compile_program, empty, config):
    """Cover the production suffix and optimizer leaves absent from the component fixtures."""
    import torch
    from blockspec.pypto_dsv41_ep_vjp import EXPERT_CAPACITY
    from blockspec.pypto_dsv41_moe_vjp import make_frozen_w8a8_expert_input_vjp
    from blockspec.pypto_dsv41_online_programs import make_adam, make_rms
    from blockspec.pypto_dsv41_optimizer_programs import make_publish, make_reduce, make_stats
    from blockspec.pypto_dsv41_tail_vjp import make_hc_pre_data_vjp, make_hc_pre_projection_forward

    rows, hidden = 32, 5120
    fp32 = lambda shape: empty(shape, torch.float32)
    residual, function = empty((rows * 4, hidden)), fp32((24, hidden * 4))
    projection, inverse = fp32((rows, 32)), fp32((rows, 16))
    programs = {
        "hc_pre_projection": compile_program(
            make_hc_pre_projection_forward(), residual, function, projection, inverse, config=config,
        ),
        "hc_pre_data_vjp": compile_program(
            make_hc_pre_data_vjp(), residual, fp32((rows, 4)), function, inverse,
            projection, empty((rows, hidden)), residual, empty((rows * 4, hidden)),
            fp32((rows, 16)), config=config,
        ),
    }
    for expert_rows, intermediate in ((rows, 768), (EXPERT_CAPACITY, 2304)):
        hidden_values = empty((expert_rows, hidden))
        gate_width = 2 * intermediate
        programs[f"expert_input_vjp_{expert_rows}"] = compile_program(
            make_frozen_w8a8_expert_input_vjp(expert_rows, hidden, intermediate),
            hidden_values, hidden_values,
            empty((hidden * gate_width // 64, 64), torch.int8), fp32((gate_width,)),
            empty((intermediate * hidden // 64, 64), torch.int8), fp32((hidden,)),
            empty((expert_rows, hidden)), config=config,
        )
    for width in (512, 1280, hidden):
        x, gamma, inv = empty((rows, width)), empty((1, width)), fp32((rows, 8))
        for mode, inputs in (
            ("forward", (x, gamma, x, inv)),
            ("input", (x, gamma, inv, x, x)),
            ("weight", (x, inv, x, fp32((1, width)))),
        ):
            programs[f"rms_{mode}_{width}"] = compile_program(
                make_rms(rows, width, mode=mode, epsilon=1e-20, gamma_bf16=True),
                *inputs, config=config,
            )
    for shape in ((16,), (8192, 1280)):
        master = fp32(shape)
        stats, count = make_stats(shape)
        sums, flags = fp32((count, 8)), fp32((count, 8))
        programs[f"gradient_stats_{shape}"] = compile_program(
            stats, master, sums, flags, config=config,
        )
        while count > 1:
            reduce, next_count = make_reduce(count)
            out_sums, out_flags = fp32((next_count, 8)), fp32((next_count, 8))
            programs[f"gradient_reduce_{count}"] = compile_program(
                reduce, sums, flags, out_sums, out_flags, config=config,
            )
            sums, flags, count = out_sums, out_flags, next_count
        programs[f"adam_{shape}"] = compile_program(
            make_adam(shape), *(master,) * 7, 1e-5, .1, .001, 1., config=config,
        )
    programs["publish_projection"] = compile_program(
        make_publish((8192, 1280)), fp32((8192, 1280)), empty((8192, 1280)), config=config,
    )
    return programs


def check_c1a_resident_dispatch(empty):
    """Run the real Serving call sites with shape-only tensors and check every binding."""
    from threading import Lock
    from types import SimpleNamespace

    import torch
    from blockspec.dsv41_suffix import DSV41SuffixPlan
    from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
    from blockspec.pypto_dsv41_runtime import DraftRound, ResidentDraftAttention, native

    class ShapeOnlyResident:
        def workspace(self, shape, dtype, rank):
            return empty(shape, dtype)

        def execute(self, compiled, rank, arguments, abi):
            for argument, spec in zip(arguments, abi["params"], strict=True):
                if not isinstance(argument, torch.Tensor):
                    if spec["shape"] is not None:
                        raise TypeError(f"{spec['name']} expects a tensor")
                    continue
                dtype = str(argument.dtype).removeprefix("torch.")
                if not all(expected in (-1, actual) for expected, actual in
                           zip(spec["shape"], argument.shape, strict=True)):
                    raise ValueError(f"{spec['name']} shape: {argument.shape} vs {spec['shape']}")
                if spec["dtype"] not in {dtype, dtype.replace("float", "fp"), dtype.replace("int", "i")}:
                    raise TypeError(f"{spec['name']} dtype: {argument.dtype} vs {spec['dtype']}")

    layer, owner = 39, (0, 0, 1)
    operands = dict(
        wq_a=empty((5120, 1280)), wq_b=empty((1280, 8192)), wkv=empty((5120, 512)),
        q_norm=empty((1280,)), kv_norm=empty((512,)), sink=empty((16,), torch.float32),
        wo_a=empty((2, 1024, 4096)), wo_b=empty((2048, 5120)),
    )
    draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
    draft.rank_ids, draft.heads, draft.joined_attention = {0: 0}, 16, False
    draft.resident, draft.native = ShapeOnlyResident(), native
    draft.config = dict(
        num_hidden_layers=40, hidden_size=5120, q_lora_rank=1280, num_attention_heads=64,
        head_dim=512, qk_rope_head_dim=64, rms_norm_eps=1e-6, o_groups=8, o_lora_rank=1024,
    )
    draft.programs, draft.compile_lock = {}, Lock()
    draft.operands = {(0, layer): operands}
    draft.constants = {0: dict(swa_zero=empty((128, 512)), sparse_zero=empty((128, 512)))}
    draft.online_bank = SimpleNamespace(operands=lambda *_: operands)
    history = dict(
        committed_tokens=129, role=native("cache").resolve_layer_role(layer),
        planes={
            "swa": [dict(shape=(2, 128, 1, 512), stride_rows=128, offset_rows=0,
                         pages=(0, 1), handle=empty((256, 512)))],
            "long": [dict(handle=empty((512, 512)))],
        },
    )
    round_ = DraftRound(
        draft, {(0, layer, owner): history},
        capture=(DSV41SuffixPlan(1), {owner: ("fixture", (10,))}),
    )
    round_.index_rows = lambda *_: dict(
        physical_rows=empty((32, 512), torch.int32), lengths=empty((32,), torch.int32),
        bias=empty((32, 512), torch.float32),
    )
    rope = empty((32, 32), torch.float32)
    hidden, contribution = empty((32, 5120)), empty((32, 5120))
    with TemporaryDirectory(prefix="dsv41-serving-dispatch-") as directory:
        draft.output = Path(directory)
        round_.project_block(0, layer, owner, hidden, rope, rope, contribution, record=True)
        reverse = DSV41ResidentPrimitives.__new__(DSV41ResidentPrimitives)
        reverse.draft, reverse.resident = draft, draft.resident
        reverse._draft_operand = lambda rank, layer, key, shape: operands[key].view(shape)
        reverse._reverse_c1a_rank(
            layer, 0, round_.captured_attention[0, layer, owner], contribution,
        )
    return len(draft.programs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocabulary", type=int, default=129280)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--resident-abi", action="store_true",
                        help="Compile and check the exact L2 Serving ABI without weights or devices")
    parser.add_argument("--skip-full-shape-compile", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.vocabulary < 64 or args.vocabulary % 64:
        parser.error("fresh output and a 64-aligned vocabulary required")
    if args.resident_abi and (not args.compile_only or args.skip_full_shape_compile):
        parser.error("--resident-abi requires --compile-only and production shapes")

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
        c1a_vjp_factories,
        make_block_extract,
        make_probability_block_extract,
        make_sink_reduce,
        make_softmax_vjp,
    )
    from blockspec.pypto_dsv41_tail_vjp import (
        make_hc_collapse_forward,
        make_hc_collapse_vjp,
        make_hc_post_data_vjp,
        make_hc_post_gate_vjp,
        make_hc_pre_gate_vjp,
    )
    from blockspec.pypto_dsv41_moe_vjp import (
        make_clipped_swiglu_vjp,
        make_dequantize_blocked_weight,
        make_moe_combine_vjp,
        make_normalized_route_vjp, make_router_input_vjp, make_router_logits_vjp,
    )

    torch.set_num_threads(4)
    torch.manual_seed(41)
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )

    def compile_program(program, *arguments, config):
        if not args.resident_abi:
            return program.compile(*arguments, config=config)

        from blockspec.pypto_dsv41_resident_abi import compile_resident_entry, resident_entry
        from blockspec.pypto_dsv41_runtime import native

        chip = resident_entry(program)
        if "position_weights" in chip.param_names and len(arguments) == 6:
            arguments = (*arguments[:2], arguments[0], *arguments[2:])
        with TemporaryDirectory(prefix="dsv41-resident-abi-") as directory:
            options = RunConfig(
                arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                save_kernels=True, save_kernels_dir=directory,
            )
            compiled, _ = compile_resident_entry(
                chip, arguments, options, abi=native("network").network_parameter_abi,
                name=chip._func.__name__,
            )
            return compiled

    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    vocabulary = args.vocabulary
    attention_fixture = C1AAttentionVJPGeometry(
        local_heads=1, swa=0, compressed=0, padded_columns=64,
    )
    programs = {
        "tv": compile_program(make_tv(32, vocabulary),
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "kl": compile_program(make_kl(32, vocabulary),
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "adam": compile_program(make_adam((64, 128)),
            *[empty((64, 128), torch.float32) for _ in range(7)],
            1e-4, .1, .001, 1., config=config,
        ),
        "linear": compile_program(make_linear((32, 128), (64, 128)),
            empty((32, 128)), empty((64, 128)), empty((32, 64)), config=config,
        ),
        "linear_input_grad": compile_program(make_linear((32, 64), (128, 64)),
            empty((32, 64)), empty((128, 64)), empty((32, 128)), config=config,
        ),
        "linear_weight_grad": compile_program(make_linear(
            (32, 64), (32, 128), transpose_a=True, transpose_b=False, gradient_weight=True,
        ),empty((32, 64)), empty((32, 128)), empty((64, 128), torch.float32), config=config),
        "rms_forward": compile_program(make_rms(32, 128),
            empty((32, 128)), empty((1, 128), torch.float32), empty((32, 128)),
            empty((32, 8), torch.float32), config=config,
        ),
        "rms_input_grad": compile_program(make_rms(32, 128, mode="input"),
            empty((32, 128)), empty((1, 128), torch.float32), empty((32, 8), torch.float32),
            empty((32, 128)), empty((32, 128)), config=config,
        ),
        "rms_weight_grad": compile_program(make_rms(32, 128, mode="weight"),
            empty((32, 128)), empty((32, 8), torch.float32), empty((32, 128)),
            empty((1, 128), torch.float32), config=config,
        ),
        "silu_grad": compile_program(make_element((32, 128), "silu_grad"),
            empty((32, 128)), empty((32, 128)), empty((32, 128)), config=config,
        ),
        "publish": compile_program(make_publish((64, 128)),
            empty((64, 128), torch.float32), empty((64, 128)), config=config,
        ),
        # A small but genuinely linked resident update.  These four programs
        # share device handles in the validator, so the loss gradient is never
        # copied back to the host before the parameter VJP, Adam and publish.
        "linked_tv": compile_program(make_tv(32, 64),
            empty((32, 64)), empty((32, 64)), empty((32, 64)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "c1a_softmax_vjp": compile_program(make_softmax_vjp(attention_fixture),
            empty((attention_fixture.rows, attention_fixture.padded_columns), torch.float32),
            empty((attention_fixture.rows, attention_fixture.padded_columns)),
            empty((attention_fixture.rows, attention_fixture.padded_columns)),
            empty((attention_fixture.rows, 16), torch.float32), config=config,
        ),
        "c1a_sink_reduce": compile_program(make_sink_reduce(attention_fixture),
            empty((attention_fixture.rows, 16), torch.float32),
            empty((attention_fixture.local_heads, 16), torch.float32), config=config,
        ),
        "c1a_block_extract": compile_program(make_block_extract(attention_fixture),
            empty((attention_fixture.rows, attention_fixture.padded_columns)),
            empty((attention_fixture.rows, attention_fixture.block)), config=config,
        ),
        "hc_collapse_forward": compile_program(make_hc_collapse_forward(32, 128),
            empty((32 * 4, 128)), empty((32, 4), torch.float32),
            empty((32, 128)), config=config,
        ),
        "hc_collapse_vjp": compile_program(make_hc_collapse_vjp(32, 128),
            empty((32 * 4, 128)), empty((32, 4), torch.float32),
            empty((32, 128)), empty((32 * 4, 128)),
            empty((32, 16), torch.float32), config=config,
        ),
        "hc_post_data_vjp": compile_program(make_hc_post_data_vjp(32, 128),
            empty((32 * 4, 128)), empty((32, 4), torch.float32),
            empty((32 * 4, 4), torch.float32), empty((32, 128)),
            empty((32 * 4, 128)), config=config,
        ),
        "hc_post_gate_vjp": compile_program(make_hc_post_gate_vjp(32, 128),
            empty((32, 128)), empty((32 * 4, 128)), empty((32 * 4, 128)),
            empty((32, 16), torch.float32), empty((32 * 4, 16), torch.float32),
            config=config,
        ),
        "hc_pre_gate_vjp": compile_program(make_hc_pre_gate_vjp(),
            empty((32, 24), torch.float32), empty((3,), torch.float32),
            empty((24,), torch.float32), empty((32, 4), torch.float32),
            empty((32, 4), torch.float32), empty((32 * 4, 4), torch.float32),
            empty((32, 24), torch.float32), config=config,
        ),
        "moe_combine_vjp": compile_program(make_moe_combine_vjp(4, 128),
            empty((4, 128)), empty((4 * 6, 128)), empty((4, 6), torch.float32),
            empty((4, 128)), empty((4 * 6, 128)), empty((4, 16), torch.float32),
            config=config,
        ),
        "normalized_route_vjp": compile_program(make_normalized_route_vjp(4),
            empty((4, 6), torch.float32), empty((4, 6), torch.float32),
            empty((4, 16), torch.float32), 2.5, config=config,
        ),
        "router_logits_vjp": compile_program(make_router_logits_vjp(4),
            empty((4, 384), torch.float32), empty((4, 6), torch.int32),
            empty((4, 6), torch.float32), empty((4, 384), torch.float32), 2.5,
            config=config,
        ),
        "router_input_vjp": compile_program(make_router_input_vjp(4, 128),
            empty((4, 384), torch.float32), empty((4, 6), torch.int32),
            empty((384, 128), torch.float32), empty((4, 128)), config=config,
        ),
        "w8a8_vjp_weight": compile_program(make_dequantize_blocked_weight(256, 128),
            empty((256 * 2, 64), torch.int8), empty((128,), torch.float32),
            empty((256, 128)), config=config,
        ),
        "clipped_swiglu_vjp": compile_program(make_clipped_swiglu_vjp(4, 128),
            empty((4, 256)), empty((4, 128)),
            empty((4, 256)), config=config,
        ),
    }
    if not args.skip_full_shape_compile:
        production_attention = C1AAttentionVJPGeometry()
        programs["c1a_softmax_vjp_tp4_k32"] = compile_program(make_softmax_vjp(production_attention),
            empty((production_attention.rows, production_attention.padded_columns), torch.float32),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.rows, 16), torch.float32), config=config,
        )
        programs["c1a_sink_reduce_tp4_k32"] = compile_program(make_sink_reduce(production_attention),
            empty((production_attention.rows, 16), torch.float32),
            empty((production_attention.local_heads, 16), torch.float32), config=config,
        )
        programs["c1a_block_extract_tp4_k32"] = compile_program(make_block_extract(production_attention),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.rows, production_attention.block)), config=config,
        )
        programs["c1a_probability_block_extract_tp4_k32"] = compile_program(make_probability_block_extract(
            production_attention,
        ),
            empty((production_attention.rows, production_attention.padded_columns), torch.float32),
            empty((production_attention.rows, production_attention.block)), config=config,
        )
        factories = c1a_vjp_factories(production_attention)
        programs["c1a_joined_history_tp4_k32"] = compile_program(factories["joined_history"](),
            empty((128, production_attention.dimension)), empty((2, 3), torch.int32),
            empty((512, production_attention.dimension)),
            empty((production_attention.block, production_attention.compressed), torch.int32),
            empty((production_attention.block,), torch.int32),
            empty((production_attention.block, production_attention.dimension)),
            empty((128, production_attention.dimension)),
            empty((production_attention.block * production_attention.padded_columns,
                   production_attention.dimension)), config=config,
        )
        programs["c1a_probability_replay_tp4_k32"] = compile_program(factories["probability_replay"](),
            empty((production_attention.rows, production_attention.dimension)),
            empty((production_attention.block * production_attention.padded_columns,
                   production_attention.dimension)),
            empty((1, production_attention.swa), torch.float32),
            empty((production_attention.block, production_attention.compressed), torch.float32),
            empty((production_attention.local_heads,), torch.float32),
            empty((production_attention.rows, 1), torch.float32),
            empty((production_attention.rows, production_attention.padded_columns), torch.float32),
            config=config,
        )
        programs["c1a_probability_gradient_tp4_k32"] = compile_program(factories["d_probability"](),
            empty((production_attention.rows, production_attention.dimension)),
            empty((production_attention.block * production_attention.padded_columns,
                   production_attention.dimension)),
            empty((production_attention.rows, production_attention.padded_columns)), config=config,
        )
        programs["c1a_query_gradient_tp4_k32"] = compile_program(factories["d_query"](),
            empty((production_attention.rows, production_attention.padded_columns)),
            empty((production_attention.block * production_attention.padded_columns,
                   production_attention.dimension)),
            empty((production_attention.rows, production_attention.dimension)), config=config,
        )
        for name in ("d_key_block", "d_value_block"):
            programs[f"c1a_{name}_tp4_k32"] = compile_program(factories[name](),
                empty((production_attention.rows, production_attention.block)),
                empty((production_attention.rows, production_attention.dimension)),
                empty((production_attention.block, production_attention.dimension), torch.float32),
                config=config,
            )
        programs["hc_post_data_vjp_k32x5120"] = compile_program(make_hc_post_data_vjp(),
            empty((32 * 4, 5120)), empty((32, 4), torch.float32),
            empty((32 * 4, 4), torch.float32), empty((32, 5120)),
            empty((32 * 4, 5120)), config=config,
        )
        programs["hc_collapse_forward_k32x5120"] = compile_program(make_hc_collapse_forward(),
            empty((32 * 4, 5120)), empty((32, 4), torch.float32),
            empty((32, 5120)), config=config,
        )
        programs["hc_collapse_vjp_k32x5120"] = compile_program(make_hc_collapse_vjp(),
            empty((32 * 4, 5120)), empty((32, 4), torch.float32),
            empty((32, 5120)), empty((32 * 4, 5120)),
            empty((32, 16), torch.float32), config=config,
        )
        programs["hc_post_gate_vjp_k32x5120"] = compile_program(make_hc_post_gate_vjp(),
            empty((32, 5120)), empty((32 * 4, 5120)), empty((32 * 4, 5120)),
            empty((32, 16), torch.float32), empty((32 * 4, 16), torch.float32),
            config=config,
        )
        programs["moe_combine_vjp_k32x5120"] = compile_program(make_moe_combine_vjp(),
            empty((32, 5120)), empty((32 * 6, 5120)), empty((32, 6), torch.float32),
            empty((32, 5120)), empty((32 * 6, 5120)), empty((32, 16), torch.float32),
            config=config,
        )
        programs["normalized_route_vjp_k32"] = compile_program(make_normalized_route_vjp(),
            empty((32, 6), torch.float32), empty((32, 6), torch.float32),
            empty((32, 16), torch.float32), 2.5, config=config,
        )
        programs["router_logits_vjp_k32x384"] = compile_program(make_router_logits_vjp(),
            empty((32, 384), torch.float32), empty((32, 6), torch.int32),
            empty((32, 6), torch.float32), empty((32, 384), torch.float32), 2.5,
            config=config,
        )
        programs["router_input_vjp_k32x5120"] = compile_program(make_router_input_vjp(32, 5120),
            empty((32, 384), torch.float32), empty((32, 6), torch.int32),
            empty((384, 5120), torch.float32), empty((32, 5120)), config=config,
        )
        programs["moe_down_vjp_weight_2304x5120"] = compile_program(make_dequantize_blocked_weight(
            2304, 5120,
        ),
            empty((2304 * (5120 // 64), 64), torch.int8),
            empty((5120,), torch.float32), empty((2304, 5120)), config=config,
        )
        programs["moe_gate_up_vjp_weight_5120x4608"] = compile_program(make_dequantize_blocked_weight(
            5120, 4608,
        ),
            empty((5120 * (4608 // 64), 64), torch.int8),
            empty((4608,), torch.float32), empty((5120, 4608)), config=config,
        )
        programs["clipped_swiglu_vjp_k32x2304"] = compile_program(make_clipped_swiglu_vjp(),
            empty((32, 4608)), empty((32, 2304)),
            empty((32, 4608)), config=config,
        )
        # The frozen LM head is not updated, but its transpose is required to
        # propagate dLogits into the final trainable draft layer. Compile the
        # real V4.1 shape so a small fixture cannot hide an ABI/tile failure.
        local_vocabulary = vocabulary // 4
        if vocabulary % 256:
            parser.error("full-shape TP4 validation requires a 256-aligned vocabulary")
        programs["lm_head_input_grad_tp4"] = compile_program(make_linear(
            (32, local_vocabulary), (5120, local_vocabulary), reduction_limit=64,
        ),
            empty((32, local_vocabulary)), empty((5120, local_vocabulary)), empty((32, 5120)), config=config,
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
            programs[f"{name}_input_grad"] = compile_program(make_linear(
                (32, outputs), (inputs, outputs), reduction_limit=64,
            ),
                empty((32, outputs)), empty((inputs, outputs)), empty((32, inputs)), config=config,
            )
            programs[f"{name}_weight_grad"] = compile_program(make_linear(
                (32, outputs), (32, inputs), transpose_a=True, transpose_b=False,
                gradient_weight=True, reduction_limit=32,
            ),
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
        }
        for name, (outputs, inputs) in frozen_tail_linears.items():
            programs[name] = compile_program(make_linear(
                (32, outputs), (inputs, outputs), reduction_limit=64,
            ),
                empty((32, outputs)), empty((inputs, outputs)), empty((32, inputs)), config=config,
            )
    if args.resident_abi:
        programs.update(compile_resident_update_extras(compile_program, empty, config))
    report = {
        "format": "orthrus-dsv41-pypto-online-programs-v1",
        "passed": False,
        "compiled": sorted(programs),
        "executed": False,
        "vocabulary": vocabulary,
        "cases": [],
    }
    if args.resident_abi:
        report["entry"] = "resident-l2"
        report["resident_c1a_dispatches"] = check_c1a_resident_dispatch(empty)
    if args.compile_only:
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.resident_abi:
            print(json.dumps({"stage": "resident-online-abi", "passed": True,
                              "compiled": len(programs), "npu_execution": False,
                              "resident_c1a_dispatches": report["resident_c1a_dispatches"]}))
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

        gate_up = (torch.randn(4, 256) * 7).bfloat16()
        # Clamp endpoints are representable in BF16 and must keep their
        # derivative, matching torch.clamp/autograd exactly.
        gate_up[0, 0] = 10
        gate_up[0, 128] = -10
        gate_up[1, 1] = 10
        gate_up[1, 129] = 10
        expert_dy = torch.randn(4, 128).bfloat16()
        gate_up_gradient_handle = allocate(torch.zeros(4, 256, dtype=torch.bfloat16))
        worker.run(
            programs["clipped_swiglu_vjp"], allocate(gate_up), allocate(expert_dy),
            gate_up_gradient_handle, config=config,
        )
        gate, up = gate_up.float().chunk(2, dim=-1)
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
            actual_score_gradient,
            (expected_score_gradient * attention_fixture.dimension ** -.5).bfloat16(),
            atol=.002, rtol=.01,
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
        joined_gradient = torch.randn(
            attention_fixture.rows, attention_fixture.padded_columns,
        ).bfloat16()
        block_gradient_handle = allocate(torch.zeros(
            attention_fixture.rows, attention_fixture.block, dtype=torch.bfloat16,
        ))
        worker.run(
            programs["c1a_block_extract"], allocate(joined_gradient), block_gradient_handle, config=config,
        )
        torch.testing.assert_close(
            read(block_gradient_handle, torch.empty(
                attention_fixture.rows, attention_fixture.block, dtype=torch.bfloat16,
            )),
            joined_gradient[:, attention_fixture.block_begin:
                            attention_fixture.block_begin + attention_fixture.block],
            atol=0, rtol=0,
        )
        report["cases"].append({
            "operator": "c1a_attention_vjp",
            "block": attention_fixture.block,
            "sink_column": attention_fixture.sink_column,
            "padded_columns": attention_fixture.padded_columns,
        })

        collapse_streams = torch.randn(32, 4, 128).bfloat16()
        collapse_premix = torch.randn(32, 4, dtype=torch.float32)
        collapse_dy = torch.randn(32, 128).bfloat16()
        collapse_output = allocate(torch.zeros(32, 128, dtype=torch.bfloat16))
        worker.run(
            programs["hc_collapse_forward"], allocate(collapse_streams.flatten(0, 1)),
            allocate(collapse_premix), collapse_output, config=config,
        )
        expected_collapse = torch.einsum(
            "ti,tid->td", collapse_premix, collapse_streams.float(),
        ).bfloat16()
        torch.testing.assert_close(
            read(collapse_output, torch.empty_like(expected_collapse)),
            expected_collapse, atol=.016, rtol=.016,
        )
        collapse_stream_gradient = allocate(torch.zeros(32 * 4, 128, dtype=torch.bfloat16))
        collapse_premix_gradient = allocate(torch.zeros(32, 16, dtype=torch.float32))
        worker.run(
            programs["hc_collapse_vjp"], allocate(collapse_streams.flatten(0, 1)),
            allocate(collapse_premix), allocate(collapse_dy),
            collapse_stream_gradient, collapse_premix_gradient, config=config,
        )
        expected_stream_gradient = (
            collapse_dy.float().unsqueeze(1) * collapse_premix.unsqueeze(-1)
        ).bfloat16()
        expected_premix_gradient = torch.einsum(
            "td,tid->ti", collapse_dy.float(), collapse_streams.float(),
        )
        torch.testing.assert_close(
            read(collapse_stream_gradient, torch.empty_like(expected_stream_gradient.flatten(0, 1))),
            expected_stream_gradient.flatten(0, 1), atol=.016, rtol=.016,
        )
        torch.testing.assert_close(
            read(collapse_premix_gradient, torch.empty(32, 16))[:, :4],
            expected_premix_gradient, atol=.02, rtol=.002,
        )
        report["cases"].append({"operator": "hc_collapse_forward_vjp", "shape": [32, 4, 128]})

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
        router_logits = torch.randn(4, 384, dtype=torch.float32)
        route_ids = torch.stack([torch.randperm(384)[:6] for _ in range(4)]).int()
        selected_logits = router_logits.gather(1, route_ids.long())
        scores = torch.nn.functional.softplus(selected_logits).sqrt()
        probability = scores / scores.sum(-1, keepdim=True)
        expected_score_grad = 2.5 / scores.sum(-1, keepdim=True) * (
            expected_weight_grad
            - (expected_weight_grad * probability).sum(-1, keepdim=True)
        )
        expected_selected_logit_grad = expected_score_grad * selected_logits.sigmoid() / (2. * scores)
        router_grad_handle = allocate(torch.full((4, 384), float("nan")))
        worker.run(
            programs["router_logits_vjp"], allocate(router_logits), allocate(route_ids),
            allocate(actual_weight_grad.contiguous()), router_grad_handle, 2.5, config=config,
        )
        actual_router_grad = read(router_grad_handle, torch.empty(4, 384))
        expected_router_grad = torch.zeros(4, 384).scatter(
            1, route_ids.long(), expected_selected_logit_grad,
        )
        torch.testing.assert_close(actual_router_grad, expected_router_grad, atol=.01, rtol=.002)
        gate = torch.randn(384, 128)
        router_input = allocate(torch.empty(4, 128, dtype=torch.bfloat16))
        worker.run(
            programs["router_input_vjp"], router_grad_handle, allocate(route_ids),
            allocate(gate), router_input, config=config,
        )
        torch.testing.assert_close(
            read(router_input, torch.empty(4, 128, dtype=torch.bfloat16)),
            (actual_router_grad @ gate).bfloat16(), atol=.01, rtol=.0079,
        )
        report["cases"].append({"operator": "fixed_route_moe_vjp", "shape": [4, 6, 128]})
        report["cases"].append({"operator": "router_logits_vjp", "shape": [4, 384]})
        report["cases"].append({"operator": "router_input_vjp", "shape": [4, 128]})

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

        # Exercise the production hand-off pattern without host materializing
        # any intermediate: TV dLogits -> projection dWeight -> Adam candidate
        # -> BF16 execution bank.  The final readback is validation only.
        linked_student = torch.randn(32, 64, dtype=torch.bfloat16)
        linked_teacher = torch.randn_like(linked_student)
        linked_input = torch.randn(32, 128, dtype=torch.bfloat16)
        linked_master = torch.randn(64, 128, dtype=torch.float32)
        linked_gradient = allocate(torch.zeros(32, 64, dtype=torch.bfloat16))
        linked_losses = allocate(torch.zeros(32, 8, dtype=torch.float32))
        linked_parameter_gradient = allocate(torch.zeros(64, 128, dtype=torch.float32))
        linked_state = [
            allocate(linked_master),
            allocate(torch.zeros_like(linked_master)),
            allocate(torch.zeros_like(linked_master)),
        ]
        linked_next = [allocate(torch.zeros_like(linked_master)) for _ in range(3)]
        linked_execution = allocate(torch.zeros(64, 128, dtype=torch.bfloat16))
        worker.run(
            programs["linked_tv"], allocate(linked_student), allocate(linked_teacher),
            linked_gradient, linked_losses, 7, 7, config=config,
        )
        worker.run(
            programs["linear_weight_grad"], linked_gradient, allocate(linked_input),
            linked_parameter_gradient, config=config,
        )
        worker.run(
            programs["adam"], linked_state[0], linked_parameter_gradient,
            linked_state[1], linked_state[2], *linked_next,
            1e-4, .1, .001, 1., config=config,
        )
        worker.run(programs["publish"], linked_next[0], linked_execution, config=config)

        linked_student_reference = linked_student.float().requires_grad_()
        linked_teacher_log = linked_teacher.float().log_softmax(-1)
        linked_loss_reference = .5 * (
            linked_student_reference.log_softmax(-1).exp() - linked_teacher_log.exp()
        ).abs().sum(-1)[:7].mean()
        linked_dlogits, = torch.autograd.grad(linked_loss_reference, linked_student_reference)
        linked_dw = (linked_dlogits.bfloat16().T @ linked_input).bfloat16().float()
        linked_mean = .1 * linked_dw
        linked_variance = .001 * linked_dw.square()
        linked_expected = linked_master - 1e-4 * (linked_mean / .1) / (
            (linked_variance / .001).sqrt() + 1e-8
        )
        torch.testing.assert_close(
            read(linked_execution, torch.empty(64, 128, dtype=torch.bfloat16)),
            linked_expected.bfloat16(), atol=.016, rtol=.004,
        )
        report["cases"].append({
            "operator": "linked_loss_vjp_adam_publish",
            "device_intermediates": ["dlogits", "parameter_gradient", "adam_state"],
            "valid": 7,
        })
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
        args.output.chmod(0o644)


if __name__ == "__main__":
    main()
