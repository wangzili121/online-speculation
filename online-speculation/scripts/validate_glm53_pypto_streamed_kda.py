#!/usr/bin/env python3
"""Run a frozen GLM KDA layer's native replay and backward, one TP8 shard at a time."""

import argparse
import json
import sys
from pathlib import Path

import torch

from blockspec import pypto_glm53_online_kda as kernels


PROJECTION = ("w_q", "w_k", "w_v", "w_f_a", "w_f_b", "dt_bias", "a_log", "w_b", "w_g_a", "w_g_b", "weight")


def reference(x, initial_conv, initial_state, weights, upstream, flash):
    """FP64 reference with the forward's BF16 casts and identity cast adjoints."""
    tokens, heads, dim = len(x), len(initial_state), initial_state.shape[-1]
    x = x.double().requires_grad_()

    def bf16(value):
        return value + (value.bfloat16().double() - value).detach()

    def linear(value, name):
        return bf16(value @ weights[name].double().T)

    mixed = torch.cat(tuple(linear(x, name) for name in PROJECTION[:3]), dim=-1)
    padded = torch.cat((initial_conv.double(), mixed))
    conv = (padded.unfold(0, 4, 1) * weights["weight"].double().T).sum(-1)
    q, k, v = bf16(torch.nn.functional.silu(conv)).reshape(tokens, 3, heads, dim).unbind(1)

    def normalized(value):
        return value * (value.square().sum(-1, keepdim=True) + 1e-6).rsqrt()

    q, k = normalized(q) * dim ** -0.5, normalized(k)
    g = linear(linear(x, "w_f_a"), "w_f_b") + weights["dt_bias"].double()
    log_decay = (g.view(tokens, heads, dim) * weights["a_log"].double().exp()[None, :, None]).sigmoid()
    decay = (log_decay * flash.linear_lower_bound).exp()
    beta = linear(x, "w_b").sigmoid()
    gate = linear(linear(x, "w_g_a"), "w_g_b").view(tokens, heads, dim)
    state = initial_state.double()
    outputs = []
    for t in range(tokens):
        previous = state * decay[t, :, None, :]
        memory = (previous * k[t, :, None, :]).sum(-1)
        delta = (v[t] - memory) * beta[t, :, None]
        state = previous + delta[..., None] * k[t, :, None, :]
        outputs.append((state * q[t, :, None, :]).sum(-1))
    core = bf16(torch.stack(outputs))
    inverse = (core.square().mean(-1, keepdim=True) + flash.rms_norm_eps).rsqrt()
    combined = bf16(core * inverse * weights["norm_weight"].double() * gate.sigmoid())
    y = linear(combined.reshape(tokens, -1), "w_o")
    (gradient,) = torch.autograd.grad(y, x, upstream.double())
    return gradient, core.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tp-ranks", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resident-replay-seed", action="store_true",
                        help="seed the graph from resident pre-norm activations and committed states")
    parser.add_argument("--resident-adjoint", action="store_true",
                        help="use the resident KDA suffix programs and reduce the input gradient")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.resident_adjoint:
        args.resident_replay_seed = True
    if (len(set(args.tp_ranks)) != len(args.tp_ranks)
            or any(rank not in range(8) for rank in args.tp_ranks) or args.output.exists()):
        parser.error("use distinct logical TP8 ranks and a fresh output path")

    # Keep the native kernel shard geometry at TP8 while running on one physical worker.
    sys.argv.extend(("--tp", "8", "--ep", "8"))
    from models.glm5_3_flash.config import D, FLASH, KDA_DIM
    from models.glm5_3_flash.kda import BETA_PAD
    from models.glm5_3_flash.rmsnorm import golden_rmsnorm_quant
    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, StackedDeviceTensor

    torch.set_num_threads(4)
    checkpoint = Checkpoint(args.checkpoint)
    tokens, active, heads, dim = 16, 6, 8, KDA_DIM
    local = heads * dim
    rows = tokens * heads
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    generator = torch.Generator().manual_seed(191)
    x = (torch.randn(tokens, D, generator=generator) * 0.25).bfloat16()
    upstream = torch.randn(tokens, D, generator=generator) * 0.025
    upstream[active:] = 0
    starts = torch.tensor([0, tokens], dtype=torch.int32)
    slots = torch.tensor([[0, tokens, *range(1, tokens + 1)]], dtype=torch.int32)
    state_bank = torch.zeros((tokens + 1, heads, dim, dim))
    conv_bank = torch.zeros((tokens + 1, 3, 3 * local), dtype=torch.bfloat16)

    values = {"x": x, "starts": starts, "slots": slots, "upstream": upstream,
              "states": state_bank, "conv_states": conv_bank}
    if args.resident_adjoint:
        for name in ("partial", "reduced", "collapsed_grad"):
            values[name] = torch.empty((tokens, D))
    for name in ("q", "k", "v", "core", "gate"):
        values[name] = torch.empty((tokens, heads, dim), dtype=torch.bfloat16)
    values["mixed"] = torch.empty((tokens, 3 * local), dtype=torch.bfloat16)
    values["decay"] = torch.empty((tokens, heads, dim))
    values["beta"] = torch.empty((tokens, BETA_PAD))
    for name in ("q_norm", "k_norm", "value_fp", "core_grad", "gate_grad", "dq", "dk", "dv",
                 "d_decay", "dq_conv", "dk_conv"):
        values[name] = torch.empty(rows, dim)
    values["beta_per_head"] = torch.empty(rows, 8)
    values["d_beta"] = torch.empty(rows, 8)
    values["db_proj"] = torch.empty(tokens, heads)
    values["conv_grad"] = torch.empty(tokens, 3 * local)
    values["mixed_grad"] = torch.empty(tokens, 3 * local)
    values["dx"] = torch.empty(tokens, D)

    first_rank, *following = args.tp_ranks
    first_weights = checkpoint.attention(44, tp_rank=first_rank, tp=8)
    names = (*PROJECTION, "w_o", "norm_weight")
    values.update({name: first_weights[name].contiguous() for name in names})
    seed_keys = ("collapsed", "input_norm", "conv_q", "conv_k", "conv_v", "committed_conv",
                 "committed_recurrence", "x", "weight", "conv_states", "states")
    if args.resident_replay_seed:
        input_norm = checkpoint.read(checkpoint.layer_key(44, "input_layernorm.weight")).bfloat16()
        values.update(collapsed=x.clone(), input_norm=input_norm, committed_conv=conv_bank[0].clone(),
                      committed_recurrence=state_bank[0].clone())
        for name, weight in zip(("conv_q", "conv_k", "conv_v"), first_weights["weight"].split(local, 1), strict=True):
            values[name] = weight.T[:, None, :].contiguous()
    views = {
        **{name + "_flat": (name, (rows, dim)) for name in ("q", "k", "v", "core", "gate", "decay")},
        "gate_grad_matrix": ("gate_grad", (tokens, local)),
        "decay_matrix": ("decay", (tokens, local)),
        "d_decay_matrix": ("d_decay", (tokens, local)),
        "states_flat": ("states", ((tokens + 1) * heads * dim, dim)),
        "initial_conv": ("conv_states", (3, 3 * local)),
    }
    stages = (
        (kernels.make_kda_training_replay(tokens, ranks=1 if args.resident_replay_seed else None),
         ("x", "starts", "slots", *PROJECTION, "conv_states", "states", "mixed", "decay", "beta", "gate", "q", "k", "v", "core")),
        (kernels.make_kda_recurrence_inputs(tokens, heads, BETA_PAD),
         ("q_flat", "k_flat", "v_flat", "beta", "q_norm", "k_norm", "value_fp", "beta_per_head")),
        (kernels.make_kda_output_backward(tokens, heads, D, FLASH.rms_norm_eps),
         ("upstream", "core_flat", "gate_flat", "w_o", "norm_weight", "core_grad", "gate_grad")),
        (kernels.make_kda_recurrence_backward(tokens, heads),
         ("q_norm", "k_norm", "value_fp", "decay_flat", "beta_per_head", "states_flat", "core_grad", "dq", "dk", "dv", "d_decay", "d_beta")),
        (kernels.make_kda_qk_backward(tokens, heads),
         ("q_flat", "k_flat", "dq", "dk", "dq_conv", "dk_conv")),
        (kernels.make_kda_projection_inputs(tokens, heads),
         ("dq_conv", "dk_conv", "dv", "d_beta", "conv_grad", "db_proj")),
        (kernels.make_kda_convolution_backward(tokens, 3 * local),
         ("mixed", "initial_conv", "weight", "conv_grad", "mixed_grad")),
        (kernels.make_kda_projection_backward(tokens, heads, D, FLASH.linear_lower_bound, beta_width=BETA_PAD),
         ("mixed_grad", "decay_matrix", "a_log", "d_decay_matrix", "beta", "db_proj", "gate_grad_matrix",
          "w_q", "w_k", "w_v", "w_f_a", "w_f_b", "w_b", "w_g_a", "w_g_b", "dx")),
    )
    report = {"passed": False, "physical_devices": 1, "layer": 44,
              "resident_weight_mib": sum(values[name].nbytes for name in names) / 2**20, "cases": []}
    try:
        metadata = dict(values)
        metadata.update({view: torch.empty(shape, dtype=values[source].dtype) for view, (source, shape) in views.items()})
        compiled = []
        for index, (entry, inputs) in enumerate(stages[:1] if args.resident_adjoint else stages):
            report["compiled"] = index
            signature_only = index == 0 and args.resident_replay_seed
            bindings = () if signature_only else tuple(metadata[name] for name in inputs)
            compiled.append(entry.compile(*bindings, config=config))
        report["compiled"] = len(compiled)
        seed = kernels.make_kda_replay_seed(1).compile(config=config) if args.resident_replay_seed else None
        all_programs = [seed, *compiled] if seed is not None else compiled
        adjoints = []
        if args.resident_adjoint:
            adjoints = [entry.compile(config=config) for entry in kernels.make_kda_resident_backward(1)]
            all_programs.extend(adjoints)
        with DistributedWorker(all_programs, config=config, persistent=True, reset_persistent_windows=False) as worker:
            device = {name: worker.alloc_tensor(value.shape, value.dtype, init=value.share_memory_())
                      for name, value in values.items()}
            device.update({view: worker.view_tensor(device[source], shape) for view, (source, shape) in views.items()})

            def stacked(name):
                return StackedDeviceTensor([device[name]], (1, *device[name].shape), [0])

            def upload(name, value):
                host = value.contiguous().share_memory_()
                worker.copy_to(device[name].data_ptr, host.data_ptr(), host.nbytes)

            def download(name):
                host = torch.empty_like(values[name]).share_memory_()
                worker.copy_from(host.data_ptr(), device[name].data_ptr, host.nbytes)
                return host

            for epoch, rank in enumerate((first_rank, *following), start=1):
                weights = first_weights if rank == first_rank else checkpoint.attention(44, tp_rank=rank, tp=8)
                if rank != first_rank:
                    for name in names:
                        upload(name, weights[name])
                conv_bank[0].copy_((torch.randn(conv_bank[0].shape, generator=generator) * 0.1).bfloat16())
                state_bank[0].normal_(std=0.015, generator=generator)
                if seed is None:
                    upload("conv_states", conv_bank)
                    upload("states", state_bank)
                    replay_x = x
                else:
                    upload("committed_conv", conv_bank[0])
                    upload("committed_recurrence", state_bank[0])
                    for name, weight in zip(("conv_q", "conv_k", "conv_v"), weights["weight"].split(local, 1), strict=True):
                        upload(name, weight.T[:, None, :])
                    worker.run(seed, *(stacked(name) for name in seed_keys), config=config)
                    for name, expected_initial in (("conv_states", conv_bank[0]), ("states", state_bank[0])):
                        torch.testing.assert_close(download(name)[0], expected_initial, atol=0, rtol=0)
                    torch.testing.assert_close(download("weight"), weights["weight"], atol=0, rtol=0)
                    replay_x, _, _ = golden_rmsnorm_quant(x, input_norm)
                    torch.testing.assert_close(download("x").float(), replay_x.float(), atol=1e-4, rtol=0.008)
                    replay_x = download("x")
                expected, expected_core = reference(replay_x, conv_bank[0], state_bank[0], weights, upstream, FLASH)
                if args.resident_adjoint:
                    worker.run(compiled[0], *(stacked(name) for name in stages[0][1]), config=config)
                    adjoint_inputs = (
                        ("upstream", "core", "gate", "q", "k", "v", "beta", "decay", "states", "mixed",
                         "initial_conv", "weight", "w_o", "norm_weight", "mixed_grad", "d_decay", "db_proj", "gate_grad"),
                        ("mixed_grad", "decay", "a_log", "d_decay", "beta", "db_proj", "gate_grad",
                         "w_q", "w_k", "w_v", "w_f_a", "w_f_b", "w_b", "w_g_a", "w_g_b", "partial"),
                        ("collapsed", "input_norm", "partial", "reduced", "collapsed_grad"),
                    )
                    for index, (program, keys) in enumerate(zip(adjoints, adjoint_inputs, strict=True)):
                        scalars = (epoch,) if index == len(adjoints) - 1 else ()
                        worker.run(program, *(stacked(name) for name in keys), *scalars, config=config)
                    got = download("collapsed_grad")
                    collapsed = x.double().requires_grad_()
                    normed = collapsed * (collapsed.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt()
                    expected, = torch.autograd.grad(normed * input_norm.double(), collapsed, expected)
                else:
                    for index, (program, (_, keys)) in enumerate(zip(compiled, stages, strict=True)):
                        arguments = (stacked(name) for name in keys) if index == 0 and seed is not None else (
                            device[name] for name in keys
                        )
                        worker.run(program, *arguments, config=config)
                    got = download("dx")
                core = download("core")
                torch.testing.assert_close(core.float(), expected_core.float(), atol=2e-5, rtol=0.025)
                torch.testing.assert_close(got.double(), expected, atol=8e-6, rtol=0.012)
                if got[active:].count_nonzero():
                    raise AssertionError("inactive KDA rows acquired gradients")
                report["cases"].append({"tp_rank": rank, "core_max_abs": float((core - expected_core).abs().max()),
                                        "dx_max_abs": float((got - expected).abs().max())})
        report["resident_replay_seed"] = args.resident_replay_seed
        report["resident_adjoint"] = args.resident_adjoint
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
