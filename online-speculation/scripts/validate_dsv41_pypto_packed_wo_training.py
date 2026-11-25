#!/usr/bin/env python3
"""Check the captured BF16 Wo operands and the resident parameter VJPs."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from validate_dsv41_pypto_packed_wo import reference_output


def make_capture(block, world, model, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from blockspec.pypto_dsv41_draft import make_packed_draft_output, make_packed_draft_output_tape

    hidden, dim = model.hidden_size, model.head_dim
    heads, groups = model.num_attention_heads // tp, model.o_groups // tp
    middle, width = groups * model.o_lora_rank, heads * dim
    a_shape = (groups, width // groups, model.o_lora_rank)
    b_shape = (middle, hidden)
    phase = (block, model.qk_rope_head_dim // 2)
    head_major = int(block == 16)
    project = pl.jit.inline(auto_scope=False)(make_packed_draft_output(block_size=block,
                                                                       head_major=bool(head_major))._func)
    capture = make_packed_draft_output_tape(block, 2)

    @pl.jit
    def forward(
        attention: pl.Tensor[[block * heads, dim], pl.FP32],
        a: pl.Tensor[a_shape, pl.BF16, pl.NZ], b: pl.Tensor[b_shape, pl.BF16, pl.NZ],
        cos: pl.Tensor[phase, pl.FP32], sin: pl.Tensor[phase, pl.FP32],
        out: pl.Out[pl.Tensor[[block, hidden], pl.BF16]],
        rotated: pl.InOut[pl.Tensor[[block * 2, width], pl.BF16]],
        latent: pl.InOut[pl.Tensor[[block * 2, middle], pl.BF16]],
        slot: pl.Scalar[pl.INT32],
    ):
        if slot >= -1:
            capture(attention, a, b, cos, sin, out, rotated, latent, head_major, slot)
        else:
            project(attention, a, b, cos, sin, out)
        return out, rotated, latent

    @pl.jit.host
    def dispatch(
        attention: pl.Tensor[[world, block * heads, dim], pl.FP32],
        a: pl.Tensor[[world, *a_shape], pl.BF16, pl.NZ], b: pl.Tensor[[world, *b_shape], pl.BF16, pl.NZ],
        cos: pl.Tensor[[world, *phase], pl.FP32], sin: pl.Tensor[[world, *phase], pl.FP32],
        out: pl.Out[pl.Tensor[[world, block, hidden], pl.BF16]],
        rotated: pl.InOut[pl.Tensor[[world, block * 2, width], pl.BF16]],
        latent: pl.InOut[pl.Tensor[[world, block * 2, middle], pl.BF16]],
        slot: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            forward(attention[rank], a[rank], b[rank], cos[rank], sin[rank], out[rank], rotated[rank],
                    latent[rank], slot, device=rank)
    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    world, block = len(devices), args.block
    if not args.compile_only and args.checkpoint is None:
        parser.error("--checkpoint is required for execution")

    import torch
    from blockspec.dsv41_packed_wo_vjp import PackedWoInputVJP, make_packed_wo_input_vjp
    from blockspec.pypto_dsv41_draft import OUTPUT_TENSORS, pack_draft_output_projection, prepare_output
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE as tp
    from models.deepseek_v4_1_flash_a3.device_views import resident_graph_argument
    from models.deepseek_v4_1_flash_a3.o_proj import golden_grouped_output, rope_interleave
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, bf16_nz_shape, unpack_bf16_nz
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    backward = make_packed_wo_input_vjp(block, model, tp, world=world, parameters=True).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, tp=tp, world=world)))
        return
    forward = make_capture(block, world, model, tp).compile(config=config)
    checkpoint = Checkpoint(args.checkpoint)
    hidden, dim = model.hidden_size, model.head_dim
    tail, heads = model.qk_rope_head_dim, model.num_attention_heads // tp
    groups, rank_dim = model.o_groups // tp, model.o_lora_rank
    width, middle = heads * dim, groups * rank_dim
    position, nr, layer = 5, 2, model.num_hidden_layers - 1
    bank_slot, bank_layers = 2, 3
    rng = torch.Generator().manual_seed(89316)
    angle = torch.randn(2, block * 2, tail // 2, generator=rng)
    cosine, sine = angle.cos(), angle.sin()
    cos, sin = cosine[1, position:position + block], sine[1, position:position + block]
    raw = {key: checkpoint.tensor(f"layers.{layer}.attn.{key}") for key in OUTPUT_TENSORS}

    with DistributedWorker([forward, backward], config, persistent=True) as runtime:
        ranks = tuple(range(world))

        def upload(value, device):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=device)

        def read(handle, device):
            result = torch.empty(handle.shape, dtype=handle.dtype)
            runtime.copy_from(result.data_ptr(), handle.data_ptr, result.nbytes, worker_id=device)
            return result

        def stack(handles):
            return StackedDeviceTensor(handles, (world, *handles[0].shape), ranks)

        bank, inputs, forwards, tapes, seeds, weights = [], [], [], {}, [], []
        for device in ranks:
            w = prepare_output(raw, asdict(model), tp_rank=device % tp, tp=tp)
            weights.append(w)
            output = pack_draft_output_projection(w)
            a = output["wo_a"].reshape(bf16_nz_shape(*output["wo_a"].shape))
            b = output["wo_b"].reshape(bf16_nz_shape(1, *output["wo_b"].shape))
            # Retain the real byte stride and a nonzero layer offset.
            bank_entry = dict(freqs_cos=upload(cosine, device), freqs_sin=upload(sine, device))
            for name, value in (("draft_wo_a", a), ("draft_wo_b", b)):
                full = runtime.alloc_tensor((bank_layers * value.shape[0], *value.shape[1:]),
                                            value.dtype, worker_id=device)
                runtime.copy_to(full.data_ptr, value.data_ptr(), value.nbytes,
                                dst_offset=bank_slot * value.nbytes, worker_id=device)
                bank_entry[name] = full
            bank.append(bank_entry)
            attention = torch.randn(block * heads, dim, generator=rng) * .5
            inputs.append(attention)
            seed = torch.empty(block, hidden)
            seeds.append(upload(seed, device))
            tape = {key: upload(torch.full((nr * block, columns), -7, dtype=torch.bfloat16), device)
                    for key, columns in (("attention_tape_wo_input", width), ("attention_tape_wo_latent", middle))}
            tapes[device] = tape
            source = attention.reshape(block, heads, dim).transpose(0, 1).reshape(-1, dim) if block == 16 else attention
            forwards.append((upload(source, device), upload(a, device), upload(b, device),
                             upload(cos, device), upload(sin, device),
                             upload(torch.empty(block, hidden, dtype=torch.bfloat16), device), *tape.values()))
        args_device = [resident_graph_argument(list(handles), ranks, parameter)
                       for handles, parameter in zip(zip(*forwards, strict=True), forward._get_metadata()[0][:-1],
                                                     strict=True)]
        timing = {slot: [] for slot in (-2, -1, 1)}
        for _ in range(8):
            for slot in timing:
                start = perf_counter()
                runtime.run(forward, *args_device, slot, config=config)
                timing[slot].append((perf_counter() - start) * 1000)
        captured = [read(handles[5], device) for device, handles in enumerate(forwards)]
        for mode in (-2, -1):
            runtime.run(forward, *args_device, mode, config=config)
            for device, handles in enumerate(forwards):
                assert torch.equal(captured[device], read(handles[5], device)), (mode, device)
        state = []
        for device in ranks:
            xr, z = (read(tape, device) for tape in tapes[device].values())
            assert torch.all(xr[:block] == -7) and torch.all(z[:block] == -7)
            original = inputs[device].reshape(block, heads, dim).bfloat16()
            rotated = torch.cat((original[..., :-tail], rope_interleave(original[..., -tail:], cos, sin, inverse=True)),
                                dim=-1).reshape(block, width)
            torch.testing.assert_close(xr[block:], rotated, atol=.0005, rtol=.008)
            w = weights[device]
            latent = golden_grouped_output(xr[block:].reshape(block, groups, -1), w["wo_a"].transpose(-2, -1))
            torch.testing.assert_close(z[block:], latent.flatten(1), atol=.001, rtol=.008)
            ref_output = (z[block:].float() @ w["wo_b"].float()).bfloat16()
            torch.testing.assert_close(captured[device], ref_output, atol=.003, rtol=1 / 128)
            state.append((xr[block:], z[block:]))

        proposer = SimpleNamespace(runtime=runtime, config=config, world=world, block=block, tp=tp, ranks=bank,
                                   last_layers=nr)
        geometry = SimpleNamespace(**{**asdict(model), "num_hidden_layers": bank_layers})
        actor = PackedWoInputVJP(proposer, backward, geometry, parameters=True)
        cases = []
        for active in (1, 7, block):
            seed = torch.randn(block, hidden, generator=rng) * .1
            for device in ranks:
                runtime.copy_to(seeds[device].data_ptr, seed.data_ptr(), seed.nbytes, worker_id=device)
            student = SimpleNamespace(owner=0, position=position, rows=active, ffn_tape=tapes)
            gradient = SimpleNamespace(owner=0, layer=bank_slot, rows=active, delta=stack(seeds), residual=None)
            runs = [actor(student, gradient) for _ in range(5)]
            result = runs[-1]
            metrics = []
            for device, (xr, z) in enumerate(state):
                w = weights[device]
                delta = seed[:active].double()
                latent_gradient = delta @ w["wo_b"].double().T
                ref_a = torch.einsum("tgi,tgo->gio", xr[:active].double().reshape(active, groups, -1),
                                     latent_gradient.reshape(active, groups, rank_dim))
                ref_b = z[:active].double().T @ delta
                a_grad = read(result.parameter_gradients["draft_wo_a"].shards[device], device)
                b_grad = read(result.parameter_gradients["draft_wo_b"].shards[device], device)
                actual_a = unpack_bf16_nz(a_grad.reshape(ref_a.shape))
                actual_b = unpack_bf16_nz(b_grad.reshape(ref_b.shape))
                actual_context = read(result.context.shards[device], device)
                ref_context = reference_output(seed[:active], w, cos[:active], sin[:active], heads, rank_dim)
                assert torch.count_nonzero(actual_context[active * heads:]) == 0
                row = {}
                for name, actual, reference in (("wo_a", actual_a, ref_a), ("wo_b", actual_b, ref_b),
                                                ("context", actual_context[:active * heads], ref_context)):
                    error = actual.double() - reference
                    relative = float(error.norm() / reference.norm())
                    assert relative < 2e-5, (name, device, active, relative)
                    row[name] = dict(relative_l2=relative, max_abs=float(error.abs().max()))
                metrics.append(row)
            cases.append(dict(active_rows=active, median_ms=1e3 * median(run.seconds for run in runs[1:]),
                              lanes=metrics))
    print(json.dumps(dict(passed=True, block=block, tp=tp, world=world, layer=layer,
                          output_byte_exact=True, earlier_slot_untouched=True,
                          capture_median_ms={slot: median(times[1:]) for slot, times in timing.items()},
                          cases=cases)))


if __name__ == "__main__":
    main()
