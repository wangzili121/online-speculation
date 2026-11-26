#!/usr/bin/env python3
"""Validate the device gradient handoff between two packed draft layers."""

import argparse
import json
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--checkpoint")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only and not args.checkpoint:
        parser.error("--checkpoint is required on device")

    import torch
    from blockspec.dsv41_packed_attention_tape import packed_attention_tape_shapes
    from blockspec.dsv41_packed_attn_input_vjp import PackedAttentionInputVJP, make_packed_attn_input_vjp
    from blockspec.dsv41_packed_ffn_tape import PackedFFNPostVJP, make_packed_ffn_post_vjp, packed_ffn_tape_shapes
    from blockspec.dsv41_packed_final_vjp import make_packed_final_vjp
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE as tp
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    devices = tuple(map(int, args.devices.split(",")))
    world, block, n = len(devices), args.block, 2
    dp, teachers = world // tp, MAX_LOGIT_ROWS
    owners = tuple(range(0, world, tp))
    hidden, copies = model.hidden_size, model.hc_mult
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    terminal = make_packed_final_vjp(teachers, hidden, copies, dp, tp, model.rms_norm_eps,
                                     output_rows=block).compile(config=config)
    carry = make_packed_attn_input_vjp(block, n, model, dp, tp).compile(config=config)
    post = make_packed_ffn_post_vjp(block, n, block, hidden, copies, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, world=world)))
        return

    checkpoint = Checkpoint(args.checkpoint)
    layer = model.num_hidden_layers - 1
    gamma = checkpoint.tensor(f"layers.{layer}.attn_norm.weight").contiguous()
    gamma_bank = torch.zeros((layer + 1, hidden), dtype=torch.bfloat16)
    gamma_bank[layer].copy_(gamma)
    attention_shapes = packed_attention_tape_shapes(block, n, hidden, copies)
    ffn_shapes = packed_ffn_tape_shapes(block, n, tp, hidden, copies)
    rng = torch.Generator().manual_seed(126846)
    pad = torch.nn.functional.pad

    def random(*shape):
        return torch.randn(*shape, generator=rng) * .1

    def loss_at(x, coeff, seed):
        pieces = [x[:, i] * coeff[:, i:i + 1] for i in range(copies)]
        collapsed = (pieces[0] + pieces[1]) + (pieces[2] + pieces[3])
        with torch.no_grad():
            physical = [x[:, i].float() * coeff[:, i:i + 1].float() for i in range(copies)]
            rounded = ((physical[0] + physical[1]) + (physical[2] + physical[3])).bfloat16().double()
        collapsed = rounded + (collapsed - collapsed.detach())
        normed = collapsed * (collapsed.square().mean(-1, keepdim=True) + model.rms_norm_eps).rsqrt()
        return (normed * gamma.double() * seed.double()).sum()

    fixtures = []
    for _ in owners:
        previous = random(block, copies, hidden)
        combined = random(block, hidden).bfloat16()
        multiplier = random(block, copies)
        mixture = random(block, copies, copies)
        coefficient = random(block, copies)
        streams = torch.einsum("bst,bsh->bth", mixture, previous) + combined.float()[:, None] * multiplier[:, :, None]
        d_norm = random(block, hidden)
        d_skip = random(block, copies, hidden)
        d_skip[7:].zero_()
        tape = {name: torch.zeros(shape, dtype=getattr(torch, dtype))
                for name, (shape, dtype) in {**attention_shapes, **ffn_shapes}.items()}
        tape["attention_tape_residual"][-block * copies:].copy_(streams.flatten(0, 1))
        tape["attention_tape_mixes"][-block:, :copies].copy_(coefficient)
        tape["ffn_tape_residual"][:block * copies].copy_(previous.flatten(0, 1))
        tape["ffn_tape_mixes"][:block, copies:2 * copies].copy_(multiplier)
        tape["ffn_tape_mixes"][:block, 2 * copies:2 * copies + copies * copies].copy_(mixture.flatten(1))
        tape["ffn_tape_vectors"][block:2 * block].copy_(combined)
        fixtures.append((tape, streams, coefficient, d_norm, d_skip, previous, combined, multiplier, mixture))

    cases = []
    with DistributedWorker([terminal, carry, post], config, persistent=True) as runtime:
        def upload(values):
            shards = [runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                      for rank, value in zip(owners, values, strict=True)]
            return StackedDeviceTensor(shards, (len(shards), *shards[0].shape), owners)

        def field(index, transform=lambda x: x):
            return upload([transform(fixture[index]) for fixture in fixtures])

        def read(value, owner):
            result = torch.empty(value.shape, dtype=value.dtype)
            runtime.copy_from(result.data_ptr(), value.data_ptr, result.nbytes, worker_id=owner)
            return result

        banks = [{} for _ in devices]
        for index, owner in enumerate(owners):
            banks[owner] = {
                key: runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=owner)
                for key, value in dict(fixtures[index][0], attn_norm_weight=gamma_bank).items()
            }
        proposer = SimpleNamespace(runtime=runtime, config=config, block=block, last_layers=n, ranks=banks)
        normed, direct = field(3), field(4, lambda value: value.flatten(0, 1))
        qkv = SimpleNamespace(owners=owners, outputs=(normed,))
        attn_input = PackedAttentionInputVJP(proposer, SimpleNamespace(residual=direct), qkv, carry, model)
        terminal_grad = upload([torch.full((block, copies, hidden), float("nan"))] * dp)
        terminal_premix = upload([torch.full((block, 16), float("nan"))] * dp)
        final = SimpleNamespace(owners=owners, grad_streams=terminal_grad, grad_premix=terminal_premix)
        ff_post = PackedFFNPostVJP(proposer, final, post, model.num_hidden_layers)
        terminal_inputs = (field(1, lambda value: value[:teachers].contiguous()),
                           field(2, lambda value: value[:teachers].contiguous()), upload([gamma] * dp),
                           field(3, lambda value: pad(value[:7], (0, 0, 0, teachers - 7))))
        for index, owner in enumerate(owners):
            _, x, coeff, grad, direct_grad, prior_x, prior_value, prior_post, prior_mix = fixtures[index]
            runtime.run(terminal, *terminal_inputs, terminal_grad, terminal_premix, owner, config=config)
            tx, tc = (value[:teachers].double().requires_grad_() for value in (x, coeff))
            teacher = pad(grad[:7], (0, 0, 0, teachers - 7))
            tx_grad, tc_grad = torch.autograd.grad(loss_at(tx, tc, teacher), (tx, tc))
            terminal_actual = [read(stack.shards[index], owner) for stack in (terminal_grad, terminal_premix)]
            terminal_expected = (pad(tx_grad, (0, 0, 0, 0, 0, block - teachers)),
                                 pad(tc_grad, (0, 16 - copies, 0, block - teachers)))
            for actual, reference in zip(terminal_actual, terminal_expected, strict=True):
                torch.testing.assert_close(actual.double(), reference, atol=3e-6, rtol=2e-5)
                assert torch.count_nonzero(actual[7:]) == 0
            student = SimpleNamespace(owner=owner, ffn_tape={owner: banks[owner]})
            seed = SimpleNamespace(owner=owner, layer=layer, rows=block,
                                   normalized=normed.shards[index], residual=direct.shards[index])
            stage_times, first = [], None
            for _ in range(8):
                boundary = attn_input(student, seed)
                earlier = ff_post(student, boundary, layer=layer - 1)
                stage_times.append((boundary.seconds * 1000, earlier.seconds * 1000))
                actual = [read(value, owner) for value in
                          (boundary.streams, boundary.premix, earlier.value, earlier.residual, earlier.mixes)]
                if first is None:
                    first = actual
                else:
                    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                               for a, b in zip(first, actual, strict=True))
            x, coeff = (value.double().requires_grad_() for value in (x, coeff))
            dx, dc = torch.autograd.grad(loss_at(x, coeff, grad) + (x * direct_grad.double()).sum(), (x, coeff))
            previous, combined, multiplier, mixture = (value.double().requires_grad_()
                                                       for value in (prior_x, prior_value, prior_post, prior_mix))
            earlier_x = torch.einsum("bst,bsh->bth", mixture, previous) + combined[:, None] * multiplier[:, :, None]
            dv, dr, dm, db = torch.autograd.grad((earlier_x * dx).sum(), (combined, previous, multiplier, mixture))
            mix_gradient = pad(torch.cat((dc, dm, db.flatten(1)), 1), (0, 8))
            expected = (dx, pad(dc, (0, 16 - copies)), dv, dr.flatten(0, 1), mix_gradient)
            errors = [float((a.double() - b).norm() / b.norm()) for a, b in zip(first, expected, strict=True)]
            assert max(errors) < 2e-5, errors
            assert all(torch.count_nonzero(value) > 0 for value in (first[0][7:], first[2][7:]))
            cases.append(dict(owner=owner, repeated_byte_exact=len(stage_times), live_rows=block,
                              terminal_padding_exact=True, max_relative_l2=max(errors),
                              warm_median_ms=dict(attention_input=median(time[0] for time in stage_times[1:]),
                                                  previous_ffn_post=median(time[1] for time in stage_times[1:]))))
    print(json.dumps(dict(passed=True, block=block, world=world, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
