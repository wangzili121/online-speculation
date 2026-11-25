#!/usr/bin/env python3
"""Validate the packed FFN mHC VJP against the original checkpoint and FP64 autograd."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--fixture-tp", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--block", type=int, default=32, choices=(8, 16, 32))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    block, tp, world = args.block, args.fixture_tp, len(devices)
    if len(set(devices)) != world or world % tp or args.checkpoint is None and not args.compile_only:
        parser.error("Use distinct devices in complete TP groups and supply a checkpoint for execution")

    import torch
    from blockspec.dsv41_packed_ffn_input_vjp import PackedFFNInputGradient
    from blockspec.dsv41_packed_ffn_mhc_vjp import PackedFFNMHCVJP, make_packed_ffn_mhc_vjp
    from blockspec.dsv41_packed_ffn_tape import PackedFFNPostGradient, packed_ffn_tape_shapes
    from models.deepseek_v4_1_flash_a3.config import FLASH as model
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    dp, last_layers = world // tp, 4
    hidden, copies, channels = model.hidden_size, model.hc_mult, model.mix_hc
    ranks, owners = tuple(range(world)), tuple(range(0, world, tp))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_ffn_mhc_vjp(block, last_layers, model, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, tp=tp, dp=dp,
                              last_layers=last_layers)))
        return

    checkpoint = Checkpoint(args.checkpoint)
    weights = {name: torch.stack([checkpoint.tensor(f"layers.{layer}.{name}").float() for layer in (36, 39)])
               for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")}
    generator = torch.Generator().manual_seed(659412)
    tensor_specs = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    tape_residual = torch.randn(dp, *tensor_specs["ffn_tape_residual"][0], generator=generator) * .2
    tape_mix = torch.randn(dp, *tensor_specs["ffn_tape_mixes"][0], generator=generator) * .1
    post_dx = torch.randn(dp, block * copies, hidden, generator=generator) * .05
    post_dmix = torch.randn(dp, block, 32, generator=generator) * .1
    collapsed_dx = torch.randn(world, block, hidden, generator=generator) * .05
    first_layer = -2
    fixtures = ((owners[0], 1, 1, "all"), (owners[-1], 7, 0, "gate"),
                (owners[0], block, 1, "all"))
    cases = []

    with DistributedWorker([graph], config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(value, rank):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def stacked(values, worker_ids):
            return StackedDeviceTensor([upload(value, rank) for value, rank in zip(values, worker_ids, strict=True)],
                                       values.shape, worker_ids)

        def read(allocation, rank):
            host = torch.empty(allocation.shape, dtype=allocation.dtype)
            runtime.copy_from(host.data_ptr(), allocation.data_ptr, host.nbytes, worker_id=rank)
            return host

        tapes = [{} for _ in ranks]
        banks = [{} for _ in ranks]
        for replica, owner in enumerate(owners):
            tapes[owner] = {
                "ffn_tape_residual": upload(tape_residual[replica], owner),
                "ffn_tape_mixes": upload(tape_mix[replica], owner),
            }
            banks[owner] = {name: upload(value, owner) for name, value in weights.items()}
        post = SimpleNamespace(owners=owners, outputs=None)
        ffn = SimpleNamespace(output=None)
        seeded_mix = stacked(post_dmix, owners)
        source_dx = {
            "all": (post_dx, collapsed_dx),
            "gate": (torch.zeros_like(post_dx), torch.zeros_like(collapsed_dx)),
        }
        device_dx = {mode: (stacked(direct, owners), stacked(collapsed, ranks))
                     for mode, (direct, collapsed) in source_dx.items()}
        proposer = SimpleNamespace(runtime=runtime, config=config, ranks=tapes, last_layers=last_layers, block=block)
        actor_model = SimpleNamespace(num_hidden_layers=first_layer + last_layers, hidden_size=hidden, hc_mult=copies)
        mhc = PackedFFNMHCVJP(proposer, post, ffn, graph, banks, actor_model)

        for owner, active, layer, mode in fixtures:
            residual_dx, norm_dx = source_dx[mode]
            direct, ffn.output = device_dx[mode]
            post.outputs = (None, direct, seeded_mix)
            replica, slot = owner // tp, layer - first_layer
            student = SimpleNamespace(ffn_tape={owner: tapes[owner]})
            before = PackedFFNPostGradient(owner, layer, active, None, post.outputs[1].shards[replica],
                                           post.outputs[2].shards[replica], 0.)
            after = PackedFFNInputGradient(owner, layer, active, ffn.output.shards[owner], 0.)
            samples = [mhc(student, before, after) for _ in range(8)]
            streams = read(samples[-1].streams, owner).reshape(block, copies, hidden)
            premix = read(samples[-1].premix, owner)

            x = tape_residual[replica].reshape(last_layers, block, copies, hidden)[slot, :active]
            x = x.double().requires_grad_(True)
            a = tape_mix[replica, slot * block:slot * block + active, :copies].double().requires_grad_(True)
            flat = x.flatten(1)
            function, scale, bias = (matrix[layer].double() for matrix in weights.values())
            inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + model.rms_norm_eps)
            packed = (flat @ function.T) * inv
            pre, post_gate, residual = packed.split((copies, copies, copies * copies), dim=-1)
            b1, b2, b3 = bias.split((copies, copies, copies * copies))
            pre = torch.sigmoid(pre * scale[0] + b1) + model.hc_eps
            post_gate = 2 * torch.sigmoid(post_gate * scale[1] + b2)
            residual = (residual * scale[2] + b3).reshape(-1, copies, copies).softmax(-1) + model.hc_eps
            residual = residual / (residual.sum(-2, keepdim=True) + model.hc_eps)
            for _ in range(model.hc_sinkhorn_iters - 1):
                residual = residual / (residual.sum(-1, keepdim=True) + model.hc_eps)
                residual = residual / (residual.sum(-2, keepdim=True) + model.hc_eps)
            collapsed = (x * a.unsqueeze(-1)).sum(1)
            incoming = post_dmix[replica, :active, :channels].double().split((copies, copies, copies * copies), -1)
            seed = (incoming[0], incoming[1], incoming[2].reshape(-1, copies, copies),
                    norm_dx[owner, :active].double(), residual_dx[replica].reshape(block, copies, hidden)[:active].double())
            expected = torch.autograd.grad((pre, post_gate, residual, collapsed, x), (x, a), seed)
            result_metrics = {}
            for name, actual, ref in (("stream", streams[:active], expected[0]),
                                       ("premix", premix[:active, :copies], expected[1])):
                delta = actual.double() - ref
                denom = float(ref.norm())
                rel = float(delta.norm()) / denom if denom else 0.
                assert (rel < .003 if denom else not torch.count_nonzero(delta)), (owner, active, layer, name, rel)
                result_metrics[name] = dict(relative_l2=rel, max_abs=float(delta.abs().max()))
            assert not torch.count_nonzero(streams[active:]) and not torch.count_nonzero(premix[active:])
            assert not torch.count_nonzero(premix[:, copies:])
            cases.append(dict(owner=owner, active_rows=active, checkpoint_layer=(36, 39)[layer],
                              tape_slot=slot, mode=mode, metrics=result_metrics,
                              warm_median_ms=1000 * median(value.seconds for value in samples[1:])))
    print(json.dumps(dict(passed=True, block=block, tp=tp, dp=dp, checkpoint=str(args.checkpoint),
                          last_layers=last_layers, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
