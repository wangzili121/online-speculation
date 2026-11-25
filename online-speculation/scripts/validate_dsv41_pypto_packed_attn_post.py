#!/usr/bin/env python3
"""Check the captured draft Attention post backward and FP32 TP broadcast."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--fixture-tp", type=int, choices=(2, 4), default=2)
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    block, tp, world = args.block, args.fixture_tp, len(devices)
    if len(set(devices)) != world or world % tp or args.checkpoint is None and not args.compile_only:
        parser.error("Use distinct devices in complete TP groups and pass a W8A8 checkpoint for execution")

    import torch
    from blockspec.dsv41_packed_attention_tape import packed_attention_tape_shapes
    from blockspec.dsv41_packed_attn_post_vjp import PackedAttentionPostVJP, make_packed_attn_post_vjp
    from blockspec.dsv41_packed_ffn_mhc_vjp import PackedFFNAttentionGradient
    from models.deepseek_v4_1_flash_a3.config import FLASH as model
    from models.deepseek_v4_1_flash_a3.hc import golden_mhc_mixes
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    dp, last_n = world // tp, 4
    hidden, copies, channels = model.hidden_size, model.hc_mult, model.mix_hc
    ranks, owners = tuple(range(world)), tuple(range(0, world, tp))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_attn_post_vjp(block, last_n, model, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, tp=tp, dp=dp, ep=world, last_n=last_n)))
        return

    checkpoint = Checkpoint(args.checkpoint)
    layers = (36, 39)
    names = ("hc_attn_fn", "hc_attn_scale", "hc_attn_base")
    weights = {key: torch.stack([checkpoint.tensor(f"layers.{idx}.{key}").float() for idx in layers]) for key in names}
    generator = torch.Generator().manual_seed(613909)
    shape = packed_attention_tape_shapes(block, last_n, hidden, copies)
    tape = {key: torch.empty((dp, *dimensions), dtype=getattr(torch, dtype))
            for key, (dimensions, dtype) in shape.items()}
    tape["attention_tape_residual"].normal_(generator=generator)
    tape["attention_tape_delta"].normal_(generator=generator)
    tape["attention_tape_mixes"].zero_()
    first = -2  # Two fixture bank layers select slots 2 and 3 of the real last-4 ABI.
    for replica in range(dp):
        for layer in range(len(layers)):
            slot = layer - first
            x = tape["attention_tape_residual"][replica].reshape(last_n, block, copies, hidden)[slot]
            _, post, combo = golden_mhc_mixes(x, *(weight[layer] for weight in weights.values()))
            ds = tape["attention_tape_mixes"][replica, slot * block:(slot + 1) * block]
            ds[:, copies:2 * copies] = post
            ds[:, 2 * copies:channels] = combo.flatten(1)
    streams = torch.randn(dp, block * copies, hidden, generator=generator) * .05
    premix = torch.randn(dp, block, 16, generator=generator) * .1
    cases = []

    with DistributedWorker([graph], config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(value, rank):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def read(value, rank):
            result = torch.empty(value.shape, dtype=value.dtype)
            runtime.copy_from(result.data_ptr(), value.data_ptr, result.nbytes, worker_id=rank)
            return result

        def stack(values):
            return StackedDeviceTensor([upload(value, rank) for rank, value in zip(owners, values, strict=True)],
                                       values.shape, owners)

        rank_tapes = [{} for _ in ranks]
        banks = [{} for _ in ranks]
        for replica, owner in enumerate(owners):
            rank_tapes[owner] = {key: upload(value[replica], owner) for key, value in tape.items()}
            banks[owner] = {key: upload(value, owner) for key, value in weights.items()}
        actor_model = SimpleNamespace(num_hidden_layers=last_n + first, hc_mult=copies, hidden_size=hidden)
        proposer = SimpleNamespace(world=world, block=block, last_layers=last_n, runtime=runtime,
                                   config=config, ranks=rank_tapes)
        ffn_mhc = SimpleNamespace(owners=owners, outputs=(stack(streams), stack(premix)))
        reverse = PackedAttentionPostVJP(proposer, ffn_mhc, graph, banks, actor_model)
        for owner, active, layer in ((owners[0], block, 1), (owners[-1], 7, 0), (owners[0], 1, 1)):
            replica, slot = owner // tp, layer - first
            student = SimpleNamespace(ffn_tape={owner: rank_tapes[owner]})
            seed = PackedFFNAttentionGradient(owner, layer, active, ffn_mhc.outputs[0].shards[replica],
                                              ffn_mhc.outputs[1].shards[replica], 0.)
            samples = [reverse(student, seed) for _ in range(7)]
            result = samples[-1]
            dx = read(result.residual, owner).reshape(block, copies, hidden)
            delta = read(result.delta.shards[owner], owner)
            for lane in range(1, tp):
                torch.testing.assert_close(read(result.delta.shards[owner + lane], owner + lane), delta, atol=0, rtol=0)

            x = tape["attention_tape_residual"][replica].reshape(last_n, block, copies, hidden)[slot, :active]
            x = x.double().requires_grad_(True)
            d = tape["attention_tape_delta"][replica, slot * block:slot * block + active]
            d = d.double().requires_grad_(True)
            fn, scale, bias = (weight[layer].double() for weight in weights.values())
            flat = x.flatten(1)
            p = torch.nn.functional.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True)
                                                                 + model.rms_norm_eps)
            p1, p2, p3 = p.split((copies, copies, copies * copies), dim=-1)
            b1, b2, b3 = bias.split((copies, copies, copies * copies))
            pre = torch.sigmoid(p1 * scale[0] + b1) + model.hc_eps
            post = 2 * torch.sigmoid(p2 * scale[1] + b2)
            combo = (p3 * scale[2] + b3).reshape(-1, copies, copies).softmax(-1) + model.hc_eps
            combo = combo / (combo.sum(-2, keepdim=True) + model.hc_eps)
            for _ in range(model.hc_sinkhorn_iters - 1):
                combo = combo / (combo.sum(-1, keepdim=True) + model.hc_eps)
                combo = combo / (combo.sum(-2, keepdim=True) + model.hc_eps)
            y = d[:, None] * post[:, :, None] + (combo.transpose(1, 2) @ x)
            expected = torch.autograd.grad((y, pre), (x, d),
                                            (streams[replica].reshape(block, copies, hidden)[:active].double(),
                                             premix[replica, :active, :copies].double()))
            metrics = {}
            for name, actual, ref in (("residual", dx, expected[0]), ("projection", delta, expected[1])):
                error = actual[:active].double() - ref
                relative = float(error.norm() / ref.norm())
                assert relative < 1e-5 and not torch.count_nonzero(actual[active:]), (name, owner, active, relative)
                metrics[name] = dict(relative_l2=relative, max_abs=float(error.abs().max()))
            cases.append(dict(owner=owner, active_rows=active, checkpoint_layer=layers[layer], tape_slot=slot,
                              metrics=metrics, tp_byte_exact=True,
                              warm_median_ms=1000 * median(step.seconds for step in samples[1:])))
    print(json.dumps(dict(passed=True, block=block, tp=tp, dp=dp, last_n=last_n, checkpoint=str(args.checkpoint),
                          cases=cases)), flush=True)


if __name__ == "__main__":
    main()
