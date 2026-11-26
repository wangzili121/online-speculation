#!/usr/bin/env python3
"""Compare the packed Q/KV weight and input gradients with FP64 autograd."""

import argparse
from dataclasses import asdict
import json
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import torch

from validate_dsv41_pypto_packed_qkv import checkpoint_operands


def make_forward(block, world, model, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from blockspec.dsv41_packed_attention_tape import make_packed_qkv_capture
    from blockspec.pypto_dsv41_draft import make_packed_draft_qkv

    hidden, latent, dim = model.hidden_size, model.q_lora_rank, model.head_dim
    width, joint = model.num_attention_heads // tp * dim, latent + dim
    project = make_packed_draft_qkv(asdict(model), tp=tp, capacity=block, expose_joint=True)
    retain = make_packed_qkv_capture(block, 1, model)

    @pl.jit
    def forward(
        x: pl.Tensor[[block, hidden], pl.BF16], a: pl.Tensor[[hidden, joint], pl.BF16],
        q_scale: pl.Tensor[[1, latent], pl.BF16], b: pl.Tensor[[latent, width], pl.BF16],
        kv_scale: pl.Tensor[[1, dim], pl.BF16], cosine: pl.Tensor[[block, 32], pl.FP32],
        sine: pl.Tensor[[block, 32], pl.FP32],
        low_saved: pl.Out[pl.Tensor[[block, latent], pl.BF16]],
        raw_saved: pl.Out[pl.Tensor[[block, joint], pl.BF16]],
    ):
        low = pl.create_tensor([block, latent], dtype=pl.BF16)
        q = pl.create_tensor([block, width], dtype=pl.BF16)
        kv = pl.create_tensor([block, dim], dtype=pl.BF16)
        raw = pl.create_tensor([block, joint], dtype=pl.BF16)
        project(x, a, pl.reshape(q_scale, [latent]), b, pl.reshape(kv_scale, [dim]),
                cosine, sine, low, q, kv, raw, block)
        retain(low, raw, low_saved, raw_saved, True, 0)
        return low_saved, raw_saved

    @pl.jit.host
    def dispatch(
        x: pl.Tensor[[world, block, hidden], pl.BF16], a: pl.Tensor[[world, hidden, joint], pl.BF16],
        q_scale: pl.Tensor[[world, 1, latent], pl.BF16], b: pl.Tensor[[world, latent, width], pl.BF16],
        kv_scale: pl.Tensor[[world, 1, dim], pl.BF16], cosine: pl.Tensor[[world, block, 32], pl.FP32],
        sine: pl.Tensor[[world, block, 32], pl.FP32],
        low: pl.Out[pl.Tensor[[world, block, latent], pl.BF16]],
        raw: pl.Out[pl.Tensor[[world, block, joint], pl.BF16]],
    ):
        for rank in pl.range(pld.world_size()):
            forward(x[rank], a[rank], q_scale[rank], b[rank], kv_scale[rank], cosine[rank], sine[rank],
                    low[rank], raw[rank], device=rank)
    return dispatch


def reference(x, joined, q_a_kv, q_b, gamma, cosine, sine, low, dq, dkv, eps, active):
    def differentiate(value):
        return value.double().requires_grad_(True)

    def use_forward_value(value, captured):
        return captured.double() + (value - value.detach())

    def rms(value, scale):
        inverse = (value.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
        return value * inverse * scale

    def rope(values):
        tail = cosine.shape[-1] * 2
        even, odd = values[..., -tail:].unflatten(-1, (-1, 2)).unbind(-1)
        c, s = cosine.double().unsqueeze(1), sine.double().unsqueeze(1)
        rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
        return torch.cat((values[..., :-tail], rotated), dim=-1)

    x, a = differentiate(x), differentiate(q_a_kv)
    q_scale, kv_scale = map(differentiate, gamma)
    b = list(map(differentiate, q_b))
    q_raw, kv_raw = use_forward_value(x @ a, joined).split((q_scale.numel(), kv_scale.numel()), dim=-1)
    q_normed = rms(q_raw, q_scale)
    kv = rope(rms(kv_raw, kv_scale).unsqueeze(1)).squeeze(1)
    total = 0.
    for lane, shard in enumerate(b):
        query = use_forward_value(q_normed, low[lane]) @ shard
        query = rope(query.reshape_as(dq[lane]))
        total += (query[:active] * dq[lane][:active].double()).sum() + (kv * dkv[lane].double()).sum()
    return torch.autograd.grad(total, (x, a, q_scale, kv_scale, *b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=16)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--checkpoint")
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only and args.checkpoint is None:
        parser.error("--checkpoint is required on device")

    from blockspec.dsv41_packed_qkv_vjp import PackedQKVVJP, make_packed_qkv_vjp
    from blockspec.pypto_dsv41_draft import pack_draft_input_projection
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE as tp
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    devices, block = tuple(map(int, args.devices.split(","))), args.block
    world, heads, dim = len(devices), model.num_attention_heads // tp, model.head_dim
    latent = model.q_lora_rank
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_qkv_vjp(block, model, world=world, tp=tp).compile(config=config)
    capture = make_forward(block, world, model, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, world=world)))
        return
    rng = torch.Generator().manual_seed(774145)
    x, first_lane = checkpoint_operands(args.checkpoint, args.layer, 0)
    wb = [first_lane["wq_b"]] + [checkpoint_operands(args.checkpoint, args.layer, lane)[1]["wq_b"]
                                for lane in range(1, tp)]
    joined_weights = pack_draft_input_projection(first_lane)
    x = x[:block]
    gamma = tuple(first_lane[key].reshape(1, -1) for key in ("q_norm", "kv_norm"))
    angle = torch.randn(block, 32, generator=rng)
    cosine, sine = angle.cos(), angle.sin()
    dq = [torch.randn(block, heads, dim, generator=rng) * .035 for _ in devices]
    dkv = [torch.randn(block, dim, generator=rng) * .07 for _ in devices]
    cases = []
    ranks, owners = tuple(range(world)), tuple(range(0, world, tp))
    with DistributedWorker([capture, graph], config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(host, assigned=ranks):
            shards = [runtime.alloc_tensor(t.shape, t.dtype, init=t, worker_id=rank)
                      for rank, t in zip(assigned, host, strict=True)]
            return StackedDeviceTensor(shards, (len(shards), *host[0].shape), assigned)

        def repeat(value, assigned=ranks):
            return upload([value] * len(assigned), assigned)

        def read(stack, index):
            shard = stack.shards[index]
            result = torch.empty(shard.shape, dtype=shard.dtype)
            runtime.copy_from(result.data_ptr(), shard.data_ptr, result.nbytes, worker_id=stack.worker_ids[index])
            return result

        activations, a = repeat(x), repeat(joined_weights)
        b = upload([wb[rank % tp] for rank in ranks])
        q_gamma, kv_gamma = map(repeat, gamma)
        cos, sin = repeat(cosine), repeat(sine)
        low = repeat(torch.empty(block, latent, dtype=torch.bfloat16))
        raw = repeat(torch.empty(block, latent + dim, dtype=torch.bfloat16))
        runtime.run(capture, activations, a, q_gamma, b, kv_gamma, cos, sin, low, raw, config=config)
        low_host = [read(low, rank) for rank in ranks]
        raw_host = [read(raw, rank) for rank in ranks]
        assert all(torch.equal(raw_host[0], value) for value in raw_host)
        assert all(torch.equal(low_host[0], value) for value in low_host)
        query_seeds, kv_seeds = upload([x.flatten(0, 1) for x in dq]), upload(dkv)
        banks = {
            "draft_q_a_kv": a, "draft_q_b": b, "draft_q_gamma": q_gamma, "draft_kv_gamma": kv_gamma,
            "freqs_cos": repeat(torch.stack((cosine, cosine))), "freqs_sin": repeat(torch.stack((sine, sine))),
            "attention_tape_normed": activations, "attention_tape_q_normed": low, "attention_tape_qkv_raw": raw,
        }
        by_rank = [{name: shards.shards[rank] for name, shards in banks.items()} for rank in ranks]
        proposer = SimpleNamespace(runtime=runtime, config=config, ranks=by_rank, block=block,
                                   tp=tp, world=world, last_layers=1)
        vjp = PackedQKVVJP(proposer, graph, SimpleNamespace(**{**asdict(model), "num_hidden_layers": 1}))
        dx, dw_a, dw_b, dw_q, dw_kv = vjp.outputs
        for active, owner in ((7, 0), (1, owners[-1]), (block, 0)):
            group = range(owner, owner + tp)
            replica = owner // tp
            expected = reference(x, raw_host[owner], joined_weights, wb, gamma, cosine, sine,
                                 low_host[owner:owner + tp], dq[owner:owner + tp], dkv[owner:owner + tp],
                                 model.rms_norm_eps, active)
            times = []
            first = None
            student = SimpleNamespace(owner=owner, rows=active, position=0,
                                      ffn_tape={rank: by_rank[rank] for rank in group})
            seed = SimpleNamespace(owner=owner, rows=active, layer=0, query=query_seeds,
                                   temporary=kv_seeds, residual=None, parameter_gradients={})
            for _ in range(4):
                start = perf_counter()
                result = vjp(student, seed)
                times.append((perf_counter() - start) * 1e3)
                assert result.rows == block and result.normalized is dx.shards[replica]
                actual = [read(stack, replica) for stack in (dx, dw_a, dw_q, dw_kv)]
                actual.extend(read(dw_b, lane) for lane in group)
                if first is None:
                    first = actual
                else:
                    assert all(torch.equal(p.view(torch.uint8), q.view(torch.uint8))
                               for p, q in zip(first, actual, strict=True)), (block, active, owner)
            names = ("input", "q_a_kv", "q_norm", "kv_norm", *(f"q_b_lane_{i}" for i in range(tp)))
            errors = {}
            for name, actual, target in zip(names, first, expected, strict=True):
                diff = actual.double() - target
                relative = float(diff.norm() / target.norm().clamp_min(1e-30))
                assert relative < 3e-5, (block, active, owner, name, relative, diff.abs().max().item())
                errors[name] = dict(relative_l2=relative, max_abs=diff.abs().max().item())
            if active < block:
                assert first[0][active:].count_nonzero() > 0
            cases.append(dict(active_q=active, owner=owner, repeated_byte_exact=len(times), first_ms=times[0],
                              warm_median_ms=median(times[1:]), errors=errors))
    print(json.dumps(dict(passed=True, layer=args.layer, block=block, world=world, cases=cases)))


if __name__ == "__main__":
    main()
