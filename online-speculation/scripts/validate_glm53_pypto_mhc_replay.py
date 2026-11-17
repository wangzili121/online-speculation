#!/usr/bin/env python3
"""Check a GLM residual sublayer replay and the complete stream adjoint."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.pypto_glm53_online_mhc import make_mhc_sublayer_adjoint


def run(args, report):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH, HC_MULT
    from models.glm5_3_flash.mhc import hc_mixes, hc_pre
    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    tokens, rows, pad = 6, 16, 8
    streams, width = HC_MULT, HC_MULT * D
    ranks = len(args.devices)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
        device_ids=args.devices, num_sub_workers=0,
    ))
    start, finish = make_mhc_sublayer_adjoint(ranks, tokens)
    report["compiled_stages"] = []
    programs = []
    for entry in (start, finish):
        programs.append(entry.compile(config=config))
        report["compiled_stages"].append(entry._func.__name__)
    if args.compile_only:
        return

    @pl.jit
    def scale_rows(
        inputs: pl.Tensor[[rows, D], pl.FP32], output: pl.Out[pl.Tensor[[rows, D], pl.FP32]],
    ):
        tile = 512
        for task in pl.spmd(rows * (D // tile), name_hint="glm_mhc_fixture_scaled_adjoint"):
            row = task // (D // tile)
            col = task % (D // tile) * tile
            output[row:row + 1, col:col + tile] = pl.mul(inputs[row:row + 1, col:col + tile], 0.25)
        return output

    @pl.jit.host
    def fake_sublayer(
        inputs: pl.Tensor[[ranks, rows, D], pl.FP32],
        output: pl.Out[pl.Tensor[[ranks, rows, D], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            scale_rows(inputs[rank], output[rank], device=rank)

    compute = fake_sublayer.compile(config=config)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(43)
    cp = Checkpoint(args.checkpoint)

    def replicate(tensor):
        return tensor.unsqueeze(0).repeat((ranks,) + (1,) * tensor.ndim).share_memory_()

    def buffer(*shape, dtype=torch.float32):
        return torch.empty((ranks, *shape), dtype=dtype).share_memory_()

    try:
        keys = [cp.layer_key(args.layer, f"ffn_hc.{suffix}") for suffix in ("fn", "scale", "base")]
        function, scale, base = (cp.read(key) for key in keys)
    except KeyError:
        keys = [cp.layer_key(args.layer, f"hc_ffn_{suffix}") for suffix in ("fn", "scale", "base")]
        function, scale, base = (cp.read(key) for key in keys)
    function = function.reshape(-1, width).bfloat16()
    scale, base = scale.reshape(3).float(), base.flatten().float()
    x = (torch.randn((tokens, streams, D), generator=generator) * 0.6).bfloat16()
    p, q, r = hc_mixes(x, function, scale, base)
    collapse_ref = hc_pre(x, p)
    y = (collapse_ref.float() * 0.25).bfloat16()
    upstream = torch.zeros((rows, streams, D), dtype=torch.float32)
    upstream[:tokens] = torch.randn((tokens, streams, D), generator=generator) * 0.02
    tape = replicate(torch.cat([x, y[:, None]], dim=1))
    weights = tuple(replicate(t) for t in (function, scale, base))

    residual = buffer(rows * streams, D, dtype=torch.bfloat16)
    collapse = buffer(rows, D, dtype=torch.bfloat16)
    pre, post, dy, dpost = buffer(rows, pad), buffer(rows, pad), buffer(rows, D), buffer(rows, pad)
    logits, dmix = buffer(rows * pad, pad), buffer(rows * streams, pad)
    direct, output = buffer(rows * streams, D), buffer(rows * streams, D)
    dz = buffer(rows, D)
    start_args = (tape, *weights, replicate(upstream.flatten(0, 1)), residual, collapse, pre, post, logits,
                  dy, direct, dpost, dmix, buffer(rows, D, dtype=torch.bfloat16), buffer(rows * streams, pad))
    finish_args = (residual, *weights[:2], pre, post, logits, direct, dpost, dmix, dz, output,
                   buffer(rows * streams, D), buffer(rows, pad), buffer(rows * pad, pad),
                   buffer(rows * pad, pad), buffer(rows, width))
    with DistributedWorker([*programs, compute], config=config, persistent=True) as worker:
        buffers = {}
        for tensor in (*start_args, *finish_args):
            if id(tensor) not in buffers:
                buffers[id(tensor)] = worker.alloc_stacked_tensor(tensor)
        for program, arguments in ((programs[0], start_args), (compute, (dy, dz)), (programs[1], finish_args)):
            worker.run(program, *(buffers[id(t)] for t in arguments), config=config)
        for target in (collapse, dy, output):
            for rank, shard in enumerate(buffers[id(target)].shards):
                worker.copy_from(target[rank].data_ptr(), shard.data_ptr, target[rank].nbytes, worker_id=rank)

    reference_x = x.double().requires_grad_()
    flat = reference_x.flatten(1)
    projection = (flat @ function.double().T) * (flat.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt()
    pre_z, post_z, mix_z = projection.split((streams, streams, streams * streams), dim=-1)
    p_smooth = (pre_z * scale[0] + base[:streams]).sigmoid() + FLASH.hc_eps
    q_smooth = (post_z * scale[1] + base[streams:2 * streams]).sigmoid() * 2
    z = (mix_z * scale[2] + base[2 * streams:]).unflatten(1, (streams, streams))
    r_smooth = z.softmax(-1) + FLASH.hc_eps
    for step in range(2 * FLASH.hc_sinkhorn_iters - 1):
        r_smooth = r_smooth / (r_smooth.sum(-2 if step % 2 == 0 else -1, keepdim=True) + FLASH.hc_eps)
    p_ste = p_smooth + (p.double() - p_smooth).detach()
    q_ste = q_smooth + (q.bfloat16().double() - q_smooth).detach()
    r_ste = r_smooth + (r.bfloat16().double() - r_smooth).detach()
    smooth_y = (reference_x * p_ste.unsqueeze(-1)).sum(1) * 0.25
    y_ste = smooth_y + (y.double() - smooth_y).detach()
    value = q_ste.unsqueeze(-1) * y_ste.unsqueeze(1) + r_ste.transpose(1, 2) @ reference_x
    expected, = torch.autograd.grad(value, reference_x, upstream[:tokens].double())
    result = output.reshape(ranks, rows, streams, D)
    torch.testing.assert_close(collapse[:, :tokens], collapse_ref.expand(ranks, -1, -1), rtol=0.008, atol=0.0001)
    expected_y = (upstream[:tokens].double() * q.bfloat16().double().unsqueeze(-1)).sum(1).float()
    torch.testing.assert_close(dy[:, :tokens], expected_y.expand(ranks, -1, -1), rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(result[:, :tokens], expected.float().expand(ranks, -1, -1, -1), rtol=2e-4, atol=4e-6)
    if result[:, tokens:].count_nonzero():
        raise AssertionError("padding received a residual gradient")
    report["max_abs"] = float((result[:, :tokens] - expected).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--layer", type=int, choices=(43, 44), default=44)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.devices)) != len(args.devices):
        parser.error("use a fresh output path and distinct devices")
    if args.checkpoint is None and not args.compile_only:
        parser.error("device validation requires checkpoint weights")
    report = {"passed": False, "ranks": len(args.devices), "layer": args.layer}
    try:
        run(args, report)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
