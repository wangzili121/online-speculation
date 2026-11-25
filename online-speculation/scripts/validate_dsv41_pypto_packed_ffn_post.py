#!/usr/bin/env python3
"""Check the packed suffix capture and FP32 FFN post-mix VJP on device."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Run the native router with checkpoint layer 39 weights")
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    tp, world = 4, len(devices)
    if world % tp or len(set(devices)) != world or not 1 <= args.last_layers <= 40:
        parser.error("Expected full groups of four distinct devices and a valid last-N suffix")

    import torch
    import pypto.language as pl
    from blockspec.dsv41_packed_ffn_tape import (
        PackedFFNPostVJP, make_packed_ffn_capture, make_packed_ffn_post_vjp, packed_ffn_tape_shapes,
    )
    from models.deepseek_v4_1_flash_a3 import config as native
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    native.set_moe_tokens(16)
    from models.deepseek_v4_1_flash_a3.gate import gate_normalized, golden_gate_reference

    FLASH = native.FLASH
    block, n = args.block, args.last_layers
    hidden, copies, topk = FLASH.hidden_size, FLASH.hc_mult, FLASH.num_experts_per_tok
    local, rows, dp = block // tp, 16, world // tp
    experts, route_rows = FLASH.n_routed_experts, native.MOE_TOKENS
    real_gate = args.checkpoint is not None
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    reverse_graph = make_packed_ffn_post_vjp(block, n, rows, hidden, copies, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, last_layers=n, tp=tp, dp=dp)))
        return

    tape_shapes = packed_ffn_tape_shapes(block, n, tp, hidden, copies)
    residual_shape, mix_shape, vector_shape, route_shape, _ = [shape for shape, _ in tape_shapes.values()]
    save = make_packed_ffn_capture(block, n, tp, hidden, copies, topk)

    @pl.jit(auto_scope=False)
    def capture_rank(
        residual: pl.Tensor[[block, copies, hidden], pl.FP32],
        pre: pl.Tensor[[block, copies], pl.FP32],
        post: pl.Tensor[[block, copies], pl.FP32],
        comb: pl.Tensor[[block, copies, copies], pl.FP32],
        collapsed: pl.Tensor[[block, hidden], pl.BF16],
        combined: pl.Tensor[[block, hidden], pl.BF16],
        normed: pl.Tensor[[route_rows, hidden], pl.BF16],
        gate: pl.Tensor[[experts, hidden], pl.FP32],
        bias: pl.Tensor[[experts], pl.FP32],
        ids: pl.InOut[pl.Tensor[[route_rows, topk], pl.INT32]],
        weights: pl.InOut[pl.Tensor[[route_rows, topk], pl.FP32]],
        saved_residual: pl.Out[pl.Tensor[residual_shape, pl.FP32]],
        saved_mixes: pl.Out[pl.Tensor[mix_shape, pl.FP32]],
        saved_vectors: pl.Out[pl.Tensor[vector_shape, pl.BF16]],
        saved_route_ids: pl.Out[pl.Tensor[route_shape, pl.INT32]],
        saved_route_weights: pl.Out[pl.Tensor[route_shape, pl.FP32]],
        slot: pl.Scalar[pl.INT32],
    ):
        if real_gate == True:
            codes = pl.create_tensor([route_rows, hidden], dtype=pl.INT8)
            scales = pl.create_tensor([1, route_rows], dtype=pl.FP32)
            gate_normalized(normed, gate, bias, local, codes, scales, ids, weights)
        save(residual, pre, post, comb, collapsed, combined, normed, ids, weights,
             saved_residual, saved_mixes, saved_vectors, saved_route_ids, saved_route_weights, slot)

    @pl.jit.host
    def capture(
        residual: pl.Tensor[[world, block, copies, hidden], pl.FP32],
        pre: pl.Tensor[[world, block, copies], pl.FP32],
        post: pl.Tensor[[world, block, copies], pl.FP32],
        comb: pl.Tensor[[world, block, copies, copies], pl.FP32],
        collapsed: pl.Tensor[[world, block, hidden], pl.BF16],
        combined: pl.Tensor[[world, block, hidden], pl.BF16],
        normed: pl.Tensor[[world, route_rows, hidden], pl.BF16],
        gate: pl.Tensor[[world, experts, hidden], pl.FP32],
        bias: pl.Tensor[[world, experts], pl.FP32],
        ids: pl.InOut[pl.Tensor[[world, route_rows, topk], pl.INT32]],
        weights: pl.InOut[pl.Tensor[[world, route_rows, topk], pl.FP32]],
        saved_residual: pl.Out[pl.Tensor[[world, *residual_shape], pl.FP32]],
        saved_mixes: pl.Out[pl.Tensor[[world, *mix_shape], pl.FP32]],
        saved_vectors: pl.Out[pl.Tensor[[world, *vector_shape], pl.BF16]],
        saved_route_ids: pl.Out[pl.Tensor[[world, *route_shape], pl.INT32]],
        saved_route_weights: pl.Out[pl.Tensor[[world, *route_shape], pl.FP32]],
        slot: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(world):
            capture_rank(residual[rank], pre[rank], post[rank], comb[rank], collapsed[rank], combined[rank],
                         normed[rank], gate[rank], bias[rank], ids[rank], weights[rank], saved_residual[rank], saved_mixes[rank],
                         saved_vectors[rank], saved_route_ids[rank], saved_route_weights[rank], slot, device=rank)

    capture_graph = capture.compile(config=config)
    generator = torch.Generator().manual_seed(83914)
    identities = tuple(range(world))
    owners = tuple(range(0, world, tp))
    x = torch.randn(world, block, copies, hidden, generator=generator)
    pre = torch.randn(world, block, copies, generator=generator).softmax(-1)
    post = torch.randn(world, block, copies, generator=generator).sigmoid()
    comb = torch.randn(world, block, copies, copies, generator=generator).softmax(-1)
    collapsed = torch.randn(world, block, hidden, dtype=torch.bfloat16, generator=generator)
    combined = torch.randn(world, block, hidden, dtype=torch.bfloat16, generator=generator)
    normed = torch.randn(world, route_rows, hidden, dtype=torch.bfloat16, generator=generator)
    ids = torch.randint(0, experts, (world, route_rows, topk), dtype=torch.int32, generator=generator)
    weights = torch.randn(world, route_rows, topk, generator=generator).softmax(-1)
    gate = torch.zeros(experts, hidden)
    bias = torch.zeros(experts)
    if real_gate:
        from models.deepseek_v4_1_flash_a3.weights import Checkpoint

        checkpoint = Checkpoint(args.checkpoint)
        gate = checkpoint.tensor("layers.39.ffn.gate.weight").float()
        bias = checkpoint.tensor("layers.39.ffn.gate.bias").float()
    grad = torch.randn(dp, rows, copies, hidden, generator=generator)
    grad_pre = torch.randn(dp, rows, 16, generator=generator)
    grad[:, 7:].zero_()
    grad_pre[:, 7:].zero_()

    errors, timings, route_errors = [], [], []
    with DistributedWorker([capture_graph, reverse_graph], config, persistent=True) as runtime:
        def upload(tensor, to):
            shards = [runtime.alloc_tensor(part.shape, part.dtype, init=part, worker_id=rank)
                      for part, rank in zip(tensor, to, strict=True)]
            return StackedDeviceTensor(shards, tensor.shape, to)

        def read(device, rank):
            cpu = torch.empty(device.shape, dtype=device.dtype)
            runtime.copy_from(cpu.data_ptr(), device.data_ptr, cpu.nbytes, worker_id=rank)
            return cpu

        capture_inputs = [upload(tensor, identities) for tensor in (
            x, pre, post, comb, collapsed, combined, normed,
            gate.repeat(world, 1, 1), bias.repeat(world, 1), ids, weights,
        )]
        tapes = [upload(torch.zeros((world, *shape), dtype=getattr(torch, dtype)), identities)
                 for shape, dtype in tape_shapes.values()]
        ranks = [dict(zip(tape_shapes, (tape.shards[rank] for tape in tapes), strict=True)) for rank in identities]
        runtime.run(capture_graph, *capture_inputs, *tapes, n - 1, config=config)
        if n > 1:
            early = upload(x * .25, identities)
            runtime.run(capture_graph, early, *capture_inputs[1:], *tapes, 0, config=config)
        for rank in identities:
            tensors = [read(tape.shards[rank], rank) for tape in tapes]
            want_mixes = torch.nn.functional.pad(torch.cat((pre[rank], post[rank], comb[rank].flatten(1)), dim=1), (0, 8))
            actual_ids = read(capture_inputs[-2].shards[rank], rank)[:local]
            actual_weights = read(capture_inputs[-1].shards[rank], rank)[:local]
            if real_gate:
                ref_weights, ref_ids = golden_gate_reference(normed[rank, :local], gate, bias)
                torch.testing.assert_close(actual_ids, ref_ids, atol=0, rtol=0)
                torch.testing.assert_close(actual_weights, ref_weights, atol=2e-6, rtol=2e-5)
                route_errors.append(float((actual_weights - ref_weights).abs().max()))
            expected = [x[rank].flatten(0, 1), want_mixes, torch.cat((collapsed[rank], combined[rank], normed[rank, :local])),
                        torch.nn.functional.pad(actual_ids, (0, 16 - topk)),
                        torch.nn.functional.pad(actual_weights, (0, 16 - topk))]
            early_expected = [expected[0] * .25, *expected[1:]]
            for actual, want, early_want in zip(tensors, expected, early_expected, strict=True):
                width = want.shape[0]
                torch.testing.assert_close(actual[-width:], want, atol=0, rtol=0)
                if n > 1:
                    torch.testing.assert_close(actual[:width], early_want, atol=0, rtol=0)
                    assert not torch.any(actual[width:-width])
        final = SimpleNamespace(owners=owners, grad_streams=upload(grad, owners), grad_premix=upload(grad_pre, owners))
        proposer = SimpleNamespace(runtime=runtime, config=config, ranks=ranks, block=block, last_layers=n)
        reverse = PackedFFNPostVJP(proposer, final, reverse_graph, FLASH.num_hidden_layers)
        for slot, owner in enumerate(owners):
            student = SimpleNamespace(owner=owner, ffn_tape=dict(enumerate(ranks)))
            incoming = SimpleNamespace(owner=owner, streams=final.grad_streams.shards[slot], premix=final.grad_premix.shards[slot])
            outputs = [reverse(student, incoming) for _ in range(10)]
            result = outputs[-1]
            value, residual, gating = [read(tensor, owner) for tensor in (result.value, result.residual, result.mixes)]
            value_ref, res_ref, post_ref, comb_ref = [t.double().requires_grad_()
                                                     for t in (combined[owner], x[owner], post[owner], comb[owner])]
            update = value_ref[:, None, :] * post_ref[:, :, None] + torch.einsum("bst,bsh->bth", comb_ref, res_ref)
            dy = torch.zeros_like(update)
            dy[:rows] = grad[slot]
            dv, dr, dpost, dcomb = torch.autograd.grad(update, (value_ref, res_ref, post_ref, comb_ref), dy)
            want_grad_pre = torch.zeros(block, copies)
            want_grad_pre[:rows] = grad_pre[slot, :, :copies]
            dmix = torch.nn.functional.pad(torch.cat((want_grad_pre, dpost, dcomb.flatten(1)), 1), (0, 8))
            for actual, want in zip((value, residual, gating), (dv, dr.flatten(0, 1), dmix), strict=True):
                delta = actual.double() - want
                relative = float(delta.norm() / want.norm())
                assert relative < 2e-5, relative
                errors.append(relative)
            timings.append(1000 * median(result.seconds for result in outputs[1:]))
    print(json.dumps(dict(passed=True, devices=devices, tp=tp, block=block, last_layers=n, rows_with_gradient=7,
                          capture_byte_exact=True, distinct_layer_slots=n > 1,
                          max_relative_l2=max(errors), warm_median_ms=timings,
                          checkpoint_routes=real_gate, max_route_weight_error=max(route_errors, default=None))), flush=True)


if __name__ == "__main__":
    main()
