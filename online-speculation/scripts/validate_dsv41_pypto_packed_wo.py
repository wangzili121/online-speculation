#!/usr/bin/env python3
"""Validate the resident NZ output projection input VJP against FP64."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace


def reference_output(delta, weights, cosine, sine, heads, rank):
    import torch

    block, groups, tail = delta.size(0), weights["wo_a"].size(0), 2 * cosine.size(1)
    middle = (delta.double() @ weights["wo_b"].double().T).reshape(block, groups, rank)
    value = torch.einsum("bgr,grd->bgd", middle, weights["wo_a"].double()).reshape(block, heads, -1)
    even, odd = value[..., -tail:].reshape(block, heads, -1, 2).unbind(-1)
    c, s = cosine[:, None].double(), sine[:, None].double()
    value[..., -tail:] = torch.stack((even * c - odd * s, odd * c + even * s), -1).flatten(-2)
    return value.flatten(0, 1)


def measure(result, reference, active, heads, times, limit):
    import torch

    error = result.double() - reference
    relative = float(error.norm() / reference.norm())
    assert relative < limit and not torch.count_nonzero(result[active * heads:]), (active, relative)
    return dict(relative_l2=relative, max_abs=float(error.abs().max()), warm_median_ms=median(times))


def check_tp(runtime, graph, config, world, checkpoint, model, block, tp):
    import torch
    from blockspec.dsv41_packed_wo_vjp import PackedWoInputVJP
    from blockspec.pypto_dsv41_draft import OUTPUT_TENSORS, pack_draft_output_projection, prepare_output
    from pypto.runtime import StackedDeviceTensor

    heads, dim, tail = model.num_attention_heads // tp, model.head_dim, model.qk_rope_head_dim
    layers = (0, None, 39)
    projected = [{} for _ in layers]
    ranks, gradients = [], []
    generator = torch.Generator().manual_seed(391442)
    angles = torch.randn(2, 4 * block, tail // 2, generator=generator)
    cosine, sine = angles.cos(), angles.sin()
    for rank in range(world):
        weights = []
        for j, layer in enumerate(layers):
            if layer is None:
                weights.append(weights[0])
                continue
            if rank < tp:
                raw = {name: checkpoint.tensor(f"layers.{layer}.attn.{name}") for name in OUTPUT_TENSORS}
                projected[j][rank] = prepare_output(raw, asdict(model), tp_rank=rank, tp=tp)
            weights.append(pack_draft_output_projection(projected[j][rank % tp]))
        host = {f"draft_{name}": torch.stack([row[name] for row in weights]) for name in ("wo_a", "wo_b")}
        host.update(freqs_cos=cosine, freqs_sin=sine)
        ranks.append({name: runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=rank)
                      for name, tensor in host.items()})
        gradients.append(runtime.alloc_tensor((block, model.hidden_size), torch.float32, worker_id=rank))
    delta = StackedDeviceTensor(gradients, (world, block, model.hidden_size), tuple(range(world)))
    proposer = SimpleNamespace(runtime=runtime, config=config, world=world, block=block, tp=tp, ranks=ranks)
    reverse = PackedWoInputVJP(proposer, graph, model)
    cases = []
    for owner, layer, active, position in ((0, 2, 1, 3), (world - tp, 0, 7, block + 7), (0, 2, block, 2 * block)):
        seed = torch.zeros(block, model.hidden_size)
        seed[:active].normal_(generator=generator)
        for rank in range(owner, owner + tp):
            runtime.copy_to(gradients[rank].data_ptr, seed.data_ptr(), seed.nbytes, worker_id=rank)
        student = SimpleNamespace(owner=owner, position=position, rows=active)
        incoming = SimpleNamespace(owner=owner, layer=layer, rows=active, delta=delta, residual=object())
        samples = [reverse(student, incoming) for _ in range(8)]
        result = samples[-1]
        assert result.residual is incoming.residual
        metrics = []
        for rank in range(owner, owner + tp):
            output = torch.empty(block * heads, dim)
            runtime.copy_from(output.data_ptr(), result.context.shards[rank].data_ptr, output.nbytes, worker_id=rank)
            profile = int(layer >= 2)
            ref = reference_output(seed, projected[layer][rank % tp], cosine[profile, position:position + block],
                                   sine[profile, position:position + block], heads, model.o_lora_rank)
            metrics.append(measure(output, ref, active, heads, [sample.seconds * 1000 for sample in samples[1:]], 2e-5))
        cases.append(dict(owner=owner, layer=layers[layer], position=position, active_rows=active, lanes=metrics))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices")
    parser.add_argument("--fixture-tp", type=int, choices=(2, 4), default=4)
    parser.add_argument("--block", type=int, default=32, choices=(16, 32))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only and args.checkpoint is None:
        parser.error("--checkpoint is required for device execution")

    import torch
    from blockspec.dsv41_packed_wo_vjp import make_packed_wo_input_vjp
    from blockspec.pypto_dsv41_draft import OUTPUT_TENSORS, pack_draft_output_projection, prepare_output
    from models.deepseek_v4_1_flash_a3.config import FLASH as model
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    devices = [args.device] if args.devices is None else [int(value) for value in args.devices.split(",")]
    world = len(devices)
    block, tp = args.block, args.fixture_tp
    if len(set(devices)) != world or (world > 1 and world % tp):
        parser.error("--devices must contain distinct devices in complete TP groups")
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    modes = (("compensated", True),) if world > 1 else (("bf16", False), ("compensated", True))
    graphs = {name: make_packed_wo_input_vjp(block, model, tp, compensated=compensated, world=world)
              .compile(config=config) for name, compensated in modes}
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=list(graphs), block=block, tp=tp, world=world)))
        return

    checkpoint = Checkpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(391441)
    heads, dim, tail = model.num_attention_heads // tp, model.head_dim, model.qk_rope_head_dim
    cases = []
    if world > 1:
        graph = graphs["compensated"]
        with DistributedWorker([graph], config, persistent=True) as runtime:
            cases = check_tp(runtime, graph, config, world, checkpoint, model, block, tp)
        print(json.dumps(dict(passed=True, block=block, tp=tp, world=world, checkpoint=str(args.checkpoint), cases=cases)))
        return
    with DistributedWorker(list(graphs.values()), config, persistent=True) as runtime:
        def upload(value):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=0)

        for layer, rank in ((36, 3), (39, 0)):
            weights = prepare_output({name: checkpoint.tensor(f"layers.{layer}.attn.{name}") for name in OUTPUT_TENSORS},
                                     asdict(model), tp_rank=rank, tp=tp)
            bank = pack_draft_output_projection(weights)
            bank = [upload(bank["wo_a"].reshape(model.o_groups // tp, -1, 16)),
                    upload(bank["wo_b"].reshape(-1, 16))]
            angles = torch.randn(block, tail // 2, generator=generator)
            cosine, sine = angles.cos(), angles.sin()
            table = [upload(value) for value in (cosine, sine)]
            result = torch.empty(block * heads, dim)
            result_device = upload(result)
            for active in (1, 7, block):
                delta = torch.zeros(block, model.hidden_size)
                delta[:active].normal_(generator=generator)
                seed = upload(delta)
                reference = reference_output(delta, weights, cosine, sine, heads, model.o_lora_rank)
                modes = [(name, graph, active) for name, graph in graphs.items()]
                if active < block:
                    modes.append(("compensated_full", graphs["compensated"], block))
                timing = {name: [] for name, _, _ in modes}
                metrics = {}
                for iteration in range(8):
                    for name, graph, rows in modes:
                        start = perf_counter()
                        runtime.run(graph, seed, *bank, *table, result_device, rows, config=config)
                        timing[name].append((perf_counter() - start) * 1000)
                        if iteration == 7:
                            runtime.copy_from(result.data_ptr(), result_device.data_ptr, result.nbytes, worker_id=0)
                            limit = .006 if name == "bf16" else 2e-5
                            metrics[name] = measure(result, reference, active, heads, timing[name][1:], limit)
                cases.append(dict(layer=layer, tp_rank=rank, active_rows=active, metrics=metrics))
                runtime.free_tensor(seed, worker_id=0)
            for value in (*bank, *table, result_device):
                runtime.free_tensor(value, worker_id=0)
    print(json.dumps(dict(passed=True, block=block, tp=tp, checkpoint=str(args.checkpoint), cases=cases)))


if __name__ == "__main__":
    main()
