#!/usr/bin/env python3
"""Compare resident packed Adam and publication with Torch across DP owners."""

import argparse
import json
from dataclasses import asdict
from math import prod, sqrt
from pathlib import Path
from types import SimpleNamespace

import torch

from blockspec.dsv41_packed_adam import PackedAdam, compile_packed_adam, packed_parameters


def checkpoint_bank(path, model, layer, lane, tp):
    from blockspec.pypto_dsv41_draft import (
        OUTPUT_TENSORS, pack_draft_input_projection, pack_draft_output_projection, prepare_output,
    )
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from validate_dsv41_pypto_packed_qkv import checkpoint_operands

    checkpoint = Checkpoint(path)
    _, operands = checkpoint_operands(path, layer, lane)
    prefix = f"layers.{layer}.attn."
    output = {key: checkpoint.tensor(prefix + key) for key in OUTPUT_TENSORS}
    packed = pack_draft_output_projection(prepare_output(output, asdict(model), tp_rank=lane, tp=tp))
    return {
        "draft_q_a_kv": pack_draft_input_projection(operands),
        "draft_q_gamma": operands["q_norm"],
        "draft_q_b": operands["wq_b"],
        "draft_kv_gamma": operands["kv_norm"],
        "draft_sink": checkpoint.tensor(prefix + "attn_sink").float().chunk(tp)[lane],
        "draft_wo_a": packed["wo_a"],
        "draft_wo_b": packed["wo_b"],
    }


def exercise(runtime, config, model, programs, tp, world, initial, *, staged=False):
    from pypto.runtime import StackedDeviceTensor

    layers = model.num_hidden_layers
    parameters = packed_parameters(model, tp)

    def upload(value, rank):
        return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

    def read_slice(bank, logical_rank, offset, count):
        values = torch.empty(count, dtype=bank.dtype)
        runtime.copy_from(values.data_ptr(), bank.data_ptr, values.nbytes,
                          src_offset=offset * values.element_size(), worker_id=logical_rank)
        return values

    banks = [{p.name: upload(initial[rank % tp][p.name], rank) for p in parameters} for rank in range(world)]
    proposer = SimpleNamespace(runtime=runtime, config=config, last_layers=layers, ranks=banks, tp=tp, world=world)
    if staged:
        from blockspec.dsv41_staged_adam import StagedPackedAdam

        class HostPermutation:
            # Emulate the DP collective for this fixture so the actual device packing and optimizer
            # can be exercised even on a one-card validation machine.
            mappings = SimpleNamespace(register=lambda *_: None)

            def exchange_resident(self, stage, transfers, *, collective, group):
                assert collective == "all_reduce" and group == "dp"
                for lane in range(tp):
                    ranks = range(lane, world, tp)
                    rows = [read_slice(transfers[rank][0][0], rank, 0, transfers[rank][0][0].shape[0])
                            for rank in ranks]
                    reduced = torch.stack(rows).sum(0)
                    for rank in ranks:
                        destination = transfers[rank][0][1]
                        runtime.copy_to(destination.data_ptr, reduced.data_ptr(), reduced.nbytes, worker_id=rank)

        def reverse(stage, *, owner, feedback, finished, consume):
            for layer in reversed(range(layers)):
                consume(layer, SimpleNamespace(owner=owner, parameter_gradients={
                    p.name: gradients[layer, p.name] for p in parameters
                }))
                for p in parameters:
                    check(layer, p, before=True)
            return dict(owner=owner, teacher_rows=7, loss=.012)

        proposer.target = SimpleNamespace(link=SimpleNamespace(exchange=lambda _stage, value: value))
        proposer.ffn = SimpleNamespace(moe=SimpleNamespace(transport=HostPermutation()))
        optimizer = StagedPackedAdam(proposer, SimpleNamespace(run=reverse), model, programs)
    else:
        adam = PackedAdam(proposer, model, programs)
    references, gradients, published = {}, {}, {}
    samples = None if max(prod(p.shape) for p in parameters) < 32768 else 32
    max_abs = 0.
    for layer in range(layers):
        for p in parameters:
            size = prod(p.shape)
            count = min(samples, size) if samples else size
            lanes = 1 if p.replicated else tp
            for lane in range(lanes):
                weight = initial[lane][p.name][layer].flatten()[:count].float().clone().requires_grad_()
                references[layer, p.name, lane] = weight, torch.optim.Adam([weight], lr=.001, foreach=False)
                published[layer, p.name, lane] = weight.detach().to(initial[lane][p.name].dtype).clone()

    def check(layer, p, before=False):
        nonlocal max_abs
        size = prod(p.shape)
        count = min(samples, size) if samples else size
        for rank in range(world):
            lane = 0 if p.replicated else rank % tp
            name = p.name
            weight, _ = references[layer, name, lane]
            actual = read_slice(banks[rank][name], rank, layer * size, count)
            canonical = read_slice(banks[lane][name], lane, layer * size, count)
            torch.testing.assert_close(actual, canonical, atol=0, rtol=0)
            expected = weight.detach().to(actual.dtype)
            if before:
                torch.testing.assert_close(actual, published[layer, name, lane], atol=0, rtol=0)
            elif p.fp32:
                torch.testing.assert_close(actual.float(), expected.float(), atol=2e-7, rtol=2e-6)
            else:
                torch.testing.assert_close(actual.float(), expected.float(), atol=2e-6, rtol=1 / 128)
            max_abs = max(max_abs, float((actual.float() - expected.float()).abs().max()))
            if not before:
                published[layer, name, lane] = canonical

    results = []
    for index, (owner, invalid) in enumerate(((0, False), (world - tp, False), (0, True), (0, False))):
        norm_sq = 0.
        for layer in range(layers):
            for p_index, p in enumerate(parameters):
                size = prod(p.shape)
                values = []
                ranks = tuple(range(0, world, tp)) if p.replicated else tuple(range(world))
                for rank in ranks:
                    lane = 0 if p.replicated else rank % tp
                    active = owner <= rank < owner + tp
                    magnitude = (lane + layer + p_index + 1) * (.003 if index % 2 else -.002)
                    value = torch.full(p.shape, magnitude if active else 2.)
                    flat = value.view(-1)
                    count = min(samples, size) if samples else size
                    if active:
                        flat[:count:2].mul_(-2.)
                        norm_sq += float(value.double().square().sum())
                        references[layer, p.name, lane][0].grad = flat[:count].clone()
                        if invalid and p.name == "draft_sink":
                            flat[0] = torch.nan
                    values.append(upload(value, rank))
                gradients[layer, p.name] = StackedDeviceTensor(values, (len(ranks), *p.shape), ranks)

        def differentiate(consume, owner=owner):
            for layer in reversed(range(layers)):
                consume(SimpleNamespace(layer=layer, owner=owner, parameter_gradients={
                    p.name: gradients[layer, p.name] for p in parameters
                }))
                for p in parameters:
                    check(layer, p, before=True)
            return SimpleNamespace(value=.012, rows=7)

        if staged:
            result = SimpleNamespace(**optimizer.update(
                f"fixture/{index}", owner=owner, feedback=None, finished=False, rate=.001, max_norm=1.,
            ))
        else:
            result = SimpleNamespace(**asdict(adam.update(differentiate, rate=.001)))
        assert result.applied != invalid
        if not invalid:
            expected_norm = sqrt(norm_sq)
            assert abs(result.gradient_norm - expected_norm) <= expected_norm * 1e-5, (result, expected_norm)
            for weight, reference_optimizer in references.values():
                weight.grad.mul_(result.clip)
                reference_optimizer.step()
        for layer in range(layers):
            for p in parameters:
                check(layer, p, before=invalid)
        results.append({**vars(result), "owner": owner, "invalid_gradient": invalid})
        for stacked in gradients.values():
            for rank, tensor in zip(stacked.worker_ids, stacked.shards, strict=True):
                runtime.free_tensor(tensor, worker_id=rank)
        gradients.clear()
    assert [result["version"] for result in results] == [1, 2, 2, 3]
    report = dict(passed=True, executed=True, tp=tp, world=world, layers=layers,
                  execution_weight_max_abs=max_abs, updates=results,
                  gradient_transport="host_fixture" if staged else "local_p2p")
    if staged:
        report["gradient_scratch_bytes_per_rank"] = sum(packet[0].nbytes for packet in optimizer.packets)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--fixture-tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--checkpoint-lane", type=int, choices=range(4),
                        help="Check one real TP4 shard on one NPU")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--staged", action="store_true", help="Exercise staged gradient packing with a host DP fixture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    devices = [int(part) for part in args.devices.split(",")]
    tp, world = args.fixture_tp, len(devices)
    isolated_shard = args.checkpoint_lane is not None
    if isolated_shard and (not args.checkpoint or world != 1 or tp != 1 or args.compile_only):
        parser.error("A single checkpoint lane requires one device, TP1, and real execution")
    if (len(set(devices)) != world or world % tp
            or ((args.compile_only or args.checkpoint) and tp != 4 and not isolated_shard)):
        parser.error("Use one or more complete, distinct TP groups. Production banks require TP4.")

    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    if args.checkpoint or args.compile_only:
        from models.deepseek_v4_1_flash_a3.config import FLASH

        model = SimpleNamespace(**asdict(FLASH))
        model.num_hidden_layers = 1
        if isolated_shard:
            model.num_attention_heads //= 4
            model.o_groups //= 4
    else:
        model = SimpleNamespace(hidden_size=128, q_lora_rank=64, head_dim=64, num_attention_heads=tp * 8,
                                o_groups=tp * 2, o_lora_rank=32, num_hidden_layers=2)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    if args.staged:
        from blockspec.dsv41_staged_adam import make_staged_adam_graphs

        compiled = iter(make_staged_adam_graphs(model, world, tp).values())
        programs = {key: tuple(next(compiled).compile(config=config) for _ in range(4))
                    for key in dict.fromkeys(p.program_key for p in packed_parameters(model, tp))}
    else:
        programs = compile_packed_adam(config, model, world, tp)
    entries = [entry for group in programs.values() for entry in group]
    result = dict(passed=True, compiled=len(entries), tp=tp, world=world)
    print(f"Compiled {len(entries)} optimizer graphs", flush=True)
    if not args.compile_only:
        rng = torch.Generator().manual_seed(61237)
        initial = []
        for lane in range(tp):
            checkpoint_lane = args.checkpoint_lane if isolated_shard else lane
            bank = checkpoint_bank(args.checkpoint, FLASH, args.layer, checkpoint_lane, 4) if args.checkpoint else None
            values = {}
            for p in packed_parameters(model, tp):
                dtype = torch.float32 if p.fp32 else torch.bfloat16
                shape = (model.num_hidden_layers, *p.shape)
                if p.replicated and lane:
                    values[p.name] = initial[0][p.name]
                elif bank:
                    values[p.name] = bank[p.name].reshape(shape).to(dtype).contiguous()
                else:
                    values[p.name] = torch.randn(shape, dtype=dtype, generator=rng).mul_(.06)
            initial.append(values)
        with DistributedWorker(entries, config, persistent=True, reset_persistent_windows=False) as runtime:
            print("Optimizer device worker ready", flush=True)
            result.update(exercise(runtime, config, model, programs, tp, world, initial, staged=args.staged))
        result["checkpoint_layer"] = args.layer if args.checkpoint else None
        result["checkpoint_lane"] = args.checkpoint_lane
    text = json.dumps(result)
    if args.output:
        args.output.write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
