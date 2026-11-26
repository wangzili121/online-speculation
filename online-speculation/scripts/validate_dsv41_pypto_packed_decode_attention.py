#!/usr/bin/env python3
"""Compile the resident 1122 Decode attention and final head for local TP groups."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--decode-capacity", type=int, default=16, choices=(16, 80))
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--slots-per-dp", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Run a real-weight TP4 layer-state smoke after compilation")
    parser.add_argument("--reuse", action="store_true", help="Run on-device using the artifacts in --output")
    args = parser.parse_args()

    from blockspec.dsv41_packed_decode import make_decode_final_head
    from models.deepseek_v4_1_flash_a3 import decode_fwd, driver
    from pypto.ir import DistributedConfig

    devices = list(map(int, args.devices.split(",")))
    ranks = len(devices)
    artifacts = {}
    if args.reuse:
        if not args.checkpoint:
            parser.error("--reuse requires --checkpoint")
        artifacts = {name: str(args.output / name) for name in ("attention", "head")}
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, stage in (("attention", decode_fwd.make_decode_attention_stage(ranks)),
                            ("head", make_decode_final_head(ranks))):
            specs = driver.build_specs(stage, mode="decode", capacity=args.decode_capacity,
                                       history_capacity=args.context, request_slots=args.slots_per_dp,
                                       prepare_windows=True)
            result = driver.run(
                fn=stage, specs=specs, compile_only=True,
                config=dict(backend_type="a2a3", execution_mode=False,
                            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                            save_kernels=True, save_kernels_dir=str((args.output / name).resolve())),
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
            artifacts[name] = str(result.work_dir)
    report = dict(passed=True, compiled=True, local_ranks=ranks, rows=args.decode_capacity,
                  context=args.context, slots_per_dp=args.slots_per_dp, artifacts=artifacts)
    if args.checkpoint:
        if ranks != 4 or args.slots_per_dp != 1:
            parser.error("The real-weight layer-state fixture needs one TP4 group and one request slot")
        report["device_checks"] = exercise(args, devices, artifacts)
    print(json.dumps(report), flush=True)


def exercise(args, devices, artifacts):
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from blockspec.dsv41_packed_decode import decode_attention_scratch
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.host import prepare_resident_inputs
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
    from models.deepseek_v4_1_flash_a3.utils import decode_cache_views, DECODE_CACHE_NAMES, prepare_decode_inputs
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedCompiledProgram, DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    attention, head = (DistributedCompiledProgram.from_dir(artifacts[name]) for name in ("attention", "head"))
    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    engram = HostEngram(checkpoint, Tokenizer.from_file(str(checkpoint.root / "tokenizer.json")))
    rows, model = args.decode_capacity, FLASH
    token_ids = [42, 43, 44]
    prefill = prepare_resident_inputs(weights, token_ids, rows, max_seq_len=args.context, engram=engram)
    decode = prepare_decode_inputs(prefill)
    caches = decode_cache_views({source: prefill[source] for source in DECODE_CACHE_NAMES.values()})
    view_rows = caches["window_cache_pool"].shape[0] // model.num_hidden_layers
    scratch = decode_attention_scratch(rows, decode["c1a_index_block_table"].shape[-1])
    scratch.update(logits=((MAX_LOGIT_ROWS, model.vocab_size), torch.float32),
                   sampled_ids=((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), torch.int32))
    # The first host processes rank 0..7. Simulate one TP group on the
    # second host to exercise a different global checkpoint shard on each chip.
    layers = (0, 1, 2, 3, 20, 24, 25)
    ranks = tuple(range(len(devices)))
    with DistributedWorker([attention, head], config, persistent=True, reset_persistent_windows=False) as runtime:
        handles = [{name: runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    for name, (shape, dtype) in weights.tensor_layouts.items()} for rank in ranks]
        with ThreadPoolExecutor(max_workers=len(devices)) as pool_threads:
            list(pool_threads.map(
                lambda rank: weights.upload(runtime, rank + 8, handles[rank], layers=layers, worker_id=rank),
                ranks,
            ))
        for rank, rank_handles in enumerate(handles):
            inputs = dict(decode, **caches, count=torch.tensor([len(token_ids), rank], dtype=torch.int32))
            rank_handles.update({name: runtime.alloc_tensor(value.shape, value.dtype, worker_id=rank, init=value)
                                 for name, value in inputs.items()})
            rank_handles.update({name: runtime.alloc_tensor(shape, dtype, worker_id=rank)
                                 for name, (shape, dtype) in scratch.items()})

        def bound(program):
            for param in program._get_metadata()[0]:
                name = param.name.removesuffix("__ssa_v0")
                if name not in ("layer_index", "execute_layers"):
                    shards = [rank[name] for rank in handles]
                    yield StackedDeviceTensor(shards, (len(ranks), *shards[0].shape), ranks)

        def read(name):
            handle = handles[0][name]
            value = torch.empty(handle.shape, dtype=handle.dtype)
            runtime.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes, worker_id=0)
            return value

        attention_args, head_args = tuple(bound(attention)), tuple(bound(head))
        runtime.run(attention, *attention_args, 0, 0)
        runtime.run(head, *head_args, 0)
        for layer in layers:
            runtime.run(attention, *attention_args, layer, 1)
            output = read("attention_hidden")[:len(token_ids)]
            assert torch.isfinite(output).all() and output.abs().max().item() > 0
        observed = read("window_cache_pool").view(model.num_hidden_layers, view_rows, *caches["window_cache_pool"].shape[1:])
        changed = (observed != 0).reshape(model.num_hidden_layers, -1).any(dim=1).nonzero().flatten().tolist()
        assert changed == list(layers), f"unexpected window cache writes: {changed}"
        runtime.run(head, *head_args, 1)
        logits = read("logits")[0]
        assert torch.isfinite(logits).all() and logits.abs().max().item() > 0
        token = read("sampled_ids")[0, 0].item()
        assert token == logits.argmax().item()
    return dict(checkpoint_ranks=[8, 9, 10, 11], attention_layers=list(layers), mutated_window_layers=changed,
                finite_active_outputs=True, head_argmax_verified=True)


if __name__ == "__main__":
    main()
