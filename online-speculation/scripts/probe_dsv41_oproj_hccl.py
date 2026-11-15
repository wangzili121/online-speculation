#!/usr/bin/env python3
"""Use real V4.1 output projection shards to test BF16 HCCL repeatability."""

import argparse
import json
import os
from pathlib import Path
from time import sleep

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch_npu
from safetensors import safe_open


def all_reduce_repeats(value, rank):
    reference = value.clone()
    dist.all_reduce(reference)
    different = 0
    for step in range(32):
        # Exercise different arrival orders with the same group inputs.
        sleep(((rank + step) % dist.get_world_size()) * 0.001)
        output = value.clone()
        dist.all_reduce(output)
        different += int(not torch.equal(output, reference))
    return reference, different


def all_reduce_mean_ms(value, iterations):
    output = value.clone()
    for _ in range(10):
        output.copy_(value)
        dist.all_reduce(output)
    torch.npu.synchronize()
    dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    finish = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output.copy_(value)
        dist.all_reduce(output)
    finish.record()
    finish.synchronize()
    return start.elapsed_time(finish) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--timing-iterations", type=int, default=0,
                        help="also measure the mean input copy plus AllReduce device time")
    args = parser.parse_args()
    if args.timing_iterations < 0:
        parser.error("--timing-iterations must be nonnegative")
    index = json.loads((args.model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        upstream_tp = 8
        if dist.get_world_size() > upstream_tp:
            raise ValueError("this fixture uses up to eight of the actual TP8 shards")
        o_rank = config["o_lora_rank"]
        width = config["num_attention_heads"] // upstream_tp * config["head_dim"]
        begin, end = rank * o_rank, (rank + 1) * o_rank

        def load(suffix):
            name = f"layers.0.attn.{suffix}.weight"
            with safe_open(args.model / index[name], framework="pt", device="cpu") as source:
                return source.get_tensor(name)

        wo_a = load("wo_a")[begin:end].T.contiguous().unsqueeze(0).npu()
        wo_b = load("wo_b")[:, begin:end].contiguous().npu()
        rng = torch.Generator().manual_seed(1017 + rank)
        activations = torch.randn((32, 1, width), generator=rng).mul(.2).bfloat16().npu()

        def project(value):
            hidden = torch_npu.npu_transpose_batchmatmul(
                value, wo_a, bias=None, scale=None, perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2), perm_y=(1, 0, 2), batch_split_factor=1,
            )
            return F.linear(hidden.flatten(1), wo_b)

        local_one = project(activations[1:2])
        local_block = project(activations)
        local_difference = (local_one[0] != local_block[1]).sum().to(torch.int32)
        dist.all_reduce(local_difference)
        single, repeats_one = all_reduce_repeats(local_one, rank)
        block, repeats_block = all_reduce_repeats(local_block, rank)
        if rank == 0:
            delta = (single[0].float() - block[1].float()).abs()
            print(json.dumps({
                "world_size": dist.get_world_size(),
                "HCCL_DETERMINISTIC": os.environ.get("HCCL_DETERMINISTIC"),
                "changed_repeats": [repeats_one, repeats_block],
                "local_projection_changed_values_all_ranks": int(local_difference),
                "M1_vs_M32_changed_values": int(delta.count_nonzero()),
                "M1_vs_M32_max_abs": float(delta.max()),
            }), flush=True)
        if args.timing_iterations:
            for value in (local_one, local_block):
                timing = all_reduce_mean_ms(value, args.timing_iterations)
                if rank == 0:
                    print(json.dumps({
                        "rows": value.size(0),
                        "HCCL_DETERMINISTIC": os.environ.get("HCCL_DETERMINISTIC"),
                        "iterations": args.timing_iterations,
                        "mean_ms_including_copy": timing,
                    }), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
