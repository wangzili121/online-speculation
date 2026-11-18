"""Abort unless all 16 A2 ranks can reduce and exchange data across the nodes."""

from datetime import timedelta
import os

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 16 or int(os.environ["LOCAL_WORLD_SIZE"]) != 8 or torch.npu.device_count() != 8:
        raise RuntimeError("expected two hosts with eight visible NPU devices each")

    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(minutes=3))
    try:
        value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        dist.all_reduce(value)
        expected_sum = world * (world + 1) // 2
        if value.item() != expected_sum:
            raise RuntimeError(f"rank {rank}: unexpected HCCL sum {value.item()}")
        # Fixed-size all-to-all mirrors the cross-node expert exchange. Each
        # destination gets a distinct marker from every source rank.
        outgoing = torch.arange(world, dtype=torch.float32, device=device) + rank * world
        received = torch.empty_like(outgoing)
        dist.all_to_all_single(received, outgoing)
        expected = torch.arange(world, dtype=torch.float32) * world + rank
        if not torch.equal(received.cpu(), expected):
            raise RuntimeError(f"rank {rank}: HCCL all-to-all row mismatch")
        print(f"HCCL OK global_rank={rank} local_rank={local_rank} world={world}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
