#!/usr/bin/env python3
"""Check a staged all-rank AdamW update of a real packed MLA head."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.pypto_glm53_head import head_layout, initialize_rank_head
from blockspec.pypto_glm53_online_optimizer import (
    make_mla_head_adamw,
    make_mla_head_optimizer_init,
    make_mla_head_publish,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= len(set(args.devices)) == len(args.devices) <= 8 or args.output.exists():
        parser.error("use up to eight distinct devices and a fresh output path")

    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, StackedDeviceTensor

    torch.set_num_threads(4)
    ranks = len(args.devices)
    layout, total = head_layout()
    length = layout["index_q"][0]
    weights = Checkpoint(args.checkpoint)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=args.devices, num_sub_workers=0))
    gradient = torch.full((ranks, 1, length), 0.03)
    gradient[..., 1::2] = -0.04
    master = torch.full_like(gradient, float("nan"))
    staged = torch.empty_like(gradient, dtype=torch.bfloat16)
    first, second = torch.ones_like(gradient), torch.ones_like(gradient)
    live = torch.empty((ranks, 3, total), dtype=torch.bfloat16)
    live[:, 0].fill_(0.125)
    live[:, 2].fill_(-0.25)
    for rank in range(ranks):
        live[rank, 1].copy_(initialize_rank_head(weights, 43, rank))

    lr, decay = 1e-3, 0.01
    scalars = (lr, decay, 1 / (1 - 0.9), 1 / (1 - 0.999))
    report = {"passed": False, "ranks": ranks, "trainable_prefix": length, "packed_layer": total}
    try:
        initialize = make_mla_head_optimizer_init(ranks).compile(config=config)
        update = make_mla_head_adamw(ranks).compile(gradient, master, first, second, staged, *scalars, config=config)
        publish = make_mla_head_publish(ranks).compile(staged, live[:, 1:2, :].contiguous(), config=config)
        with DistributedWorker([initialize, update, publish], config=config, persistent=True) as worker:
            handles = [worker.alloc_stacked_tensor(value.share_memory_())
                       for value in (gradient, master, first, second, staged, live)]
            offset = live.element_size() * total
            middle = StackedDeviceTensor(
                [worker.view_tensor(handle, (1, total), byte_offset=offset, worker_id=rank)
                 for rank, handle in enumerate(handles[-1].shards)],
                (ranks, 1, total), tuple(range(ranks)),
            )

            def download(rank, shard, shape, dtype):
                host = torch.empty(shape, dtype=dtype).share_memory_()
                worker.copy_from(host.data_ptr(), shard.data_ptr, host.nbytes, worker_id=rank)
                return host

            worker.run(initialize, middle, *handles[1:4], config=config)
            for rank in range(ranks):
                initialized = download(rank, handles[1].shards[rank], (1, length), torch.float32)[0]
                torch.testing.assert_close(initialized, live[rank, 1, :length].float(), atol=0, rtol=0)
                for index in (2, 3):
                    moment = download(rank, handles[index].shards[rank], (1, length), torch.float32)
                    if moment.count_nonzero():
                        raise AssertionError("optimizer moments were not initialized to zero")
            worker.run(update, *handles[:5], *scalars, config=config)
            for rank, shard in enumerate(handles[-1].shards):
                before = download(rank, shard, (3, total), torch.bfloat16)
                if not torch.equal(before[1], live[rank, 1]):
                    raise AssertionError("staging changed the live draft")
            worker.run(publish, handles[4], middle, config=config)
            max_difference = 0.0
            for rank, shard in enumerate(handles[-1].shards):
                after = download(rank, shard, (3, total), torch.bfloat16)
                if not torch.equal(after[0], live[rank, 0]) or not torch.equal(after[2], live[rank, 2]):
                    raise AssertionError("a neighboring layer changed")
                if not torch.equal(after[1, length:], live[rank, 1, length:]):
                    raise AssertionError("the indexer parameters changed")
                g = gradient[rank, 0]
                original = live[rank, 1, :length].float()
                expected = original - lr * (g / (g.abs() + 1e-8) + decay * original)
                got_master = download(rank, handles[1].shards[rank], (1, length), torch.float32)[0]
                torch.testing.assert_close(got_master, expected, atol=2e-7, rtol=3e-6)
                packed = download(rank, handles[4].shards[rank], (1, length), torch.bfloat16)[0]
                if not torch.equal(after[1, :length], packed) or torch.equal(after[1], live[rank, 1]):
                    raise AssertionError("the inference view did not receive the staged update")
                max_difference = max(max_difference, float((got_master - expected).abs().max()))
        report.update(passed=True, resident_optimizer_init=True, master_max_abs=max_difference,
                      indexer_and_neighboring_layers_unchanged=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
