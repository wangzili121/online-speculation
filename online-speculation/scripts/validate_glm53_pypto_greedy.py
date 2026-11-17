#!/usr/bin/env python3
"""Exercise the device greedy result on sharded GLM logits."""

import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = len(args.devices)
    if ranks not in (1, 2, 4, 8) or len(set(args.devices)) != ranks or args.output.exists():
        parser.error("use one, two, four or eight distinct devices and a fresh output path")
    sys.argv = [sys.argv[0], "--tp", str(ranks), "--ep", str(ranks)]
    from models.glm5_3_flash.config import LOCAL_VOCAB
    from models.glm5_3_flash.lm_head import HEAD_VOCAB_PAD
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    from blockspec.pypto_glm53_greedy import make_sharded_top1, top1_buffer_rows

    torch.set_num_threads(4)
    rows = top1_buffer_rows(6)
    logits = torch.empty((ranks, rows, HEAD_VOCAB_PAD)).share_memory_()
    ids = torch.empty((ranks, rows), dtype=torch.int32).share_memory_()
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=args.devices, num_sub_workers=0))
    report = {"passed": False, "ranks": ranks, "cases": []}
    try:
        program = make_sharded_top1(rows).compile(logits, ids, rows, 1, config=config)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        rng = torch.Generator().manual_seed(53)
        with DistributedWorker([program], config=config, persistent=True, reset_persistent_windows=False) as worker:
            resident = worker.alloc_stacked_tensor(logits)
            device_ids = worker.alloc_stacked_tensor(ids)
            for epoch, active in enumerate((6, 5, 6), start=1):
                logits.normal_(generator=rng)
                logits[..., LOCAL_VOCAB:] = 1e10
                logits[0, 0, 15] = 99
                logits[0, 0, 271] = 99
                logits[0, 0, LOCAL_VOCAB - 2] = 99
                logits[-1, 0, 7] = 99
                logits[:, 1, :LOCAL_VOCAB] = -9
                logits[-1, 1, LOCAL_VOCAB - 1] = -1
                if epoch == 3:
                    logits[0, 2, 2] = torch.nan
                    logits[-1, 3, 3] = torch.inf
                    logits[-1, 4, 4] = -torch.inf
                for rank, shard in enumerate(resident.shards):
                    worker.copy_to(shard.data_ptr, logits[rank].data_ptr(), logits[rank].nbytes, worker_id=rank)
                worker.run(program, resident, device_ids, active, epoch, config=config)
                for rank, shard in enumerate(device_ids.shards):
                    worker.copy_from(ids[rank].data_ptr(), shard.data_ptr, ids[rank].nbytes, worker_id=rank)
                valid = logits[:, :active, :LOCAL_VOCAB]
                expected_finite = torch.isfinite(valid).all(-1)
                expected_ids = valid.transpose(0, 1).reshape(active, -1).argmax(-1)
                good = expected_finite.all(0)
                expected_ids[~good] = -1
                torch.testing.assert_close(ids[:, :active], expected_ids.int().expand(ranks, -1))
                if (ids[:, active:] != -1).any():
                    raise AssertionError("inactive rows returned token IDs")
                report["cases"].append({
                    "epoch": epoch, "rows": active, "nonfinite_rows": int((~good).sum()),
                })
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
