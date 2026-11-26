"""Probe direct host output for the same packed greedy kernel used by TP2 draft."""
import argparse
from pathlib import Path
import json
import statistics
import time

import torch
import pypto.language as pl
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig
from serving.model import packed_greedy_fwd

VOCAB = 124160


@pl.jit.host
def packed_host(logits: pl.Tensor, valid: pl.Tensor, output: pl.Out[pl.Tensor]):
    for rank in pl.range(2):
        packed_greedy_fwd(logits[rank], valid[rank], output[rank], rank, device=rank)


def measure(path):
    torch.set_num_threads(1)
    rows = (32, 64)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0))
    outputs = {r: torch.empty((2, r, 16), dtype=torch.float32).share_memory_() for r in rows}
    valid = {r: torch.zeros((2, r), dtype=torch.int32).share_memory_() for r in rows}
    programs = {
        r: packed_host.compile(
            torch.empty((2, r, VOCAB), device="meta", dtype=torch.bfloat16),
            torch.empty((2, r), device="meta", dtype=torch.int32),
            torch.empty((2, r, 16), device="meta", dtype=torch.float32), config=config)
        for r in rows
    }
    out = {}
    with DistributedWorker(list(programs.values()), config=config,
                           inherited_host_tensors=[*outputs.values(), *valid.values()]) as worker:
        for r in rows:
            generator = torch.Generator().manual_seed(r)
            cpu_logits = torch.randn((2, r, VOCAB), generator=generator, dtype=torch.bfloat16)
            logits = worker.alloc_stacked_tensor(cpu_logits)
            direct = outputs[r]
            staged = worker.alloc_stacked_tensor(torch.zeros_like(direct))
            readback = torch.empty_like(direct)

            def dispatch(shared):
                start_ns = time.perf_counter_ns()
                if shared:
                    worker.run(programs[r], logits, valid[r], direct)
                else:
                    worker.run(programs[r], logits, valid[r], staged)
                    for rank, shard in enumerate(staged.shards):
                        buf = readback[rank]
                        worker.copy_from(buf.data_ptr(), shard.data_ptr, buf.nbytes, worker_id=rank)
                return (time.perf_counter_ns() - start_ns) / 1000

            cases = [(), (0,), (0, 1, 30), tuple(range(r))]
            for selected in cases:
                valid[r].zero_()
                valid[r][:, list(selected)] = 1
                direct.fill_(float("nan"))
                dispatch(False)
                dispatch(True)
                torch.testing.assert_close(direct, readback, rtol=0, atol=0)
                torch.testing.assert_close(direct[:, :, 2], valid[r].float(), rtol=0, atol=0)
            for n in range(20):
                dispatch(bool(n % 2))
            samples = {False: [], True: []}
            for n in range(80):
                shared_first = bool(n % 2)
                for shared in (shared_first, not shared_first):
                    samples[shared].append(dispatch(shared))
            out[str(r)] = {
                "bitwise_equal_cases": len(cases),
                "explicit_readback_median_us": statistics.median(samples[False]),
                "shared_output_median_us": statistics.median(samples[True]),
                "explicit_readback_us": samples[False],
                "shared_output_us": samples[True],
            }
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    measure(parser.parse_args().output)
