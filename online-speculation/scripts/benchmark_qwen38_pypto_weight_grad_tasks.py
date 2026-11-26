"""Compare PyPTO training dW task grouping on real Qwen3.8 projection shapes."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig
from blockspec.pypto_qwen38.training_graph import SuffixShape
from blockspec.pypto_qwen38.training_programs import make_linear


def run(out, groups):
    torch.set_num_threads(2)
    torch.manual_seed(211)
    shape = SuffixShape()
    width = shape.q_heads * shape.dimension
    heads = {"full_q": (2 * width, shape.hidden),
             "full_kv": (shape.kv_heads * shape.dimension, shape.hidden),
             "full_o": (shape.hidden, width)}
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    meta = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="meta")
    report = {}
    for name, (m, n) in heads.items():
        rows = shape.block
        args = (meta((rows, m), torch.bfloat16), meta((rows, n), torch.bfloat16),
                meta((m, n), torch.float32))
        programs = {g: make_linear((rows, m), (rows, n), transpose_a=True, transpose_b=False,
                                   gradient_weight=True, task_group=g).compile(*args, config=config)
                    for g in groups}
        with DistributedWorker(list(programs.values()), config=config) as worker:
            x, y = (torch.randint(-8, 8, (rows, k), dtype=torch.int32).bfloat16() / 16
                    for k in (m, n))
            x[8:] = 0
            dx, dy = (worker.alloc_tensor(z.shape, z.dtype, init=z) for z in (x, y))
            dw = worker.alloc_tensor((m, n), torch.float32)
            result = torch.empty((m, n), dtype=torch.float32)
            reference = (x.float().T @ y.float()).bfloat16().float()

            def dispatch(group):
                before = time.perf_counter_ns()
                worker.run(programs[group], dx, dy, dw)
                return (time.perf_counter_ns() - before) / 1000

            for group in groups:
                dispatch(group)
                worker.copy_from(result.data_ptr(), dw.data_ptr, result.nbytes)
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
            for _ in range(4):
                for group in groups:
                    dispatch(group)
            samples = {g: [] for g in groups}
            for step in range(15):
                cycle = groups[step % len(groups):] + groups[:step % len(groups)]
                for group in cycle:
                    samples[group].append(dispatch(group))
            report[name] = {"shape": [m, rows, n], "bitwise_equal_to_torch": list(groups),
                            "median_us": {g: statistics.median(samples[g]) for g in groups},
                            "samples_us": samples}
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", nargs="+", type=int, default=(16, 8, 4))
    args = parser.parse_args()
    run(args.output, tuple(args.groups))
