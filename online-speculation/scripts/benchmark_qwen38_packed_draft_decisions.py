"""Check Orthrus against OneServing's native packed decisions and time CPU decoding."""
import json
import random
import statistics
import time

import torch

from blockspec.pypto_qwen38.oneserving_tp2 import TP_VOCAB, merge_draft_decisions
from serving.runtime import TensorParallelRunner


baseline = TensorParallelRunner._merge_packed_decisions
torch.set_num_threads(1)
random.seed(743)
generator = torch.Generator().manual_seed(743)


def make_input(rows, valid):
    host = torch.zeros((2, rows, 16), dtype=torch.float32)
    host[:, :, 0] = -1
    for rank in range(2):
        for row in valid:
            host[rank, row, 0] = rank * TP_VOCAB + random.randrange(TP_VOCAB)
            host[rank, row, 1] = torch.randn((), generator=generator).item()
            host[rank, row, 2] = 1
    if valid:
        host[:, valid[0], 1] = 1  # equal scores prefer the smaller TP0 vocabulary ID
    return host, valid, rows


def result(fn, args):
    try:
        return fn(*args)
    except (ValueError, RuntimeError) as error:
        return type(error).__name__, str(error)


def elapsed(fn, args, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(*args)
    return (time.perf_counter_ns() - start) / loops / 1000


checks = 0
for rows in (32, 64):
    for _ in range(64):
        valid = sorted(random.sample(range(rows), random.randrange(rows + 1)))
        args = make_input(rows, valid)
        assert baseline(*args) == merge_draft_decisions(*args)
        checks += 1
    for row, col, value in ((1, 3, 1), (1, 1, float("nan")), (1, 0, 1.5), (1, 2, .5),
                            (1, 2, 2), (1, 0, TP_VOCAB), (3, 0, 23)):
        host, valid, _ = make_input(rows, [1, 2])
        host[0, row, col] = value
        assert result(baseline, (host, valid, rows)) == result(merge_draft_decisions, (host, valid, rows))
        checks += 1
    host, _, _ = make_input(rows, [1, 2])
    for valid in ([1, 1], [rows], [True]):
        assert result(baseline, (host, valid, rows)) == result(merge_draft_decisions, (host, valid, rows))
        checks += 1

cases = []
for rows, valid in [(32, list(range(7))), (64, list(range(7)) + list(range(32, 39))),
                    (64, list(range(31)) + list(range(32, 63)))]:
    args = make_input(rows, valid)
    funcs = {"native": baseline, "bulk": merge_draft_decisions}
    times = {name: [] for name in funcs}
    for fn in funcs.values():
        elapsed(fn, args, 30)
    for run in range(5):
        order = ["native", "bulk"] if run % 2 == 0 else ["bulk", "native"]
        for name in order:
            times[name].append(elapsed(funcs[name], args, 400))
    medians = {name: statistics.median(values) for name, values in times.items()}
    cases.append(dict(rows=rows, valid_rows=len(valid), median_us=medians,
                      samples_us=times, speedup=medians["native"] / medians["bulk"]))
print(json.dumps(dict(torch_version=torch.__version__, cpu_threads=1, parity_checks=checks,
                      benchmark="CPU processing only; excludes device-to-host copy and serving",
                      cases=cases), indent=2))
