#!/usr/bin/env python3
"""真实 LM-head 输入梯度：完整词表、单个有效行、词表末尾及零行隔离。"""

import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--replay-report", type=Path, required=True)
    a = p.parse_args()
    import torch
    import pypto
    from safetensors import safe_open
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.training_programs import make_linear
    from diagnose_qwen38_pypto_reference import metrics

    torch.set_num_threads(4)
    runtime_identity(pypto)
    replay = json.loads(a.replay_report.read_text())
    capture = json.loads(Path(replay["capture"]).read_text())
    loss_node = next(v for v in replay["nodes"] if v["kind"] == "kl")
    with safe_open(replay["intermediates"], framework="pt") as f:
        delta = f.get_tensor(loss_node["outputs"][0])
    with safe_open(
        Path(capture["directory"]) / "shared.safetensors", framework="pt"
    ) as f:
        weight = f.get_tensor("lm_head")
    shape = (delta.shape[0], weight.shape[1])
    valid = capture["blocks"][0]["valid"]
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0),
    )
    meta = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
    program = make_linear(
        tuple(delta.shape), tuple(weight.shape), transpose_b=False
    ).compile(meta(delta.shape), meta(weight.shape), meta(shape), config=config)
    worker = None
    handles = []
    actual = []
    report = dict(passed=False, cases=[], shape=[*delta.shape, weight.shape[1]])
    try:
        worker = DistributedWorker([program], config=config)
        da = worker.alloc_tensor(tuple(delta.shape), torch.bfloat16)
        db = worker.alloc_tensor(tuple(weight.shape), torch.bfloat16, init=weight)
        dc = worker.alloc_tensor(shape, torch.bfloat16)
        handles = [da, db, dc]
        single = torch.zeros_like(delta)
        single[0] = delta[0]
        tail = torch.zeros_like(delta)
        tail[0, -1] = 1
        for name, value, rows in (
            ("real_delta", delta, valid),
            ("single_row", single, 1),
            ("vocab_last_column", tail, 1),
        ):
            worker.copy_to(da.data_ptr, value.data_ptr(), value.numel() * 2)
            start = time.monotonic()
            worker.run(program, da, db, dc, config=config)
            output = torch.empty(shape, dtype=torch.bfloat16)
            worker.copy_from(output.data_ptr(), dc.data_ptr, output.numel() * 2)
            actual.append((name, value, rows, output, time.monotonic() - start))
    finally:
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
    torch.set_num_threads(16)
    wf = weight.float()
    for name, value, rows, output, seconds in actual:
        expected = torch.zeros(shape, dtype=torch.bfloat16)
        expected[:rows] = (value[:rows].float() @ wf).bfloat16()
        item = dict(
            name=name,
            seconds=seconds,
            **metrics(output, expected),
            inactive_nonzero=int(torch.count_nonzero(output[rows:])),
        )
        item["passed"] = item["relative_l2"] < 0.005 and item["inactive_nonzero"] == 0
        report["cases"].append(item)
    report["passed"] = all(v["passed"] for v in report["cases"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("real LM-head VJP failed")


if __name__ == "__main__":
    main()
