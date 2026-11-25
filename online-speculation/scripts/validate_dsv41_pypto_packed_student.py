#!/usr/bin/env python3
"""Verify that the packed draft output can be retained across resident graph calls."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error("Use two distinct devices")

    import torch
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from blockspec.dsv41_packed_proposer import PackedDraftProposer, PackedStudentRows

    @pl.jit
    def emit_rows(source: pl.Tensor[[16, 128], pl.FP32], logits: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            result = pl.load(source, [0, 0], [16, 128])
            pl.store(result, [0, 0], logits)
        return logits

    @pl.jit.host
    def emit(source: pl.Tensor[[2, 16, 128], pl.FP32], logits: pl.Out[pl.Tensor[[2, 16, 128], pl.FP32]]):
        for rank in pl.range(pld.world_size()):
            emit_rows(source[rank], logits[rank], device=rank)

    config = RunConfig(
        backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    graph = emit.compile(config=config)
    source_param, logits_param = graph._get_metadata()[0]
    row_values = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128)
    with DistributedWorker(graph, config, persistent=True, reset_persistent_windows=False) as runtime:
        proposer = PackedDraftProposer.__new__(PackedDraftProposer)
        proposer.world = 2
        proposer.runtime = runtime
        proposer.ranks = [
            dict(source=runtime.alloc_tensor(row_values.shape, row_values.dtype, init=row_values + rank, worker_id=rank),
                 logits=runtime.alloc_tensor(row_values.shape, row_values.dtype, worker_id=rank))
            for rank in range(2)
        ]
        retained = runtime.alloc_tensor(row_values.shape, row_values.dtype, worker_id=1)
        proposer.student = {"request": PackedStudentRows(1, 471, 7, retained)}
        inputs = proposer._rank_argument("source", source_param)
        regular = proposer._rank_argument("logits", logits_param)
        capture = proposer._rank_argument("logits", logits_param, replacement=(1, retained))
        runtime.run(graph, inputs, capture, config=config)
        changed = row_values + 10
        runtime.copy_to(proposer.ranks[1]["source"].data_ptr, changed.data_ptr(), changed.nbytes, worker_id=1)
        runtime.run(graph, inputs, regular, config=config)
        observed = torch.empty_like(row_values)
        for rank, buffer, expected in (
            (0, proposer.ranks[0]["logits"], row_values),
            (1, proposer.ranks[1]["logits"], changed),
            (1, retained, row_values + 1),
        ):
            runtime.copy_from(observed.data_ptr(), buffer.data_ptr, observed.nbytes, worker_id=rank)
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        proposer.release_student("request")
        assert not proposer.student
        runtime.run(graph, inputs, regular, config=config)
    print(json.dumps(dict(passed=True, devices=devices, retained_rows=16, columns=128, exact=True,
                          released=True, next_dispatch_passed=True)), flush=True)


if __name__ == "__main__":
    main()
