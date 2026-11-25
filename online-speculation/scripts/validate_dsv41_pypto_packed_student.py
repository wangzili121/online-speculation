#!/usr/bin/env python3
"""Verify that the packed draft output can be retained across resident graph calls."""

import argparse
import json
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--loss", choices=("tv", "forward_kl", "both"))
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    world = len(devices)
    if world not in (2, 4) or len(set(devices)) != world:
        parser.error("Use two or four distinct devices")
    second_owner = world // 2

    import torch
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from blockspec.dsv41_packed_loss import PackedLoss, make_packed_loss_graph
    from blockspec.dsv41_packed_proposer import PackedDraftProposer, PackedStudentRows

    kinds = ("tv", "forward_kl") if args.loss == "both" else (args.loss,) if args.loss else ()
    vocabulary, tile = (129280, 1280) if kinds else (128, 128)

    @pl.jit
    def emit_rows(source: pl.Tensor[[16, vocabulary], pl.FP32],
                  logits: pl.Out[pl.Tensor[[16, vocabulary], pl.FP32]]):
        for column in pl.spmd(vocabulary // tile):
            logits[:, column * tile:(column + 1) * tile] = pl.slice(source, [16, tile], [0, column * tile])
        return logits

    @pl.jit.host
    def emit(source: pl.Tensor[[world, 16, vocabulary], pl.FP32], logits: pl.Out[pl.Tensor[[world, 16, vocabulary], pl.FP32]]):
        for rank in pl.range(pld.world_size()):
            emit_rows(source[rank], logits[rank], device=rank)

    config = RunConfig(
        backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    graph = emit.compile(config=config)
    loss_graphs = {
        kind: make_packed_loss_graph(16, vocabulary, 2, second_owner, kind).compile(config=config)
        for kind in kinds
    }
    source_param, logits_param = graph._get_metadata()[0]
    generator = torch.Generator().manual_seed(5719)
    row_values = torch.randn(16, vocabulary, generator=generator)
    reports = {}
    with DistributedWorker([graph, *loss_graphs.values()], config, persistent=True,
                           reset_persistent_windows=False) as runtime:
        proposer = PackedDraftProposer.__new__(PackedDraftProposer)
        proposer.world, proposer.tp = world, second_owner
        proposer.runtime, proposer.config = runtime, config
        proposer.ranks = [
            dict(source=runtime.alloc_tensor(row_values.shape, row_values.dtype, init=row_values + rank, worker_id=rank),
                 logits=runtime.alloc_tensor(row_values.shape, row_values.dtype, worker_id=rank))
            for rank in range(world)
        ]
        retained = runtime.alloc_tensor(row_values.shape, row_values.dtype, worker_id=second_owner)
        proposer.student = {"request": PackedStudentRows(second_owner, 471, 7, retained)}
        inputs = proposer._rank_argument("source", source_param)
        regular = proposer._rank_argument("logits", logits_param)
        capture = proposer._rank_argument("logits", logits_param, replacement={second_owner: retained})
        runtime.run(graph, inputs, capture, config=config)
        changed = row_values + 10
        runtime.copy_to(proposer.ranks[second_owner]["source"].data_ptr, changed.data_ptr(),
                        changed.nbytes, worker_id=second_owner)
        runtime.run(graph, inputs, regular, config=config)
        observed = torch.empty_like(row_values)
        for rank, buffer, expected in (
            (0, proposer.ranks[0]["logits"], row_values),
            (second_owner, proposer.ranks[second_owner]["logits"], changed),
            (second_owner, retained, row_values + second_owner),
        ):
            runtime.copy_from(observed.data_ptr(), buffer.data_ptr, observed.nbytes, worker_id=rank)
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        captures = {
            0: PackedStudentRows(0, 471, 1, proposer.ranks[0]["logits"]),
            second_owner: proposer.student["request"],
        }
        for kind, compiled in loss_graphs.items():
            operation = PackedLoss(proposer, compiled)
            reports[kind] = []
            for owner, student in captures.items():
                logits = (row_values + owner)[:student.rows]
                teacher = logits + torch.randn(logits.shape, generator=generator) * 0.22
                samples = [operation(student, teacher) for _ in range(6)]
                result = samples[-1]
                actual = torch.empty((16, vocabulary), dtype=torch.bfloat16)
                runtime.copy_from(actual.data_ptr(), result.gradient.data_ptr, actual.nbytes, worker_id=owner)
                assert not torch.any(actual[student.rows:])
                golden = logits.double().requires_grad_()
                log_student, log_teacher = golden.log_softmax(-1), teacher.double().log_softmax(-1)
                p, q = log_student.exp(), log_teacher.exp()
                expected = ((p - q).abs().sum(-1) * .5 if kind == "tv" else
                            (q * (log_teacher - log_student)).sum(-1)).mean()
                expected.backward()
                difference = actual[:student.rows].double() - golden.grad
                abs_error = abs(result.value - float(expected.detach()))
                relative_l2 = float(difference.norm() / golden.grad.norm())
                assert abs_error <= 2e-5 and relative_l2 <= 0.005
                reports[kind].append(dict(owner=owner, rows=student.rows, loss=result.value,
                                          max_abs_loss_error=abs_error, relative_l2=relative_l2,
                                          median_ms=median(sample.seconds for sample in samples[1:]) * 1000,
                                          teacher_upload_bytes=result.teacher_upload_bytes))
        proposer.release_student("request")
        assert not proposer.student
        runtime.run(graph, inputs, regular, config=config)
    print(json.dumps(dict(passed=True, devices=devices, tp=second_owner, retained_rows=16, columns=vocabulary, exact=True,
                          released=True, next_dispatch_passed=True, loss=reports)), flush=True)


if __name__ == "__main__":
    main()
