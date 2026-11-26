#!/usr/bin/env python3
"""Compare resident Verify snapshots with the host path on one A3 device."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    from blockspec.dsv41_packed_loss import PackedLoss, PackedTeacherCopy, make_packed_loss_graph, make_packed_teacher_copy

    configure_logging("warn")
    torch.set_num_threads(4)
    rows, vocab = MAX_LOGIT_ROWS, FLASH.vocab_size
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    snapshot = make_packed_teacher_copy(rows, vocab, 1, 1).compile(config=config)
    loss_graph = make_packed_loss_graph(rows, vocab, 1, 1, "tv").compile(config=config)
    rng = torch.Generator().manual_seed(5517)
    batches = [torch.randn(rows, vocab, generator=rng) for _ in range(2)]
    ranges = [(5, 2), (1, 5)]
    expected = torch.cat([value[first:first + count] for value, (first, count) in zip(batches, ranges, strict=True)])
    with DistributedWorker([snapshot, loss_graph], config, persistent=True) as runtime:
        value = torch.randn(rows, vocab, generator=rng)
        student_buffer = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=0)
        scratch = runtime.alloc_tensor(value.shape, value.dtype, worker_id=0)
        student = SimpleNamespace(owner=0, position=127, rows=len(expected), logits=student_buffer)
        learner = PackedLoss(SimpleNamespace(runtime=runtime, config=config, tp=1, world=1,
                          ranks=[{"logits": student_buffer}]), loss_graph)
        recorder = PackedTeacherCopy(learner, snapshot, [scratch])
        observed = torch.empty_like(expected)
        gradient = torch.empty((rows, vocab), dtype=torch.bfloat16)
        evidence = {}
        for mode in ("host", "resident"):
            durations, capture_times, loss_times = [], [], []
            for _ in range(6):
                feedback = recorder.capture(student) if mode == "resident" else None
                chunks = []
                position = student.position
                elapsed = 0.
                try:
                    for batch, (first, count) in zip(batches, ranges, strict=True):
                        runtime.copy_to(scratch.data_ptr, batch.data_ptr(), batch.nbytes, worker_id=0)
                        start = perf_counter()
                        if feedback is not None:
                            feedback.append_from(position, first, count)
                        else:
                            chunk = torch.empty(count, vocab)
                            runtime.copy_from(chunk.data_ptr(), scratch.data_ptr, chunk.nbytes,
                                              src_offset=first * chunk[0].nbytes, worker_id=0)
                            chunks.append(chunk)
                        elapsed += perf_counter() - start
                        position += count
                    start = perf_counter()
                    teacher = feedback.logits() if feedback is not None else torch.cat(chunks)
                    result = learner(student, teacher)
                    loss_seconds = perf_counter() - start
                    capture_times.append(elapsed)
                    loss_times.append(loss_seconds)
                    durations.append(elapsed + loss_seconds)
                    if feedback is not None:
                        runtime.copy_from(observed.data_ptr(), teacher.buffer.data_ptr, observed.nbytes, worker_id=0)
                        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                finally:
                    if feedback is not None:
                        feedback.close()
            runtime.copy_from(gradient.data_ptr(), result.gradient.data_ptr, gradient.nbytes, worker_id=0)
            if mode == "host":
                reference_loss, reference_gradient = result.value, gradient.clone()
            else:
                assert reference_loss == result.value
                torch.testing.assert_close(gradient, reference_gradient, atol=0, rtol=0)
            evidence[mode] = dict(warm_capture_and_loss_median_ms=1000 * median(durations[1:]),
                                  warm_capture_median_ms=1000 * median(capture_times[1:]),
                                  warm_loss_median_ms=1000 * median(loss_times[1:]),
                                  teacher_upload_bytes=result.teacher_upload_bytes)

        for bad in (float("nan"), float("inf"), -float("inf")):
            batches[0][0, 123] = bad
            runtime.copy_to(scratch.data_ptr, batches[0].data_ptr(), batches[0].nbytes, worker_id=0)
            feedback = recorder.capture(student)
            try:
                feedback.append_from(student.position, 0, 1)
                assert feedback.logits() is None
                try:
                    learner(student, feedback.logits(finished=True))
                except ValueError as error:
                    assert "finite" in str(error)
                else:
                    raise AssertionError("Non-finite teacher was accepted")
            finally:
                feedback.close()
    report = dict(passed=True, rows=len(expected), vocabulary=vocab, snapshot_rows=ranges,
                  loss_and_gradient_exact=True, nonfinite_values_rejected=3,
                  host_full_logit_roundtrip_bytes=2 * expected.nbytes, **evidence)
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
