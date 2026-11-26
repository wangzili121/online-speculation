#!/usr/bin/env python3
"""Compare resident Verify snapshots with the host path on one A3 device."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one-row-waves", action="store_true", help="Collect seven teacher rows over seven Verify calls")
    args = parser.parse_args()

    import torch
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    from blockspec import dsv41_packed_loss
    from blockspec.dsv41_packed_loss import PackedLoss, PackedTeacherCopy, make_packed_loss_graph, make_packed_teacher_copy

    def elementwise_check(teacher):
        if not torch.isfinite(teacher).all():
            raise ValueError("Packed teacher logits must be finite")

    configure_logging("warn")
    torch.set_num_threads(4)
    rows, vocab = MAX_LOGIT_ROWS, FLASH.vocab_size
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    snapshot = make_packed_teacher_copy(rows, vocab, 1, 1).compile(config=config)
    loss_graph = make_packed_loss_graph(rows, vocab, 1, 1, "tv").compile(config=config)
    rng = torch.Generator().manual_seed(5517)
    ranges = [(index, 1) for index in range(7)] if args.one_row_waves else [(5, 2), (1, 5)]
    batches = [torch.randn(rows, vocab, generator=rng) for _ in ranges]
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
        modes = ("host_elementwise", "host", "resident")
        measurements = {mode: [] for mode in modes}
        uploads = {}
        for iteration in range(9):
            for mode in modes:
                feedback = recorder.capture(student) if mode == "resident" else None
                chunks = []
                position = student.position
                elapsed = 0.
                check_seconds = None
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
                    check = patch.object(dsv41_packed_loss, "_require_finite_host", elementwise_check)
                    with check if mode == "host_elementwise" else nullcontext():
                        start = perf_counter()
                        teacher = feedback.logits() if feedback is not None else torch.cat(chunks)
                        result = learner(student, teacher)
                        loss_seconds = perf_counter() - start
                        if feedback is None:
                            start = perf_counter()
                            dsv41_packed_loss._require_finite_host(teacher)
                            check_seconds = perf_counter() - start
                    measurements[mode].append((elapsed, loss_seconds, check_seconds))
                    if feedback is not None:
                        runtime.copy_from(observed.data_ptr(), teacher.buffer.data_ptr, observed.nbytes, worker_id=0)
                        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                finally:
                    if feedback is not None:
                        feedback.close()
                if iteration == 8:
                    uploads[mode] = result.teacher_upload_bytes
                    runtime.copy_from(gradient.data_ptr(), result.gradient.data_ptr, gradient.nbytes, worker_id=0)
                    if mode == "host_elementwise":
                        reference_loss, reference_gradient = result.value, gradient.clone()
                    else:
                        assert reference_loss == result.value
                        torch.testing.assert_close(gradient, reference_gradient, atol=0, rtol=0)

        evidence = {}
        for mode, samples in measurements.items():
            warm = samples[1:]
            evidence[mode] = dict(warm_capture_and_loss_median_ms=1000 * median(a + b for a, b, _ in warm),
                                  warm_capture_median_ms=1000 * median(a for a, _, _ in warm),
                                  warm_loss_median_ms=1000 * median(b for _, b, _ in warm),
                                  teacher_upload_bytes=uploads[mode])
            if mode != "resident":
                evidence[mode]["warm_finite_check_median_ms"] = 1000 * median(c for _, _, c in warm)

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
