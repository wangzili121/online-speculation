#!/usr/bin/env python3
"""真实 27B、同一 worker/head/history，对比主机与设备收尾及事务回滚。"""

import argparse
import json
import logging
from pathlib import Path
import statistics
import time

from compare_qwen38_pypto_draft import tensor_digest
from validate_qwen38_pypto_neural import audit_round, audit_slot_reuse, audit_verifier, history_checksum, resident_checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4)
    a = p.parse_args()
    if a.output_dir.exists() or a.rounds < 2:
        p.error("new output directory and at least two rounds required")
    a.output_dir.mkdir(parents=True)
    import torch
    from blockspec.pypto_qwen38.baseline import native_module
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.target import NativeTarget
    torch.set_num_threads(4)
    logging.getLogger("simpler").setLevel(logging.WARNING)
    report = dict(passed=False, scope="same_worker_host_vs_device_serving", rounds=[])
    engine = native = request = serving = None
    try:
        head = HeadSource(a.model, a.head_dir)
        engine = make_engine(a.source, a.model, a.device, head, attention_backend="paged", gdn_backend="vector",
                             group_size=4, head_transfer="packed")
        start = time.perf_counter()
        engine.start()
        print("real 27B model ready", flush=True)
        native, draft = engine.runner.native, engine.runner.draft
        serving = native.device_serving
        report["startup_seconds"] = time.perf_counter() - start
        schedule = native_module(a.source, "schedule")
        # 主机基线建立同一份已提交历史，设备侧随后读取它。
        native.device_serving = None
        request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False), 40, frozenset())
        native.admit(request)
        anchor = native.run_batch([schedule.BatchSlot(request, tuple(request.prompt_ids), 0, "prefill", True)])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        initial = target.snapshot()
        expected, token = [], anchor
        for _ in range(35):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(initial)
        before = resident_checksum(native)
        signatures = {}
        for iteration in range(a.rounds + 1):
            for backend in (("host", "device") if iteration % 2 == 0 else ("device", "host")):
                native.device_serving = serving if backend == "device" else None
                start = time.perf_counter()
                tokens = draft.propose(request, anchor, 32)
                elapsed = time.perf_counter() - start
                signature = dict(tokens=tokens, logits_sha256=tensor_digest(draft.read_logits(31)))
                if backend in signatures and signature != signatures[backend]:
                    raise AssertionError("repeated identical input changed logits/tokens")
                signatures[backend] = signature
                if iteration:
                    report["rounds"].append(dict(backend=backend, seconds=elapsed, phases=dict(draft.last_timings)))
                    print(backend, iteration, elapsed, flush=True)
        assert signatures["host"] == signatures["device"], "device input preparation/selection changed draft output"
        assert before == resident_checksum(native), "draft modified target caches"
        report.update(signatures=signatures, draft_cache_byte_exact=True)
        # 同一初态下，单行/16+16 验证分组均不能改变目标 token。
        target_results = {}
        for backend in ("host", "device"):
            native.device_serving = serving if backend == "device" else None
            target.restore(initial)
            start = time.perf_counter()
            predictions = target.forward((anchor, *signatures["device"]["tokens"]))
            elapsed = time.perf_counter() - start
            target_results[backend] = dict(tokens=predictions, history_sha256=history_checksum(native, request),
                                           forward_seconds=elapsed)
        assert target_results["host"]["tokens"] == target_results["device"]["tokens"]
        assert target_results["host"]["history_sha256"] == target_results["device"]["history_sha256"]
        report["verifier"] = target_results
        native.device_serving = serving
        target.restore(initial)
        from blockspec.pypto_qwen38.greedy import verify_transaction
        report["transaction_timings"] = []
        for iteration in range(a.rounds + 1):
            for backend in (("host", "device") if iteration % 2 == 0 else ("device", "host")):
                native.device_serving = serving if backend == "device" else None
                for case in ("all_accepted", "reject_at_0"):
                    proposed = expected[:31].copy()
                    if case == "reject_at_0":
                        proposed[0] = (proposed[0] + 1) % 248320
                    target.restore(initial)
                    start = time.perf_counter()
                    result = verify_transaction(target, anchor, proposed, budget=32)
                    elapsed = time.perf_counter() - start
                    assert list(result.tokens) == expected[:len(result.tokens)]
                    if iteration:
                        report["transaction_timings"].append(dict(backend=backend, case=case, seconds=elapsed))
                        print("transaction", backend, case, elapsed, flush=True)
        native.device_serving = serving
        # 这里的初态仍是主机快照；内部事务使用设备快照，覆盖嵌套生命周期。
        report["transactions"] = audit_verifier(target, anchor, expected, initial)
        native.release(request)
        request = None
        report["slot_reuse"] = audit_slot_reuse(engine, a.source)
        report["boundaries"] = []
        for past in (15, 17, 96, 97, 224, 225, 476):
            audit = dict(past=past)
            audit_round(engine, a.source, 32, audit, prompt_ids=[353] * past)
            report["boundaries"].append(audit)
        report["transaction_medians_seconds"] = {
            case: {backend: statistics.median(row["seconds"] for row in report["transaction_timings"]
                                              if row["backend"] == backend and row["case"] == case)
                   for backend in ("host", "device")} for case in ("all_accepted", "reject_at_0")}
        report["median_seconds"] = {backend: statistics.median(r["seconds"] for r in report["rounds"] if r["backend"] == backend)
                                    for backend in ("host", "device")}
        report["median_phases"] = {backend: {key: statistics.median(r["phases"][key] for r in report["rounds"] if r["backend"] == backend)
                                            for key in report["rounds"][0]["phases"]}
                                   for backend in ("host", "device")}
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if native is not None:
            native.device_serving = serving
            if request is not None:
                native.release(request)
        if engine is not None:
            engine.stop()
        (a.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
