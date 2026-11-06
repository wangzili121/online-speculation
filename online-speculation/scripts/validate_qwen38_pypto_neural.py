#!/usr/bin/env python3
"""Compare the real PyPTO neural draft service path with its native target."""

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import time

from validate_qwen38_pypto_flow import collect


def resident_checksum(native):
    """逐段读取目标缓存，避免为了校验一次分配完整的主机副本。"""
    import torch
    result = []
    for handle in native.resident:
        digest = hashlib.sha256()
        size = prod(handle.shape) * torch.empty((), dtype=handle.dtype).element_size()
        for offset in range(0, size, 32 << 20):
            chunk = torch.empty(min(32 << 20, size - offset), dtype=torch.uint8)
            native.worker.copy_from(chunk.data_ptr(), handle.data_ptr, chunk.numel(), src_offset=offset)
            digest.update(memoryview(chunk.numpy()))
        result.append(digest.hexdigest())
    return result


def history_checksum(native, request):
    """只比较已提交历史；页内被拒绝的尾部字节允许保留，但不得再次可见。"""
    import torch
    resident = native.requests[request.request_id]
    digest = hashlib.sha256()
    for value in native._read_state(resident.slot):
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    length = resident.pages.length
    for pool in native.resident[2:]:
        layers, heads, blocks, rows, dim = pool.shape
        for layer in range(layers):
            for head in range(heads):
                for logical, physical in enumerate(resident.pages.pages):
                    count = min(rows, length - logical * rows)
                    if count <= 0:
                        break
                    value = torch.empty((count, dim), dtype=pool.dtype)
                    offset = ((layer * heads + head) * blocks + physical) * rows * dim * value.element_size()
                    native._copy_offset(value, pool, offset, False)
                    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def audit_round(engine, source, width, audit):
    """独立审计一个真实 K32 起草轮，避免接受率低时反复起草影响联调耗时。"""
    from blockspec.pypto_qwen38.baseline import native_module
    from blockspec.pypto_qwen38.greedy import verify_transaction
    from blockspec.pypto_qwen38.target import NativeTarget
    schedule = native_module(source, "schedule")
    native, draft = engine.runner.native, engine.runner.draft
    request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False),
                                    width + 4, frozenset())
    native.admit(request)
    try:
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position:position + 16])
            anchor = native.run_batch([schedule.BatchSlot(request, tokens, position, "prefill",
                                     position + len(tokens) == len(request.prompt_ids))])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        initial = target.snapshot()
        expected, token = [], anchor
        for _ in range(width + 3):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(initial)
        before = resident_checksum(native)
        pages = (native.requests[request.request_id].pages.length,
                 tuple(native.requests[request.request_id].pages.pages))
        candidates = draft.propose(request, anchor, width)
        audit.update(candidates=candidates, draft_executed=True, draft_seconds=draft.last_seconds)
        unchanged = before == resident_checksum(native) and pages == (
            native.requests[request.request_id].pages.length, tuple(native.requests[request.request_id].pages.pages))
        audit["target_cache_unchanged"] = unchanged
        if not unchanged:
            raise AssertionError("draft modified target cache bytes or page ownership")
        result = verify_transaction(target, anchor, candidates, budget=width)
        audit.update(emitted=list(result.tokens), accepted=result.accepted,
                     committed_inputs=result.committed_inputs)
        if list(result.tokens) != expected[:len(result.tokens)]:
            raise AssertionError("neural block verification changed the native greedy continuation")
        actual_history = history_checksum(native, request)
        # 从同一快照、按同样的验证分组重建接受前缀，检查 KV 与 GDN 提交结果。
        target.restore(initial)
        target.forward((anchor, *candidates)[:result.committed_inputs])
        audit["committed_history_matches_replay"] = actual_history == history_checksum(native, request)
        if not audit["committed_history_matches_replay"]:
            raise AssertionError("committed KV/GDN differs from replaying only the accepted inputs")
        token = result.tokens[-1]
        continuation = []
        for _ in range(3):
            token = target.forward([token])[0]
            continuation.append(token)
        audit["continuation_matches"] = continuation == expected[len(result.tokens):len(result.tokens) + 3]
        if not audit["continuation_matches"]:
            raise AssertionError("AR continuation after block correction differs from target-only")
        audit["passed"] = True
    finally:
        native.release(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--audit-block", action="store_true", help="audit a full K32 round and cache bytes")
    parser.add_argument("--attention-backend", choices=("native", "fused"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.device < 0 or not 3 <= args.tokens <= 64 or args.request_timeout <= 0:
        parser.error("require a new output, nonnegative device, 3..64 tokens and positive timeout")
    import pypto
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource

    report = dict(passed=False, scope="neural_draft_greedy_continuation")
    engine = runner = None
    try:
        report["runtime"] = runtime_identity(pypto)
        head = HeadSource(args.model, args.head_dir)
        report.update(head_source="trained", head_tp=head.tp, block_size=head.block_size)
        report["attention_backend"] = args.attention_backend
        engine = make_engine(args.source, args.model, args.device, head, attention_backend=args.attention_backend)
        runner = engine.runner
        engine.start()
        if args.audit_block:
            report["block_audit"] = dict(width=head.block_size, passed=False)
            audit_round(engine, args.source, head.block_size, report["block_audit"])
        # 同一 worker、同一份主模型参数；先关闭 proposer，取得目标模型的真实续写。
        engine.scheduler.runner = runner.native
        baseline = collect(engine, args.tokens, args.request_timeout)
        deadline = time.monotonic() + 30
        while engine.scheduler.counts()["active"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("baseline request was not released")
            time.sleep(.01)
        engine.scheduler.runner = runner
        actual = collect(engine, args.tokens, args.request_timeout)
        report.update(baseline=baseline, actual=actual, rounds=runner.rounds,
                      accepted=runner.accepted, restores=runner.restores,
                      neural_draft_rounds=runner.draft.rounds, block_size=head.block_size,
                      last_draft_seconds=runner.draft.last_seconds)
        if len(baseline) != args.tokens or baseline != actual or runner.draft.rounds < 1:
            raise AssertionError("neural draft continuation did not match target-only output")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if engine is not None:
                engine.scheduler.runner = runner
                engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            report["native_requests_released"] = None if runner is None else not bool(runner.native.requests)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
