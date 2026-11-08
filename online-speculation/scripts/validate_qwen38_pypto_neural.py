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


def audit_verifier(target, anchor, expected, snapshot):
    """用目标续写构造候选，覆盖全接受/中途拒绝；不作为 draft 接受率或性能结果。"""
    from blockspec.pypto_qwen38.greedy import verify_transaction

    candidates = list(expected[:31])
    cases = [("all_accepted", candidates, 32, (), 31)]
    for rejected in (0, 15, 16, 30):
        changed = candidates.copy()
        changed[rejected] = (changed[rejected] + 1) % 248320
        cases.append((f"reject_at_{rejected}", changed, 32, (), rejected))
    for budget in (1, 16):
        cases.append((f"budget_{budget}", candidates, budget, (), budget))
    stop = expected[16]
    cases.append(("stop_token", candidates, 32, (stop,), expected.index(stop) + 1))
    report = []
    try:
        for name, proposed, budget, stops, accepted in cases:
            target.restore(snapshot)
            result = verify_transaction(target, anchor, proposed, budget=budget, stop_tokens=stops)
            if result.accepted != accepted or list(result.tokens) != expected[:len(result.tokens)]:
                raise AssertionError(f"{name}: incorrect target correction or accepted count")
            actual = history_checksum(target.runner, target.request)
            target.restore(snapshot)
            target.forward((anchor, *proposed)[:result.committed_inputs])
            if actual != history_checksum(target.runner, target.request):
                raise AssertionError(f"{name}: committed KV/GDN differs from accepted-prefix replay")
            token, continuation = result.tokens[-1], []
            for _ in range(3):
                token = target.forward([token])[0]
                continuation.append(token)
            if continuation != expected[len(result.tokens):len(result.tokens) + 3]:
                raise AssertionError(f"{name}: continuation differs after correction")
            row = dict(case=name, accepted=result.accepted, committed_inputs=result.committed_inputs,
                       history_matches_replay=True, continuation_matches=True, passed=True)
            report.append(row)
            print(json.dumps(row), flush=True)
    finally:
        target.restore(snapshot)
    return report


def audit_round(engine, source, width, audit, *, prompt_ids=None):
    """独立审计一个真实 K32 起草轮，避免接受率低时反复起草影响联调耗时。"""
    from blockspec.pypto_qwen38.baseline import native_module
    from blockspec.pypto_qwen38.greedy import verify_transaction
    from blockspec.pypto_qwen38.target import NativeTarget
    schedule = native_module(source, "schedule")
    native, draft = engine.runner.native, engine.runner.draft
    request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False) if prompt_ids is None else list(prompt_ids),
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
        audit.update(candidates=candidates, draft_executed=True, draft_seconds=draft.last_seconds,
                     draft_timings=dict(draft.last_timings))
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


def audit_slot_reuse(engine, source):
    """相同上下文放到不同状态槽/物理页，候选和 logits 应相同。"""
    from blockspec.pypto_qwen38.baseline import native_module
    from compare_qwen38_pypto_draft import tensor_digest
    schedule = native_module(source, "schedule")
    native, draft = engine.runner.native, engine.runner.draft
    active = []
    held_pages = None

    def prepare(text):
        request = schedule.RequestState(engine.tokenizer.encode(text, add_special_tokens=False), 40, frozenset())
        native.admit(request)
        active.append(request)
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position:position + 16])
            anchor = native.run_batch([schedule.BatchSlot(request, tokens, position, "prefill",
                                       position + len(tokens) == len(request.prompt_ids))])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        return request, anchor

    def propose(request, anchor):
        before = resident_checksum(native)
        result = dict(slot=native.requests[request.request_id].slot,
                      pages=list(native.requests[request.request_id].pages.pages),
                      candidates=draft.propose(request, anchor, 32),
                      logits_sha256=tensor_digest(draft.host_logits[:31]))
        if before != resident_checksum(native):
            raise AssertionError("draft modified target cache during slot check")
        return result

    try:
        first, anchor = prepare("Hello")
        expected = propose(first, anchor)
        native.release(first)
        active.remove(first)
        # 原生 prefill 会自动 compact 到 slot 0，不能靠另挂一个请求强占槽位。
        # 保留一页改变物理页分配，再在 prefill 后显式迁移本请求的 GDN 状态。
        held_pages = native.cache.new_request()
        native.cache.reserve(held_pages, 1)
        second, second_anchor = prepare("Hello")
        resident = native.requests[second.request_id]
        old_slot, new_slot = resident.slot, (resident.slot + 1) % len(native.slot_ids)
        native._write_state(new_slot, *native._read_state(old_slot))
        native._write_state(old_slot, native.zero_conv, native.zero_recurrent)
        native.slot_ids[old_slot] = None
        native.slot_ids[new_slot] = second.request_id
        resident.slot = new_slot
        actual = propose(second, second_anchor)
        if (anchor != second_anchor or expected["slot"] == actual["slot"]
                or expected["pages"] == actual["pages"]
                or any(expected[k] != actual[k] for k in ("candidates", "logits_sha256"))):
            raise AssertionError("draft depends on the physical slot or page allocation")
        return dict(passed=True, first_slot=expected["slot"], second_slot=actual["slot"],
                    first_pages=expected["pages"], second_pages=actual["pages"], state_relocated=True,
                    target_cache_unchanged=True, logits_byte_exact=True)
    finally:
        for request in reversed(active):
            native.release(request)
        if held_pages is not None:
            native.cache.release(held_pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--audit-block", action="store_true", help="audit a full K32 round and cache bytes")
    parser.add_argument("--attention-backend", choices=("native", "fused", "paged"), default="native")
    parser.add_argument("--draft-group-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--gdn-backend", choices=("native", "vector"), default="native")
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
        report["gdn_backend"] = args.gdn_backend
        engine = make_engine(args.source, args.model, args.device, head, attention_backend=args.attention_backend,
                             group_size=args.draft_group_size, gdn_backend=args.gdn_backend,
                             head_transfer="packed" if args.draft_group_size == 4 else "individual")
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
