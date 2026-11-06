#!/usr/bin/env python3
"""Compare native serving, real drafting and controlled full acceptance."""

import argparse
import json
import logging
from pathlib import Path
import queue
import statistics
import time


class DraftMeter:
    def __init__(self, draft):
        self.draft, self.original = draft, draft.propose
        self.mode, self.reference = "real_head", None
        self.reset()

    def reset(self):
        self.calls = self.candidates = 0
        self.phases = {}

    def __call__(self, request, anchor, width):
        # Oracle 也先完成真实起草，包括参数上传、NPU 计算和 logits 选择。
        result = self.original(request, anchor, width)
        self.calls += 1
        self.candidates += len(result)
        for key, value in self.draft.last_timings.items():
            self.phases[key] = self.phases.get(key, 0.0) + value
        if self.mode != "oracle_keep_real_draft":
            return result
        start = len(request.generated)
        if start < 1 or anchor != self.reference[start - 1]:
            raise AssertionError("oracle anchor differs from the native reference")
        selected = self.reference[start:start + len(result)]
        if len(selected) != len(result):
            raise AssertionError("oracle reference does not cover the proposed block")
        return list(selected)


def idle(engine, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        counts = engine.scheduler.counts()
        if counts["active"] == counts["queued"] == 0:
            return
        time.sleep(.001)
    raise TimeoutError("scheduler did not release the completed request")


def generate(engine, prompt_ids, count, timeout):
    start = time.monotonic()
    request = engine.submit_tokens(prompt_ids, count)
    tokens = []
    try:
        deadline = start + timeout
        while time.monotonic() < deadline:
            try:
                kind, value = request.events.get(timeout=min(1, max(.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if kind == "token":
                tokens.append(value["token_id"])
            elif kind == "done":
                idle(engine)
                if len(tokens) != count:
                    raise AssertionError("request did not reach the fixed output budget")
                return tokens, time.monotonic() - start, request.metrics()
            elif kind in {"error", "cancelled"}:
                raise RuntimeError(f"request {kind}: {value}")
        raise TimeoutError("PyPTO speed measurement timed out")
    finally:
        request.cancelled.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--head-storage", choices=("auto", "resident", "stream"), default="auto")
    parser.add_argument("--head-transfer", choices=("individual", "packed"), default="individual")
    parser.add_argument("--attention-backend", choices=("native", "fused"), default="native")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--prompt", default="Count the integers from 1 to 100:")
    args = parser.parse_args()
    if (args.output.exists() or args.device < 0 or not 64 <= args.tokens <= 480
            or args.repeats < 2 or args.request_timeout <= 0):
        parser.error("require a new output, assigned device, 64..480 tokens and 2+ repeats")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import pypto
    from blockspec.pypto_qwen38 import greedy, target
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource

    report = dict(passed=False, scope="warmed_single_request_e2e", runtime=runtime_identity(pypto),
                  model=str(args.model), tp=1, max_model_len=512, block_size=32, ignore_eos=True,
                  prompt=args.prompt, tokens=args.tokens, head_dir=str(args.head_dir) if args.head_dir else None,
                  cases=[])
    engine = runner = meter = None
    original_correct, original_verify = greedy.correct, target.verify_transaction
    phases = {}

    def timed_correct(*a, **kw):
        start = time.monotonic()
        try:
            return original_correct(*a, **kw)
        finally:
            phases["greedy_correction_seconds"] = phases.get("greedy_correction_seconds", 0) + time.monotonic() - start

    def timed_verify(*a, **kw):
        start = time.monotonic()
        try:
            return original_verify(*a, **kw)
        finally:
            phases["verification_seconds"] = phases.get("verification_seconds", 0) + time.monotonic() - start

    try:
        head = HeadSource(args.model, args.head_dir)
        engine = make_engine(args.source, args.model, args.device, head,
                             head_storage=args.head_storage, head_transfer=args.head_transfer,
                             attention_backend=args.attention_backend)
        # 仅此测速进程忽略 EOS，与 vLLM 定长对照的 SamplingParams 一致。
        engine.scheduler.stop_token_ids = frozenset()
        prompt_ids = engine.tokenizer.encode(args.prompt, add_special_tokens=False)
        if len(prompt_ids) + args.tokens > 512:
            raise ValueError("prompt and output exceed the pinned context")
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic() - start
        runner = engine.runner
        meter = DraftMeter(runner.draft)
        runner.draft.propose = meter
        greedy.correct, target.verify_transaction = timed_correct, timed_verify
        report.update(attention_backend=runner.draft.attention_backend,
                      head_storage=runner.draft.storage, head_transfer=runner.draft.weights.transfer,
                      head_bytes=head.shared_weights.numel(),
                      head_preparation_seconds=head.preparation_seconds)
        for mode in ("target", "real_head", "oracle_keep_real_draft"):
            idle(engine)
            # 同一 worker、相同主模型与缓存配置；Target 跳过 proposer，不执行起草。
            engine.scheduler.runner = runner.native if mode == "target" else runner
            meter.mode, meter.reference = mode, report.get("reference_tokens")
            _, warmup, _ = generate(engine, prompt_ids, args.tokens, args.request_timeout)
            entry = dict(mode=mode, warmup_seconds=warmup, samples=[])
            report["cases"].append(entry)
            for repeat in range(args.repeats):
                before = dict(rounds=runner.rounds, accepted=runner.accepted, restores=runner.restores)
                meter.reset()
                phases.clear()
                tokens, seconds, metrics = generate(engine, prompt_ids, args.tokens, args.request_timeout)
                row = dict(repeat=repeat, seconds=seconds, output_tokens=len(tokens),
                           tokens_per_second=len(tokens) / seconds, metrics=metrics)
                if mode == "target":
                    report.setdefault("reference_tokens", tokens)
                if tokens != report["reference_tokens"]:
                    raise AssertionError(f"{mode} greedy output differs from native reference")
                row["greedy_matches_target"] = True
                if mode != "target":
                    delta = {name: getattr(runner, name) - value for name, value in before.items()}
                    if delta["rounds"] != meter.calls or not meter.candidates:
                        raise AssertionError("every measured draft must reach native verification")
                    row.update(counters=delta, neural_draft_calls=meter.calls,
                               proposed_candidates=meter.candidates,
                               accepted_per_round=delta["accepted"] / delta["rounds"],
                               candidate_acceptance=delta["accepted"] / meter.candidates,
                               draft_phases=dict(meter.phases), verifier_phases=dict(phases))
                    if mode == "oracle_keep_real_draft" and row["candidate_acceptance"] != 1:
                        raise AssertionError("oracle did not reach full acceptance")
                entry["samples"].append(row)
                print(json.dumps(dict(mode=mode, **row)), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            entry["median_tokens_per_second"] = statistics.median(s["tokens_per_second"] for s in entry["samples"])
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        greedy.correct, target.verify_transaction = original_correct, original_verify
        try:
            if runner is not None:
                runner.draft.propose = meter.original if meter is not None else runner.draft.propose
                engine.scheduler.runner = runner
            if engine is not None:
                engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
