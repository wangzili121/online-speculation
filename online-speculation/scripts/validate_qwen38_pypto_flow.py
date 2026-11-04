#!/usr/bin/env python3
"""Test native PyPTO verification/rollback using explicit oracle/rejection fixtures.

No neural draft runs here. These checks isolate the serving transaction from
the still-separate draft computation and must not be used as acceptance-rate
or acceleration evidence.
"""

import argparse
import json
from pathlib import Path
import queue
import time


class FixtureDraft:
    block_size, context = 32, 512

    def __init__(self, reference, reject):
        self.reference, self.reject = reference, reject

    def propose(self, request, anchor, width):
        start = len(request.generated)
        if anchor != self.reference[start - 1]:
            raise AssertionError("fixture anchor differs from target-only continuation")
        tokens = list(self.reference[start:start + width - 1])
        if self.reject is not None and self.reject < len(tokens):
            tokens[self.reject] = (tokens[self.reject] + 1) % 248320
        return tokens


def collect(engine, count, timeout):
    request = engine.submit_text("Hello", count)
    result = []
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                kind, value = request.events.get(timeout=min(1, max(.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if kind == "token":
                result.append(value["token_id"])
            elif kind == "done":
                return result
            elif kind in {"error", "cancelled"}:
                raise RuntimeError(f"request {kind}: {value}")
        raise TimeoutError("PyPTO request did not finish")
    finally:
        # Covers validation timeout without leaving a live request in the scheduler.
        request.cancelled.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.tokens <= 64 or args.device < 0 or args.request_timeout <= 0:
        parser.error("require a new output, nonnegative device, 2..64 tokens and positive timeout")
    from blockspec.pypto_qwen38.baseline import checked_source, native_module, runtime_identity
    from blockspec.pypto_qwen38.target import OrthrusRunner
    checked_source(args.source)
    import pypto
    identity = runtime_identity(pypto)
    engine_module = native_module(args.source, "engine")
    slots = native_module(args.source, "schedule").BatchSlot
    engine = engine_module.InferenceEngine(args.model, args.device, max_active=1, prefix_cache=False)
    report = dict(runtime=identity, scope="native_target_fixture_transactions", neural_draft=False,
                  trained_head_loaded=False, passed=False, cases=[])
    native = engine.runner
    try:
        engine.start()
        reference = collect(engine, args.tokens, args.request_timeout)
        if len(reference) != args.tokens:
            raise AssertionError("target-only request stopped before reaching the requested fixture length")
        report["baseline_tokens"] = reference
        for reject in (None, 0, 16, 30):
            # Wait for native request release after the scheduler's final done event.
            deadline = time.monotonic() + 30
            while engine.scheduler.counts()["active"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("previous request was not released")
                time.sleep(.01)
            adapter = OrthrusRunner(native, FixtureDraft(reference, reject), slots)
            engine.scheduler.runner = adapter
            start = time.monotonic()
            tokens = collect(engine, args.tokens, args.request_timeout)
            passed = tokens == reference
            report["cases"].append(dict(fixture="oracle" if reject is None else f"reject-{reject}",
                                        tokens=tokens, passed=passed, rounds=adapter.rounds,
                                        accepted=adapter.accepted, restores=adapter.restores,
                                        seconds=time.monotonic() - start))
            if not passed:
                raise AssertionError(f"PyPTO transaction changed target-only tokens: rejection={reject}")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        # The test reuses a live native worker; stop it through the real scheduler.
        engine.scheduler.runner = native
        engine.stop()
        report["native_requests_released"] = not bool(native.requests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
