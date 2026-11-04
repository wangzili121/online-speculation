#!/usr/bin/env python3
"""Compare the real PyPTO neural draft service path with its native target."""

import argparse
import json
from pathlib import Path
import time

from validate_qwen38_pypto_flow import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.device < 0 or not 3 <= args.tokens <= 64 or args.request_timeout <= 0:
        parser.error("require a new output, nonnegative device, 3..64 tokens and positive timeout")
    import pypto
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource

    report = dict(runtime=runtime_identity(pypto), passed=False, scope="neural_draft_greedy_continuation")
    head = HeadSource(args.model, args.head_dir)
    engine = make_engine(args.source, args.model, args.device, head)
    runner = engine.runner
    try:
        engine.start()
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
        engine.scheduler.runner = runner
        engine.stop()
        report["native_requests_released"] = not bool(runner.native.requests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
