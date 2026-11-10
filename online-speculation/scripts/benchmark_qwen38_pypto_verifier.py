#!/usr/bin/env python3
"""同一固定 head 对照完整验证与原生 tile 早停；仅使用已定义的 dev 问题。"""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import time

from benchmark_qwen38_pypto import DraftMeter, idle
from benchmark_qwen38_pypto_online import audit_warm_initialization, generate, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "checkpoint-report", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--request-timeout", type=float, default=900)
    a = p.parse_args()
    if a.output.exists() or a.device == a.learner_device or a.repeats < 2 or not 1 <= a.tokens <= 384:
        p.error("new output, two devices, two repeats and valid output budget required")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    from blockspec.pypto_qwen38 import target, greedy
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig

    source = json.loads(a.checkpoint_report.read_text())
    if not source.get("completed") or not source.get("clean_shutdown"):
        p.error("completed checkpoint experiment required")
    options = OnlineConfig(last_layers=source["arguments"]["last_layers"], stride=8, replay_blocks=8,
                           learner_device=a.learner_device, resume_dir=source["checkpoint"]["path"],
                           reset_optimizer=True)
    report = dict(completed=False, split="dev", width=32, physical_block_size=32,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  rows={"ar": [], "full": [], "tiled": []}, calibration=[])
    engine = runner = learner = meter = None
    verify_original, forward_original = target.verify_transaction, target.NativeTarget.forward
    counts = Counter()

    def forward(instance, inputs):
        counts["forward_input_rows"] += len(inputs)
        counts["native_tile_calls"] += (len(inputs)+instance.rows-1)//instance.rows
        start = time.monotonic()
        try:
            return forward_original(instance, inputs)
        finally:
            counts["target_forward_seconds"] += time.monotonic()-start

    def persist():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        temporary.replace(a.output)

    try:
        head = HeadSource(a.model)
        engine = make_engine(a.source, a.model, a.device, head, head_transfer="packed",
                             attention_backend="paged", group_size=4, gdn_backend="vector",
                             online_config=options)
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic()-start
        runner, learner = engine.runner, engine.runner.draft.learner
        if learner.head_identity != source["head_identity"]:
            raise ValueError("checkpoint head/base contract differs")
        report["initialization"] = audit_warm_initialization(learner, options.resume_dir)
        layers, owned = borrow_prefix(runner.native, head, 64-options.last_layers)
        runner.draft.handles.extend(owned)
        runner.draft.weights.publish_many(layers, expected_version=0)
        runner.draft.weights.version = 0
        runner.draft.learner = None
        meter = DraftMeter(runner.draft)
        runner.draft.propose = meter
        target.NativeTarget.forward = forward

        def measure(row, mode):
            idle(engine)
            target.verify_transaction = greedy.verify_tiled if mode == "tiled" else verify_original
            engine.scheduler.runner = runner.native if mode == "ar" else runner
            counts.clear(); meter.reset()
            before = (runner.rounds, runner.accepted)
            result = generate(engine, row, a.tokens, a.request_timeout)
            if learner.updates or learner.optimizer.steps or runner.draft.weights.version:
                raise AssertionError("verifier benchmark changed head/optimizer state")
            result.update(rounds=runner.rounds-before[0], accepted=runner.accepted-before[1],
                          proposed=meter.candidates, verification=dict(counts), draft_phases=dict(meter.phases))
            return result

        calibration = source["dataset"]["splits"]["calibration"][0]
        for mode in ("ar", "full", "tiled"):
            report["calibration"].append(dict(mode=mode, row=measure(calibration, mode)))
        for repeat in range(a.repeats):
            for question in source["dataset"]["splits"]["dev"]:
                row = {**question, "id": question["id"]+f"#repeat{repeat}"}
                for mode in (("ar", "full", "tiled") if repeat%2 == 0 else ("ar", "tiled", "full")):
                    report["rows"][mode].append(measure(row, mode))
                if not (report["rows"]["ar"][-1]["tokens"] == report["rows"]["full"][-1]["tokens"] ==
                        report["rows"]["tiled"][-1]["tokens"]):
                    raise AssertionError("verifier mode changed the delivered AR output")
                persist()
        report["summaries"] = {mode: summarize(rows, target=report["rows"]["ar"])
                               for mode, rows in report["rows"].items()}
        report["verification_totals"] = {mode: dict(sum((Counter(row["verification"]) for row in rows), Counter()))
                                         for mode, rows in report["rows"].items()}
        report["tiled_over_full_tps"] = (report["summaries"]["tiled"]["tokens_per_second"] /
                                           report["summaries"]["full"]["tokens_per_second"])
        report.update(completed=True, snapshot=engine.snapshot())
        print(json.dumps(dict(event="finished", summaries=report["summaries"],
                              verification=report["verification_totals"])), flush=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        target.verify_transaction, target.NativeTarget.forward = verify_original, forward_original
        if runner is not None:
            runner.draft.learner = learner
            engine.scheduler.runner = runner
            if meter is not None: runner.draft.propose = meter.original
        try:
            if engine is not None: engine.stop()
            report["clean_shutdown"] = True
        except BaseException as error:
            report.update(completed=False, cleanup_error=repr(error))
            raise
        finally:
            persist()


if __name__ == "__main__":
    main()
