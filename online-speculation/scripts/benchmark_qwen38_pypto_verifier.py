#!/usr/bin/env python3
"""固定 head 对照验证器或块宽度；仅用 dev 选择，用新留出题确认。"""

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import time

from benchmark_qwen38_pypto import DraftMeter, idle
from benchmark_qwen38_pypto_online import audit_warm_initialization, generate, summarize, choose_width, prepare_questions


def refinement_questions(source, candidate, count):
    old = source["dataset"]
    used = old["splits"]["test"]+old["splits"].get("confirm", [])
    if (candidate["files_sha256"] != old["files_sha256"] or
            candidate["splits"]["test"][:len(used)] != used or
            candidate["splits"]["dev"] != old["splits"]["dev"]):
        raise ValueError("refinement changed data or the previous held-out partition")
    new = candidate["splits"]["test"][len(used):]
    seen = {row["question_sha256"] for rows in old["splits"].values() for row in rows}
    if (len(new) != count or len({row["question_sha256"] for row in new}) != count or
            any(row["question_sha256"] in seen for row in new)):
        raise ValueError("fresh distinct confirmation questions required")
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "checkpoint-report", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--widths", default="", help="固定 head 的小窗口对照，如 2,3,4；不重新训练")
    p.add_argument("--data", type=Path, help="小窗口选择后，从数据集中保留新的独立确认题")
    p.add_argument("--confirmation-count", type=int, default=24)
    a = p.parse_args()
    widths = sorted(set(map(int, filter(None, a.widths.split(",")))))
    if a.output.exists() or a.device == a.learner_device or a.repeats < 2 or not 1 <= a.tokens <= 384:
        p.error("new output, two devices, two repeats and valid output budget required")
    if widths and (min(widths) < 2 or max(widths) > 32 or a.data is None or a.confirmation_count < 1):
        p.error("width refinement requires widths in 2..32, data and a fresh confirmation budget")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    from blockspec.pypto_qwen38 import target, greedy
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig

    source = json.loads(a.checkpoint_report.read_text())
    if not source.get("completed") or not source.get("clean_shutdown"):
        p.error("completed checkpoint experiment required")
    if widths and not source.get("confirmation"):
        p.error("width refinement requires the completed previous confirmation record")
    options = OnlineConfig(last_layers=source["arguments"]["last_layers"], stride=8, replay_blocks=8,
                           learner_device=a.learner_device, resume_dir=source["checkpoint"]["path"],
                           reset_optimizer=True)
    modes = [f"width{width}" for width in widths] if widths else ["full", "tiled"]
    report = dict(completed=False, split="dev", physical_block_size=32,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  rows={mode: [] for mode in ("ar", *modes)}, calibration=[])
    report["checkpoint_source_sha256"] = hashlib.sha256(a.checkpoint_report.read_bytes()).hexdigest()
    if not widths: report["width"] = 32
    engine = runner = learner = meter = None
    start_attempted = False
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
        if widths:
            used = source["dataset"]["splits"]["test"]+source["dataset"]["splits"].get("confirm", [])
            candidate = prepare_questions(a.data, engine.tokenizer, seed=source["dataset"]["seed"],
                max_prompt=source["arguments"]["max_prompt"],
                train_count=source["arguments"]["train_count"]+
                    len(source["dataset"].get("excluded_prewarm_record_ids", [])),
                dev_count=source["arguments"]["dev_count"], test_count=len(used)+a.confirmation_count)
            report["new_confirmation_questions"] = refinement_questions(source, candidate, a.confirmation_count)
            report["previous_confirmation_ids"] = [row["id"] for row in used]
            report["confirmation_data_sha256"] = candidate["files_sha256"]
        start = time.monotonic()
        start_attempted = True
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
            target.verify_transaction = greedy.verify_tiled if mode == "tiled" or mode.startswith("width") else verify_original
            if mode.startswith("width"):
                runner.draft.block_size = int(mode[5:])
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
        for mode in ("ar", *modes):
            report["calibration"].append(dict(mode=mode, row=measure(calibration, mode)))
        for repeat in range(a.repeats):
            for question in source["dataset"]["splits"]["dev"]:
                row = {**question, "id": question["id"]+f"#repeat{repeat}"}
                for mode in ("ar", *(modes if repeat%2 == 0 else reversed(modes))):
                    report["rows"][mode].append(measure(row, mode))
                if any(report["rows"][mode][-1]["tokens"] != report["rows"]["ar"][-1]["tokens"] for mode in modes):
                    raise AssertionError("verifier mode changed the delivered AR output")
                persist()
        report["summaries"] = {mode: summarize(rows, target=report["rows"]["ar"])
                               for mode, rows in report["rows"].items()}
        report["verification_totals"] = {mode: dict(sum((Counter(row["verification"]) for row in rows), Counter()))
                                         for mode, rows in report["rows"].items()}
        if widths:
            report["width_sweep"] = [dict(width=width, split="dev", summary=report["summaries"][f"width{width}"])
                                     for width in widths]
            selected = report["selected_width"] = choose_width(report["width_sweep"])
            rows, refs = [], []
            report["confirmation"] = dict(width=selected, rows=rows, target_rows=refs)
            persist()
            for row in report["new_confirmation_questions"]:
                refs.append(measure(row, "ar"))
                rows.append(measure(row, f"width{selected}"))
                if rows[-1]["tokens"] != refs[-1]["tokens"]:
                    raise AssertionError("fresh confirmation output differs from AR")
                persist()
                print(json.dumps(dict(event="confirmation", width=selected, completed=len(rows),
                                      total=len(report["new_confirmation_questions"]))), flush=True)
            report["confirmation"]["summary"] = summarize(rows, target=refs)
        else:
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
            if engine is not None and start_attempted: engine.stop()
            report["clean_shutdown"] = True
        except BaseException as error:
            report.update(completed=False, cleanup_error=repr(error))
            raise
        finally:
            persist()


if __name__ == "__main__":
    main()
