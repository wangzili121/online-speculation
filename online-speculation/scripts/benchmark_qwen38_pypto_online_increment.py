#!/usr/bin/env python3
"""Compare a frozen warm head and on-request updates from the same checkpoint."""

import argparse
import json
import logging
from pathlib import Path
from random import Random
import time

from benchmark_qwen38_pypto import DraftMeter, idle
from benchmark_qwen38_pypto_online import generate, prepare_questions, summarize


def select_questions(report, candidate, offset, count, tokens, seed):
    if candidate["files_sha256"] != report["dataset"]["files_sha256"]:
        raise ValueError("checkpoint benchmark dataset changed")
    splits = report["dataset"]["splits"]
    questions = candidate["splits"]["test"][offset:offset + count]
    used = {
        row["question_sha256"]
        for name in ("train", "dev", "calibration", "test", "confirm")
        for row in splits.get(name, [])
    }
    used.update(row["question_sha256"] for row in report["evaluation_test"])
    hashes = {row["question_sha256"] for row in questions}
    if len(questions) != count or len(hashes) != count or hashes & used:
        raise ValueError("paired requests must be distinct and disjoint from this checkpoint's recorded data")
    if any(len(row["prompt_ids"]) + tokens > 512 for row in questions):
        raise ValueError("prompt and output budget exceed the compiled context")
    Random(seed).shuffle(questions)
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "data", "checkpoint-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--learner-device", required=True, type=int)
    parser.add_argument("--question-offset", type=int, default=80)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--replay-blocks", type=int, default=1)
    parser.add_argument("--verification-rows", type=int, choices=range(1, 17), default=3)
    args = parser.parse_args()
    if (args.output.exists() or args.device == args.learner_device
            or args.question_offset < 0 or min(args.requests, args.tokens) < 1):
        parser.error("require a fresh report, distinct devices and valid request budgets")
    source = json.loads(args.checkpoint_report.read_text())
    if not source["completed"] or not source["clean_shutdown"]:
        parser.error("require a completed checkpoint report")
    checkpoint = Path(source["checkpoint"]["path"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    initial_step = source["updates"][-1]["step"]
    if manifest["step"] != initial_step:
        parser.error("checkpoint step differs from the report")

    logging.getLogger("simpler").setLevel(logging.WARNING)
    from pypto.runtime import configure_log
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig

    configure_log("warn")
    options = OnlineConfig(last_layers=source["arguments"]["last_layers"],
                           learner_device=args.learner_device, resume_dir=checkpoint,
                           objective="tv", learning_rate=1e-5,
                           stride=args.stride, replay_blocks=args.replay_blocks)
    report = {
        "completed": False, "initial_step": initial_step, "width": source["arguments"]["width"],
        "checkpoint_sha256": manifest["payload_sha256"],
        "arguments": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
        "rows": {name: [] for name in ("ar", "frozen", "online")},
        "updates": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    head = HeadSource(args.model)
    engine = make_engine(args.source, args.model, args.device, head, head_transfer="packed",
                         attention_backend="paged", group_size=4, gdn_backend="vector",
                         online_config=options, verification_backend="tiled",
                         verification_rows=args.verification_rows, partial_teacher_capture=True)
    runner = engine.runner
    learner = runner.native.learner
    try:
        candidate = prepare_questions(args.data, engine.tokenizer, seed=source["dataset"]["seed"],
            max_prompt=source["arguments"].get("max_prompt", 128),
            train_count=1, dev_count=1, test_count=args.question_offset + args.requests)
        questions = select_questions(source, candidate, args.question_offset, args.requests, args.tokens, args.seed)
        report["request_order"] = [row["id"] for row in questions]
        persist()
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic() - start
        if learner.head_identity != source["head_identity"] or learner.updates != initial_step:
            raise RuntimeError("runner restored a different head or training step")
        draft = runner.draft
        layers, owned = borrow_prefix(runner.native, head, 64 - options.last_layers)
        draft.handles.extend(owned)
        draft.weights.publish_many(layers, expected_version=initial_step)
        draft.weights.version = initial_step
        draft.block_size = report["width"]
        meter = DraftMeter(draft)
        draft.propose = meter
        original_update = learner.update

        def update():
            start = time.monotonic()
            result = original_update()
            report["updates"].append({"step": learner.updates, "seconds": time.monotonic() - start, **result})
            return result

        learner.update = update

        def measure(question, mode):
            idle(engine)
            engine.scheduler.runner = runner.native if mode == "ar" else runner
            draft.learner = learner if mode == "online" else None
            rounds, accepted = runner.rounds, runner.accepted
            step, version = learner.updates, draft.weights.version
            meter.reset()
            row = generate(engine, question, args.tokens, 900)
            if mode != "online" and (learner.updates != step or draft.weights.version != version):
                raise AssertionError("read-only request updated the head")
            if mode == "online" and learner.last_result and learner.last_result.get("skipped"):
                raise FloatingPointError(f"online update skipped: {learner.last_result}")
            return row | {"rounds": runner.rounds - rounds, "accepted": runner.accepted - accepted,
                          "proposed": meter.candidates, "updates": learner.updates - step,
                          "head_version_before": version, "head_version_after": draft.weights.version}

        calibration = source["dataset"]["splits"]["calibration"][0]
        measure(calibration, "ar")
        measure(calibration, "frozen")
        for mode in ("frozen", "online"):
            for index, question in enumerate(questions):
                row = measure(question, mode)
                if mode == "frozen":
                    report["rows"]["ar"].append(measure(question, "ar"))
                if row["tokens"] != report["rows"]["ar"][index]["tokens"]:
                    raise AssertionError("greedy speculative output changed")
                report["rows"][mode].append(row)
                persist()
                print(json.dumps({"mode": mode, "completed_requests": index + 1,
                                  "updates": learner.updates - initial_step}), flush=True)
        if learner.updates == initial_step or learner.updates != draft.weights.version:
            raise AssertionError("online stream must publish parameter updates")
        rows = report["rows"]
        report["summaries"] = {mode: summarize(values, target=rows["ar"]) for mode, values in rows.items()}
        report["online_vs_frozen"] = (report["summaries"]["online"]["tokens_per_second"] /
                                     report["summaries"]["frozen"]["tokens_per_second"])
        report["completed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        runner.draft.learner = learner
        engine.scheduler.runner = runner
        try:
            engine.stop()
            report["clean_shutdown"] = True
        finally:
            persist()


if __name__ == "__main__":
    main()
