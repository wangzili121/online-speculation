#!/usr/bin/env python3
"""将固定批次训练的同一份 master 接回服务，对照训练 anchor、训练题和留出题。"""

import argparse
import hashlib
import json
from pathlib import Path
import time


ORDER = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_masters(path, graph):
    import torch
    from safetensors.torch import load_file

    values = load_file(str(path))
    if set(values) != set(ORDER):
        raise ValueError("exactly six last-1 FP32 master tensors required")
    for name, value in values.items():
        if (
            value.dtype != torch.float32
            or tuple(value.shape) != graph.inputs[name].shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"invalid master tensor: {name}")
    return values


def publish_snapshot(learner, values):
    """仅恢复并发布执行参数，不构造或恢复不存在的 Adam 状态。"""
    import torch
    from diagnose_qwen38_pypto_training import read_tensor

    before = (learner.updates, learner.optimizer.steps)
    for name, value in values.items():
        learner.trainer.copy_to(
            learner.optimizer.state[name][0].data_ptr,
            value.data_ptr(),
            value.numel() * value.element_size(),
        )
    learner.optimizer.render(0, learner.optimizer.state)
    execution = [learner.optimizer.execution[name][0] for name in ORDER]
    learner.publisher.publish(63, execution, expected_version=learner.publisher.version)
    actual_banks = learner.draft.weights.overrides[63]
    checks = {}
    for name, bank in zip(ORDER, actual_banks, strict=True):
        expected = values[name].bfloat16()
        if "gamma" in name:
            expected = expected.float() + 1
        actual = read_tensor(learner.native.worker, bank)
        checks[name] = bool(torch.equal(actual, expected))
    if not all(checks.values()) or before != (learner.updates, learner.optimizer.steps):
        raise AssertionError(
            "snapshot publication changed training state or execution values"
        )
    return checks


def inspect_anchors(engine, learner, manifest, records, directory, state):
    """沿采集时相同的分段前缀推进；同时对照融合推理与训练采集前向。"""
    from safetensors.torch import load_file
    from blockspec.pypto_qwen38.baseline import native_module
    from blockspec.pypto_qwen38.target import NativeTarget
    from blockspec.pypto_qwen38.greedy import verify_transaction
    from diagnose_qwen38_pypto_training import read_tensor
    from diagnose_qwen38_pypto_reference import metrics
    from validate_qwen38_pypto_neural import history_checksum

    runner, draft = engine.runner, engine.runner.draft
    native = runner.native
    schedule = native_module(engine.runner.draft.source, "schedule")
    grouped = {}
    for index, sample in enumerate(manifest["samples"]):
        grouped.setdefault(sample["record_id"], []).append((index, sample))
    rows = []
    for record_id, samples in grouped.items():
        tokens = records[record_id]["tokens"]
        first = samples[0][1]["anchor"]
        request = schedule.RequestState(list(tokens[:first]), 1, frozenset())
        native.admit(request)
        try:
            for start in range(0, first, 16):
                end = min(first, start + 16)
                native.run_batch(
                    [
                        schedule.BatchSlot(
                            request,
                            tuple(tokens[start:end]),
                            start,
                            "prefill",
                            end == first,
                        )
                    ]
                )
            target = NativeTarget(native, request, schedule.BatchSlot)
            for index, sample in samples:
                anchor, width = sample["anchor"], sample["width"]
                past = target.resident.pages.length
                if past > anchor:
                    raise AssertionError(
                        "anchor replay advanced past the recorded prefix"
                    )
                if past < anchor:
                    target.forward(tokens[past:anchor])
                request.prompt_ids, request.generated = (
                    list(tokens[:anchor]),
                    [tokens[anchor]],
                )
                snapshot = target.snapshot()
                try:
                    history = history_checksum(native, request)
                    draft.learner = None
                    candidates = draft.propose(request, tokens[anchor], width)
                    serving_logits = draft.read_logits(width - 1).clone()
                    # 再走采集路径，检查冻结前缀输入与保存的训练快照是否一致。
                    draft.learner = learner
                    learner.rounds = 0
                    draft.propose(request, tokens[anchor], width)
                    captured = draft.read_logits(width - 1).clone()
                    block = learner.current
                    saved = load_file(str(directory / f"block-{index:02d}.safetensors"))
                    hidden = read_tensor(native.worker, block.inputs["hidden"])
                    input_error = metrics(hidden[:width], saved["hidden"][:width])
                    forward_error = metrics(serving_logits, captured)
                    if (
                        input_error["relative_l2"] >= 0.02
                        or forward_error["relative_l2"] >= 0.02
                    ):
                        raise AssertionError(
                            (
                                "training/serving prefix differs",
                                input_error,
                                forward_error,
                            )
                        )
                    checked = NativeTarget(
                        native, request, schedule.BatchSlot, collector=block
                    )
                    checked.forward(tokens[anchor : anchor + width])
                    teacher = read_tensor(native.worker, block.inputs["teacher"])[
                        : width - 1
                    ]
                    teacher_error = metrics(teacher, saved["teacher"][: width - 1])
                    teacher_argmax = teacher.argmax(-1).tolist()
                    stored_argmax = saved["teacher"][: width - 1].argmax(-1).tolist()
                    learner.abort()
                    draft.learner = None
                    target.restore(snapshot)
                    if history_checksum(native, request) != history:
                        raise AssertionError(
                            "draft or teacher rollback changed committed history"
                        )
                    expected, current = [], tokens[anchor]
                    for _ in range(width):
                        current = target.forward([current])[0]
                        expected.append(current)
                    target.restore(snapshot)
                    result = verify_transaction(
                        target, tokens[anchor], candidates, budget=width
                    )
                    if list(result.tokens) != expected[: len(result.tokens)]:
                        raise AssertionError(
                            "real verifier output differs from sequential AR"
                        )
                    row = dict(
                        state=state,
                        record_id=record_id,
                        anchor=anchor,
                        width=width,
                        role="answer"
                        if anchor >= records[record_id]["prompt_length"]
                        else "prompt",
                        candidates=candidates,
                        accepted=result.accepted,
                        real_ar_tokens=expected,
                        greedy_equal=True,
                        teacher_argmax_equal_snapshot=teacher_argmax == stored_argmax,
                        teacher_record_aligned=teacher_argmax
                        == list(tokens[anchor + 1 : anchor + width]),
                        input_error=input_error,
                        serving_capture_error=forward_error,
                        teacher_error=teacher_error,
                    )
                    rows.append(row)
                    print(json.dumps(dict(event="trained_anchor", **row)), flush=True)
                finally:
                    learner.abort()
                    draft.learner = None
                    target.restore(snapshot)
                    del snapshot
        finally:
            native.release(request)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "data", "capture-report", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int, required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--test-count", type=int, default=8)
    args = p.parse_args()
    if (
        args.output.exists()
        or args.device == args.learner_device
        or not 8 <= args.tokens <= 64
    ):
        p.error("new output, two devices and an 8..64 token budget required")
    import torch
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph
    from benchmark_qwen38_pypto import DraftMeter, idle
    from benchmark_qwen38_pypto_cold import response_records
    from benchmark_qwen38_pypto_online import (
        generate,
        prepare_questions,
        publish_initial_suffix,
        summarize,
    )

    torch.set_num_threads(4)
    manifest = json.loads(args.capture_report.read_text())
    if manifest["last_layers"] != 1 or not manifest.get("capture_ready"):
        raise ValueError("an immutable last-1 feedback manifest required")
    directory = Path(manifest["directory"])
    source = json.loads(Path(manifest["arguments"]["records_report"]).read_text())
    paths = {
        name: directory / f"{name}.master.safetensors" for name in ("initial", "final")
    }
    masters = {
        name: load_masters(path, LastLayerGraph(objective="kl"))
        for name, path in paths.items()
    }
    report = dict(
        completed=False,
        updates_performed=0,
        master_sha256={k: checksum(v) for k, v in paths.items()},
        manifest_sha256=checksum(args.capture_report),
        states={},
        source_arguments=manifest["arguments"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    engine = learner = None
    try:
        head = HeadSource(args.model)
        engine = make_engine(
            args.source,
            args.model,
            args.device,
            head,
            head_transfer="packed",
            attention_backend="paged",
            group_size=4,
            gdn_backend="vector",
            online_config=OnlineConfig(
                objective="kl",
                stride=8,
                replay_blocks=8,
                learner_device=args.learner_device,
            ),
        )
        dataset = prepare_questions(
            args.data,
            engine.tokenizer,
            seed=source["dataset"]["seed"],
            max_prompt=source["arguments"]["max_prompt"],
            train_count=source["arguments"]["train_count"],
            dev_count=source["arguments"]["dev_count"],
            test_count=args.test_count,
        )
        clean, _ = response_records(source, dataset)
        prompt_lengths = {
            row["id"]: len(row["prompt_ids"]) for row in dataset["splits"]["train"]
        }
        records = {
            row["id"]: dict(**row, prompt_length=prompt_lengths[row["id"]])
            for row in clean
        }
        trained_ids = {row["record_id"] for row in manifest["samples"]}
        groups = {
            "train": [
                row for row in dataset["splits"]["train"] if row["id"] in trained_ids
            ],
            "dev": dataset["splits"]["dev"],
            "test": dataset["splits"]["test"],
        }
        report["dataset"] = dataset
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic() - start
        runner, draft, learner = (
            engine.runner,
            engine.runner.draft,
            engine.runner.draft.learner,
        )
        if learner.head_identity != manifest["head_identity"]:
            raise ValueError("snapshot was trained from another base/head")
        publish_initial_suffix(learner)
        layers, owned = borrow_prefix(runner.native, head, 63)
        draft.handles.extend(owned)
        draft.weights.publish_many(layers, expected_version=0)
        draft.weights.version = 0
        draft.block_size = manifest["blocks"][0]["width"]
        draft.learner = None
        meter = DraftMeter(draft)
        draft.propose = meter

        def measure(row, target=False):
            idle(engine)
            engine.scheduler.runner = runner.native if target else runner
            meter.reset()
            before = runner.rounds, runner.accepted
            value = generate(engine, row, args.tokens, 1200)
            value.update(
                rounds=runner.rounds - before[0],
                accepted=runner.accepted - before[1],
                proposed=meter.candidates,
                draft_phases=dict(meter.phases),
            )
            if (
                learner.updates
                or learner.optimizer.steps
                or learner.current
                or learner.replay
            ):
                raise AssertionError(
                    "fixed-head evaluation unexpectedly trained the model"
                )
            return value

        # 先在同一 worker 上得到固定 AR 对照；之后两份 head 使用完全相同的问题。
        baseline = {
            split: [measure(row, True) for row in rows]
            for split, rows in groups.items()
        }
        report["target"] = baseline
        persist()
        for state in ("initial", "final"):
            idle(engine)
            checked = publish_snapshot(learner, masters[state])
            entry = dict(publication_exact=checked, version=draft.weights.version)
            report["states"][state] = entry
            draft.propose = meter.original
            try:
                entry["anchors"] = inspect_anchors(
                    engine, learner, manifest, records, directory, state
                )
            finally:
                draft.propose = meter
            persist()
            for split, rows in groups.items():
                measured = [measure(row) for row in rows]
                entry[split] = dict(
                    rows=measured, summary=summarize(measured, target=baseline[split])
                )
                if not entry[split]["summary"]["greedy_equal_all"]:
                    raise AssertionError("fixed head changed the target output")
                print(
                    json.dumps(
                        dict(
                            event="fixed_head_evaluation",
                            state=state,
                            split=split,
                            **entry[split]["summary"],
                        )
                    ),
                    flush=True,
                )
                persist()
        answers = [
            row
            for row in report["states"]["final"]["anchors"]
            if row["role"] == "answer"
        ]
        report["trained_answer_anchors_all_accepted"] = all(
            row["accepted"] == row["width"] - 1 for row in answers
        )
        if not answers or not report["trained_answer_anchors_all_accepted"]:
            raise AssertionError(
                "trained answer anchors failed real verifier; inspect parity before generalization"
            )
        if report["master_sha256"] != {k: checksum(v) for k, v in paths.items()}:
            raise AssertionError("evaluation modified the source master snapshots")
        report.update(completed=True, snapshot=engine.snapshot())
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if engine is not None:
                engine.scheduler.runner = engine.runner
                if learner is not None:
                    engine.runner.draft.learner = learner
                engine.stop()
            report["clean_shutdown"] = True
        finally:
            persist()


if __name__ == "__main__":
    main()
