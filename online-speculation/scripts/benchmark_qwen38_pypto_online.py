#!/usr/bin/env python3
"""用真实问题比较 AR 复制初始化、last-N 在线学习和固定 head 推理。"""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import queue
import random
import time

from benchmark_qwen38_pypto import DraftMeter, idle


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def prepare_questions(directory, tokenizer, *, seed=42, max_prompt=128, train_count=64,
                      dev_count=3, test_count=6):
    """只取问题；数据集的答案绝不进入 prompt 或在线监督。dev 与训练也按题目隔离。"""
    selected, seen, files = {}, set(), {}
    provenance = directory / "provenance.json"
    expected = json.loads(provenance.read_text()).get("files_sha256", {}) if provenance.exists() else {}
    for split in ("train", "test"):
        path = directory / f"{split}.jsonl"
        payload = path.read_bytes()
        files[split] = hashlib.sha256(payload).hexdigest()
        if split in expected and files[split] != expected[split]:
            raise ValueError(f"{split} data SHA256 differs from its provenance")
        rows = list(enumerate(json.loads(line) for line in payload.decode().splitlines() if line.strip()))
        random.Random(seed).shuffle(rows)
        candidates = []
        needed = train_count + dev_count + 1 if split == "train" else test_count
        for index, row in rows:
            question = row["question"].strip()
            key = digest(question)
            if key in seen:
                continue
            ids = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            if not 1 <= len(ids) <= max_prompt:
                continue
            seen.add(key)
            candidates.append(dict(id=f"{split}:{index}", question_sha256=key, prompt_ids=list(ids)))
            if len(candidates) == needed:
                break
        if len(candidates) < needed:
            raise ValueError(f"not enough disjoint {split} questions under the prompt budget")
        if split == "train":
            selected.update(calibration=candidates[:1], dev=candidates[1:1+dev_count],
                            train=candidates[1+dev_count:])
        else:
            selected["test"] = candidates
    return dict(seed=seed, files_sha256=files, splits=selected, tokenized_sha256=digest(selected))


def summarize(rows, *, target=None):
    """按总 token / 总耗时汇总；不对逐请求 TPS 做算术平均。"""
    seconds = sum(row["seconds"] for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    rounds = sum(row.get("rounds", 0) for row in rows)
    accepted = sum(row.get("accepted", 0) for row in rows)
    proposed = sum(row.get("proposed", 0) for row in rows)
    result = dict(requests=len(rows), output_tokens=tokens, seconds=seconds,
                  tokens_per_second=tokens / seconds if seconds else None,
                  rounds=rounds, accepted=accepted, proposed=proposed,
                  accepted_per_round=accepted / rounds if rounds else None,
                  candidate_acceptance=accepted / proposed if proposed else None)
    if target is not None:
        if [row["id"] for row in rows] != [row["id"] for row in target]:
            raise ValueError("speed comparison requires paired questions in the same order")
        equality = [row["tokens"] == ref["tokens"] for row, ref in zip(rows, target)]
        baseline = summarize(target)
        result.update(greedy_equal_requests=sum(equality), greedy_equal_all=all(equality),
                      target_tokens_per_second=baseline["tokens_per_second"],
                      measured_tps_ratio=result["tokens_per_second"] / baseline["tokens_per_second"],
                      lossless_speedup=(sum(row["seconds"] for row in target) / seconds
                                        if all(equality) else None))
    return result


def warm_dataset(previous, candidate, *, train_count, confirm_count):
    """预热用过的问题不再在线训练；开发集、测试集不变，另留一次性确认集。"""
    if not previous.get("completed") or not previous.get("clean_shutdown"):
        raise ValueError("warm start requires a completed, cleanly stopped prewarm run")
    old = previous["dataset"]
    if old["files_sha256"] != candidate["files_sha256"]:
        raise ValueError("warm start data SHA256 changed")
    for name in ("calibration", "dev"):
        if old["splits"][name] != candidate["splits"][name]:
            raise ValueError("warm start changed held-out prompts")
    if old["splits"]["test"] != candidate["splits"]["test"][:len(old["splits"]["test"])]:
        raise ValueError("warm start changed held-out test prompts")
    excluded = {row["id"] for row in previous["records"]}
    prompts = {row["id"]: row for row in old["splits"]["train"]}
    if not excluded <= prompts.keys():
        raise ValueError("warm records must belong to their declared training partition")
    used_hashes = {prompts[key]["question_sha256"] for key in excluded}
    train = [row for row in candidate["splits"]["train"]
             if row["id"] not in excluded and row["question_sha256"] not in used_hashes][:train_count]
    confirm = candidate["splits"]["test"][len(old["splits"]["test"]):][:confirm_count]
    if len(train) != train_count or len(confirm) != confirm_count:
        raise ValueError("not enough fresh questions for online training/confirmation")
    splits = {name: old["splits"][name] for name in ("calibration", "dev", "test")}
    splits.update(train=train, confirm=confirm)
    all_rows = [row for rows in splits.values() for row in rows]
    if len({row["question_sha256"] for row in all_rows}) != len(all_rows):
        raise ValueError("online/held-out/confirmation partitions overlap")
    return {**candidate, "splits": splits, "tokenized_sha256": digest(splits),
            "excluded_prewarm_record_ids": sorted(excluded)}


def audit_warm_initialization(learner, checkpoint):
    """从设备读回 master 与 Adam 动量，确认加载权重而非延续旧优化器。"""
    import torch
    from safetensors.torch import load_file
    from diagnose_qwen38_pypto_training import read_tensor
    values = load_file(str(Path(checkpoint)/"optimizer.safetensors"), device="cpu")
    optimizer = learner.optimizer
    if learner.options.last_layers == 1:
        states = {name+"."+label: handle for name, handles in optimizer.state.items()
                  for label, handle in zip(("master", "first", "second"), handles, strict=True)}
    else:
        states = dict(zip(("master", "first", "second"), optimizer.state, strict=True))
    rows = {}
    for name, handle in states.items():
        actual = read_tensor(learner.trainer, handle)
        master = name.endswith("master")
        if master and not torch.equal(actual, values[name]):
            raise AssertionError("warm-start master differs from checkpoint")
        if not master and torch.count_nonzero(actual).item():
            raise AssertionError("warm-start Adam moment was not reset")
        rows[name] = hashlib.sha256(memoryview(actual.view(torch.uint8).numpy())).hexdigest()
    if (learner.updates, optimizer.steps, learner.draft.weights.version) != (0, 0, 0):
        raise AssertionError("warm-start counters did not reset")
    execution = (tuple(optimizer.execution[name][optimizer.index] for name in
                       ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight"))
                 if learner.options.last_layers == 1 else optimizer.execution[optimizer.index])
    served = learner.serving_banks[learner.publisher.active]
    for source, target in zip(execution, served, strict=True):
        if not torch.equal(read_tensor(learner.trainer, source), read_tensor(learner.native.worker, target)):
            raise AssertionError("serving weights differ from restored learner execution weights")
    return dict(master_equal=True, adam_moments_zero=True, serving_equal=True,
                counters=[0, 0, 0], device_state_sha256=rows)


def choose_width(sweep):
    """只按开发集总吞吐选宽度，拒绝不满足输出一致性的候选。"""
    if not sweep or any(entry["split"] != "dev" or not entry["summary"]["greedy_equal_all"]
                        for entry in sweep):
        raise ValueError("width selection requires lossless development measurements")
    return max(sweep, key=lambda entry: entry["summary"]["tokens_per_second"])["width"]


def publish_initial_suffix(learner):
    """初始化参数也使用同一组驻留 bank，避免把更新后的少上传参数误算成学习收益。"""
    if learner.updates or learner.optimizer.steps or learner.publisher.version:
        raise ValueError("initial publication requires a fresh learner at step zero")
    if learner.options.last_layers == 1:
        order = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
        values = learner.optimizer.tensors()
        learner.publisher.publish(63, [values[name] for name in order], expected_version=0)
    else:
        learner.publisher.publish_flat(learner.optimizer.layout, learner.optimizer.execution[0], expected_version=0)
    # 这只是更新前的设备布局准备，没有执行 Adam；首个训练步骤仍从版本 0 开始。
    learner.publisher.version = 0


def check_continuation(previous, dataset, arguments):
    """续训继续消费新问题，沿用原始留出集和超参；不重放已测过的训练请求。"""
    if not previous.get("completed") or not previous.get("updates"):
        raise ValueError("continuation requires a completed experiment")
    fields = ("last_layers", "width", "stride", "replay_blocks", "learning_rate", "warmup_steps",
              "tokens", "seed", "max_prompt", "dev_count", "test_count", "share_ar_prefix")
    if any(previous["arguments"][key] != arguments[key] for key in fields):
        raise ValueError("continuation must preserve learning, inference and data settings")
    if bool(previous["arguments"].get("frozen_stream_control")) != bool(arguments.get("frozen_stream_control")):
        raise ValueError("continuation must preserve frozen stream control")
    before = previous["dataset"]
    if before["files_sha256"] != dataset["files_sha256"] or any(
            before["splits"][split] != dataset["splits"][split] for split in ("calibration", "dev", "test")):
        raise ValueError("continuation changed the dataset or held-out questions")
    offset = len(previous["train"])
    if (offset >= len(dataset["splits"]["train"]) or
            before["splits"]["train"][:offset] != dataset["splits"]["train"][:offset] or
            [row["id"] for row in previous["train"]] != [row["id"] for row in dataset["splits"]["train"][:offset]]):
        raise ValueError("continuation must retain the used training prefix and have new questions")
    return offset


@contextmanager
def initial_head_control(engine, learner, head):
    """临时读取冻结 AR 参数来重现初始化 head；不写参数，不改变学习状态。"""
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    idle(engine)
    draft = engine.runner.draft
    if draft.learner is not None:
        raise RuntimeError("initial head control requires paused learning")
    weights = draft.weights
    state = learner.updates, learner.optimizer.steps, weights.version
    saved = weights.overrides
    layers, owned = borrow_prefix(engine.runner.native, head, 64)
    # 小常量随 engine 释放，异常请求退出时也不会留下悬空的参数视图。
    draft.handles.extend(owned)
    try:
        weights.overrides = layers
        yield
    finally:
        idle(engine)
        weights.overrides = saved
        if state != (learner.updates, learner.optimizer.steps, weights.version):
            raise RuntimeError("frozen initial-head control changed learning state")


def generate(engine, row, budget, timeout):
    start = time.monotonic()
    request = engine.submit_tokens(row["prompt_ids"], budget)
    tokens = []
    try:
        while time.monotonic() - start < timeout:
            try:
                kind, value = request.events.get(timeout=1)
            except queue.Empty:
                continue
            if kind == "token":
                tokens.append(value["token_id"])
            elif kind == "done":
                idle(engine)
                return dict(id=row["id"], tokens=tokens, output_tokens=len(tokens),
                            seconds=time.monotonic()-start, metrics=request.metrics())
            elif kind in {"error", "cancelled"}:
                raise RuntimeError(f"request {kind}: {value}")
        raise TimeoutError("online effect request timed out")
    finally:
        request.cancelled.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int, required=True)
    p.add_argument("--last-layers", type=int, choices=(1, 4), required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--eval-steps", default="0,8,16,32")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--replay-blocks", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=4)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--train-count", type=int, default=64)
    p.add_argument("--dev-count", type=int, default=3)
    p.add_argument("--test-count", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-prompt", type=int, default=128)
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--share-ar-prefix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--continue-from", type=Path, help="完整实验 JSON；从其中的 Adam checkpoint 接着训练")
    p.add_argument("--warm-from", type=Path, help="已完成预热的实验 JSON；加载 head，重置 Adam，用新问题在线更新")
    p.add_argument("--confirm-count", type=int, default=0, help="预先留出的额外测试题，仅在最终 head 上评测一次")
    p.add_argument("--sweep-widths", default="", help="最终 head 在 dev 上比较的活跃宽度，如 4,8,16,32；物理缓冲仍为 K32")
    p.add_argument("--frozen-stream-control", action="store_true",
                   help="按相同训练请求额外测初始化冻结 head，直接对照在线净吞吐")
    a = p.parse_args()
    if (a.output.exists() or not 2 <= a.width <= 32 or not 1 <= a.tokens <= 384
            or not 1 <= a.max_prompt <= 512-a.tokens or a.steps < 1 or a.warmup_steps < 1
            or min(a.train_count, a.dev_count, a.test_count) < 1 or a.device == a.learner_device):
        p.error("require new output, two devices, valid positive budgets and total context <=512")
    previous = json.loads(a.continue_from.read_text()) if a.continue_from else None
    warm = json.loads(a.warm_from.read_text()) if a.warm_from else None
    sweep_widths = sorted(set(map(int, filter(None, a.sweep_widths.split(",")))))
    if sweep_widths and (min(sweep_widths) < 2 or max(sweep_widths) > 32 or not a.confirm_count):
        p.error("width sweep requires widths in 2..32 and an independent confirmation set")
    if (a.confirm_count < 0 or a.confirm_count and warm is None or
            warm is not None and (previous is not None or a.frozen_stream_control)):
        p.error("confirmation requires warm start; warm start excludes continuation/AR-initial frozen control")
    if warm is not None and (not warm.get("completed") or not warm.get("clean_shutdown") or
            any(warm["arguments"][key] != getattr(a, key) for key in ("last_layers", "width", "tokens", "test_count"))):
        p.error("warm start must preserve suffix layers, block width, output budget and test set size")
    initial_step = previous["updates"][-1]["step"] if previous and previous.get("updates") else 0
    if previous is not None and (not previous.get("completed") or not 0 < initial_step < a.steps):
        p.error("continue from a completed run to a larger total step budget")
    marks = sorted({initial_step, a.steps, *(step for step in map(int, a.eval_steps.split(",")) if step >= initial_step)})
    if marks[0] < 0 or marks[-1] > a.steps:
        p.error("evaluation steps must lie in 0..steps")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("simpler").setLevel(logging.WARNING)
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig

    options = OnlineConfig(last_layers=a.last_layers, stride=a.stride, replay_blocks=a.replay_blocks,
                           learning_rate=a.learning_rate, learner_device=a.learner_device,
                           resume_dir=(previous or warm)["checkpoint"]["path"] if previous or warm else None,
                           reset_optimizer=warm is not None)
    report = dict(completed=False, initialization="AR-copy; no pretrained diffusion head",
                  supervision="verifier reached-prefix full-vocabulary TV", physical_block_size=32,
                  max_model_len=512, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                  updates=[], train=[], evaluations=[], calibration=[])
    if warm is not None:
        manifest = json.loads((Path(options.resume_dir)/"manifest.json").read_text())
        if manifest["step"] != warm["updates"][-1]["step"]:
            p.error("prewarm checkpoint does not match its completed run")
        report.update(initialization="prewarmed last-N head; fresh TV Adam; unseen online questions",
                      warm_start=dict(report_sha256=hashlib.sha256(a.warm_from.read_bytes()).hexdigest(),
                                      checkpoint=options.resume_dir, source_step=manifest["step"],
                                      payload_sha256=manifest["payload_sha256"]))
    if previous is not None:
        manifest = json.loads((Path(options.resume_dir)/"manifest.json").read_text())
        if manifest["step"] != initial_step:
            p.error("checkpoint does not match the completed run's final step")
        for key in ("updates", "train", "evaluations", "calibration"):
            report[key] = list(previous[key])
        if "calibration_target" in previous:
            report["calibration_target"] = previous["calibration_target"]
        report["continuation"] = dict(previous_report=str(a.continue_from),
            previous_report_sha256=hashlib.sha256(a.continue_from.read_bytes()).hexdigest(),
            checkpoint=options.resume_dir, initial_step=initial_step)
    engine = runner = learner = meter = None
    update_original = None
    current_goal = 0

    def persist():
        temporary = a.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        temporary.replace(a.output)

    def emit(kind, value):
        print(json.dumps(dict(event=kind, **value), ensure_ascii=False), flush=True)
        persist()

    try:
        head = HeadSource(a.model)  # 无 head_dir：每层从同一主模型 AR 参数复制，非随机权重。
        engine = make_engine(a.source, a.model, a.device, head, head_transfer="packed",
                             attention_backend="paged", group_size=4, gdn_backend="vector",
                             online_config=options)
        dataset = report["dataset"] = prepare_questions(a.data, engine.tokenizer, seed=a.seed,
            max_prompt=a.max_prompt, train_count=a.train_count + (len(warm["records"]) if warm else 0),
            dev_count=a.dev_count, test_count=a.test_count+a.confirm_count)
        if warm is not None:
            dataset = report["dataset"] = warm_dataset(warm, dataset, train_count=a.train_count,
                                                       confirm_count=a.confirm_count)
        if (a.data/"provenance.json").exists():
            dataset["source"] = json.loads((a.data/"provenance.json").read_text())
        train_offset = check_continuation(previous, dataset, report["arguments"]) if previous else 0
        splits = dataset["splits"]
        persist()
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic()-start
        runner, learner = engine.runner, engine.runner.draft.learner
        if learner.updates != initial_step:
            raise RuntimeError("learner restored a different update step")
        if warm is not None and learner.head_identity != warm["head_identity"]:
            raise ValueError("prewarm base/head contract differs from the online model")
        if previous is None and warm is None:
            publish_initial_suffix(learner)
        if warm is not None:
            report["warm_start"]["device_audit"] = audit_warm_initialization(learner, options.resume_dir)
        report["initial_resident_suffix"] = sorted(runner.draft.weights.overrides)
        report["head_identity"] = learner.head_identity
        report["initial_head_sha256"] = hashlib.sha256(memoryview(head.shared_weights.numpy())).hexdigest()
        if warm is not None:
            # 模板 head 的 SHA 不代表已经加载的预热参数；实际初态绑定 checkpoint 和设备读回。
            report["template_head_sha256"] = report.pop("initial_head_sha256")
        meter = DraftMeter(runner.draft)
        runner.draft.propose = meter
        runner.draft.learner = None
        update_original = learner.update

        def update():
            rate = a.learning_rate * min(1., (learner.updates+1)/a.warmup_steps)
            learner.options = replace(learner.options, learning_rate=rate)
            start = time.monotonic()
            valid = [item.valid for item in learner.replay]
            result = update_original()
            record = dict(step=learner.updates, learning_rate=rate, seconds=time.monotonic()-start,
                          feedback_valid_rows=valid, **result)
            report["updates"].append(record)
            emit("update", record)
            # 在准确的更新步数暂停；该请求剩下的 token 用固定版本完成，不额外学习。
            if learner.updates >= current_goal:
                runner.draft.learner = None
            return result

        learner.update = update

        def measure(row, *, target=False, learning=False, oracle=None, capture_logits=False):
            idle(engine)
            engine.scheduler.runner = runner.native if target else runner
            runner.draft.learner = learner if learning else None
            meter.mode = "oracle_keep_real_draft" if oracle is not None else "real_head"
            meter.reference = oracle
            meter.reset()
            before = (runner.rounds, runner.accepted, learner.updates)
            version_before = runner.draft.weights.version
            original, captured = meter.original, []

            def capture(*args):
                result = original(*args)
                if not captured:
                    import torch
                    logits = runner.draft.logits
                    host = torch.empty(logits.shape, dtype=logits.dtype)
                    runner.native.worker.copy_from(host.data_ptr(), logits.data_ptr, host.numel()*host.element_size())
                    captured.append(hashlib.sha256(memoryview(host.view(torch.uint8).numpy())).hexdigest())
                return result

            if capture_logits:
                meter.original = capture
            try:
                result = generate(engine, row, a.tokens, a.request_timeout)
            finally:
                meter.original = original
            if captured:
                result["first_draft_logits_sha256"] = captured[0]
            result.update(rounds=runner.rounds-before[0], accepted=runner.accepted-before[1],
                          proposed=meter.candidates, updates=learner.updates-before[2],
                          head_version_before=version_before, head_version_after=runner.draft.weights.version,
                          draft_phases=dict(meter.phases))
            if not learning and (result["updates"] or learner.optimizer.steps != learner.updates
                                 or runner.draft.weights.version != version_before):
                raise AssertionError("evaluation changed the head or optimizer state")
            return result

        # 模型预热不学习；正式测量从同一个初始化版本开始。
        runner.draft.block_size = a.width
        measure(splits["calibration"][0], target=True)
        unshared = measure(splits["calibration"][0], capture_logits=a.share_ar_prefix or a.frozen_stream_control)
        if a.share_ar_prefix:
            from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
            layers, owned = borrow_prefix(runner.native, head, 64-a.last_layers)
            runner.draft.handles.extend(owned)
            runner.draft.weights.publish_many(layers, expected_version=initial_step)
            runner.draft.weights.version = initial_step
            shared = measure(splits["calibration"][0], capture_logits=True)
            if (shared["tokens"] != unshared["tokens"] or
                    shared.get("first_draft_logits_sha256") != unshared.get("first_draft_logits_sha256")):
                raise AssertionError("AR-copy prefix sharing changed the initial head logits/output")
            report["prefix_sharing_control"] = dict(layers=sorted(layers), unshared=unshared, shared=shared)
            emit("ar_prefix_sharing", dict(unshared_seconds=unshared["seconds"], shared_seconds=shared["seconds"]))
        if a.frozen_stream_control:
            runner.draft.learner = None
            with initial_head_control(engine, learner, head):
                frozen = measure(splits["calibration"][0], capture_logits=True)
            expected = (previous["frozen_initial_control"] if previous else
                        shared if a.share_ar_prefix else unshared)
            if (frozen["tokens"] != expected["tokens"] or frozen.get("first_draft_logits_sha256") !=
                    expected.get("first_draft_logits_sha256")):
                raise AssertionError("frozen AR bank control changed initial head logits/output")
            report["frozen_initial_control"] = frozen
        if not a.skip_calibration:
            # 仅成本标定忽略 EOS；仍执行完整的真实 draft，再替换为 AR 答案测理想接受上限。
            eos = engine.scheduler.stop_token_ids
            engine.scheduler.stop_token_ids = frozenset()
            reference = measure(splits["calibration"][0], target=True)
            report["calibration_target"] = reference
            for width in (4, 8, 16, 32):
                runner.draft.block_size = width  # 活跃宽度改变；已编译计算仍按 K32 padding。
                row = measure(splits["calibration"][0], oracle=reference["tokens"])
                entry = dict(width=width, row=row, summary=summarize([row], target=[reference]))
                report["calibration"].append(entry)
                emit("oracle_cost_only", entry["summary"] | dict(width=width))
            engine.scheduler.stop_token_ids = eos
        runner.draft.block_size = a.width

        def evaluate(step, final=False):
            entry = dict(step=step, dev={}, test={})
            for name in (("dev", "test") if step == initial_step or final else ("dev",)):
                # 每次评估交错重测同题 AR，减轻共享服务器随时间变化的负载影响。
                refs, rows = [], []
                for row in splits[name]:
                    refs.append(measure(row, target=True))
                    rows.append(measure(row))
                entry[name] = dict(rows=rows, target_rows=refs, summary=summarize(rows, target=refs))
                if not entry[name]["summary"]["greedy_equal_all"]:
                    raise AssertionError(f"{name} speculative output differs from the paired target")
            report["evaluations"] = [old for old in report["evaluations"] if old["step"] != step] + [entry]
            emit("heldout", dict(step=step, **{k:v["summary"] for k,v in entry.items() if k != "step" and v}))

        evaluate(initial_step)
        train_index = train_offset
        for goal in marks[1:]:
            current_goal = goal
            while learner.updates < goal:
                if train_index >= len(splits["train"]):
                    raise RuntimeError("training question budget exhausted; no silent data recycling")
                row = splits["train"][train_index]
                train_index += 1
                result = measure(row, learning=True)
                result["step"] = learner.updates
                report["train"].append(result)
                emit("training_request", {k:v for k,v in result.items() if k not in {"tokens", "metrics", "draft_phases"}})
                if learner.last_result is not None and learner.last_result.get("skipped"):
                    raise FloatingPointError(f"online update skipped: {learner.last_result}")
            evaluate(goal, final=goal == a.steps)
        # 按相同训练问题再跑 target；不将这些参照计算记入在线用户请求耗时。
        train_ids = {row["id"] for row in report["train"][train_offset:]}
        refs = (list(previous["online_net"]["target_rows"]) if previous else [])
        refs += [measure(row, target=True) for row in splits["train"] if row["id"] in train_ids]
        report["online_net"] = dict(summary=summarize(report["train"], target=refs), target_rows=refs,
                                    update_seconds=sum(row["seconds"] for row in report["updates"]))
        if not report["online_net"]["summary"]["greedy_equal_all"]:
            raise AssertionError("online training requests differ from the paired target")
        if a.frozen_stream_control:
            runner.draft.learner = None
            selected = {row["id"]: row for row in splits["train"]}
            frozen_rows = []
            with initial_head_control(engine, learner, head):
                for trained in report["train"]:
                    frozen_rows.append(measure(selected[trained["id"]]))
            report["frozen_stream"] = dict(rows=frozen_rows, summary=summarize(frozen_rows, target=refs))
            if not report["frozen_stream"]["summary"]["greedy_equal_all"]:
                raise AssertionError("frozen head differs from the paired target")
            report["online_net"]["versus_frozen_head_ratio"] = (
                report["online_net"]["summary"]["tokens_per_second"] /
                report["frozen_stream"]["summary"]["tokens_per_second"])
        checkpoint = a.output.with_suffix(".checkpoint")
        if a.last_layers == 1:
            from blockspec.pypto_qwen38.online_checkpoint import save
        else:
            from blockspec.pypto_qwen38.online_suffix_checkpoint import save
        start = time.monotonic()
        save(learner.optimizer, checkpoint, learner.head_identity)
        report["checkpoint"] = dict(path=str(checkpoint), seconds=time.monotonic()-start)
        if sweep_widths:
            report["width_sweep"] = []
            for width in sweep_widths:
                runner.draft.block_size = width
                refs, rows = [], []
                for row in splits["dev"]:
                    refs.append(measure(row, target=True))
                    rows.append(measure(row))
                entry = dict(width=width, split="dev", step=learner.updates, rows=rows,
                             target_rows=refs, summary=summarize(rows, target=refs))
                report["width_sweep"].append(entry)
                emit("width_sweep", dict(width=width, **entry["summary"]))
            runner.draft.block_size = choose_width(report["width_sweep"])
            report["selected_width"] = runner.draft.block_size
        if a.confirm_count:
            refs, rows = [], []
            for row in splits["confirm"]:
                refs.append(measure(row, target=True))
                rows.append(measure(row))
            report["confirmation"] = dict(step=learner.updates, rows=rows, target_rows=refs,
                width=runner.draft.block_size, summary=summarize(rows, target=refs),
                selection="predeclared final update; width chosen on dev" if sweep_widths else
                          "predeclared final update and width; no test-driven selection")
            if not report["confirmation"]["summary"]["greedy_equal_all"]:
                raise AssertionError("confirmation output differs from paired target")
            emit("confirmation", report["confirmation"]["summary"])
        report.update(completed=True, snapshot=engine.snapshot())
        emit("finished", dict(online_net=report["online_net"]["summary"], checkpoint=report["checkpoint"]))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if learner is not None:
            learner.update = update_original or learner.update
        if runner is not None:
            runner.draft.learner = learner
            if meter is not None:
                runner.draft.propose = meter.original
            engine.scheduler.runner = runner
        try:
            if engine is not None:
                engine.stop()
            report["clean_shutdown"] = True
        except BaseException as error:
            report.update(completed=False, cleanup_error=repr(error))
            raise
        finally:
            persist()


if __name__ == "__main__":
    main()
