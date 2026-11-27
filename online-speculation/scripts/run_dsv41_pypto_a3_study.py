# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare A3 AR, live updates and a frozen head on paired GSM8K inputs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


def load_prompts(path, limit, encode, context, tokens):
    records = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            prompt = encode(row["prompt"])
            if not prompt or len(prompt) + tokens > context:
                raise ValueError(f"GSM8K prompt {row['id']} and generation exceed the compiled context")
            records.append((row["id"], prompt))
            if len(records) == limit:
                break
    if len(records) != limit or len({key for key, _ in records}) != limit:
        raise ValueError(f"Expected {limit} distinct prompts in {path}")
    return records


def measure(session, rows, tokens, output, *, name, baseline=None):
    summaries, traces = [], {}
    for request_id, prompt in rows:
        emitted = []
        token_times = []

        def record(token):
            emitted.append(token)
            token_times.append(perf_counter())

        start = perf_counter()
        result = session.generate(request_id, prompt, tokens, True, record, lambda: False)
        seconds = perf_counter() - start
        updates = result.get("online_updates", [])
        row = dict(arm=name, request=request_id, token_ids=emitted, seconds=seconds,
                   ttft_seconds=token_times[0] - start,
                   decode_seconds=token_times[-1] - token_times[0],
                   update_seconds=sum(u["wall_seconds"] for u in updates),
                   speculation=result.get("speculation", {}), updates=updates,
                   phase_seconds=result.get("phase_seconds", {}))
        if baseline is not None:
            row["greedy_match"] = emitted == baseline[request_id]
        output.write(json.dumps(row) + "\n")
        output.flush()
        if row.get("greedy_match") is False:
            raise ValueError(f"{name} diverged from AR on request {request_id}")
        summaries.append(row)
        traces[request_id] = emitted

    count = sum(len(row["token_ids"]) for row in summaries)
    wall = sum(row["seconds"] for row in summaries)
    update = sum(row["update_seconds"] for row in summaries)
    full_updates = [u for row in summaries for u in row["updates"] if u["applied"]]
    proposed = sum(row["speculation"].get("proposed", 0) for row in summaries)
    accepted = sum(row["speculation"].get("accepted", 0) for row in summaries)
    rounds = sum(row["speculation"].get("rounds", 0) for row in summaries)
    phases = Counter()
    for row in summaries:
        phases.update(row["phase_seconds"])
    summary = dict(requests=len(rows), tokens=count, seconds=wall,
                   wall_tok_s=count / wall, estimated_tok_s_without_updates=count / max(wall - update, 1e-9),
                   ttft_median_seconds=median(row["ttft_seconds"] for row in summaries),
                   updates=len(full_updates), update_seconds=update, phase_seconds=dict(phases),
                   publication_seconds=sum(u["publication_seconds"] for u in full_updates),
                   proposed=proposed, accepted=accepted, accepted_per_round=accepted / rounds if rounds else None)
    print(json.dumps(dict(arm=name, **summary)), flush=True)
    return summary, traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-root", type=Path, required=True, help="DeepSeek-V4.1-Flash-A3 checkout")
    parser.add_argument("--lib-root", type=Path, required=True, help="Matching pypto-lib checkout")
    workload = parser.add_mutually_exclusive_group(required=True)
    workload.add_argument("--data", type=Path, help="Prepared GSM8K directory containing tuning and quality_test")
    workload.add_argument("--smoke", action="store_true", help="Exercise all arms on two built-in prompts")
    parser.add_argument("--verify-gate", action="store_true",
                        help="Audit all acceptance lengths and chained rejections on the same smoke worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--heldout-count", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-count", type=int, default=1)
    experiment, service_args = parser.parse_known_args(argv)
    serving = experiment.serving_root.expanduser().resolve() / "pypto-serving-deepseek-v4.1-flash-a3"
    lib = experiment.lib_root.expanduser().resolve()
    if not (serving / "serving" / "orthrus.py").is_file():
        parser.error("--serving-root must contain the A3 Orthrus serving adapter")
    if not (lib / "models" / "deepseek_v4_1_flash_a3" / "orthrus_draft").is_dir():
        parser.error("--lib-root must contain the A3 Orthrus model")
    sys.path.insert(0, str(serving))
    os.environ["PYPTO_LIB_ROOT"] = str(lib)
    from serving.api import launch_config, parse_args
    from serving.runtime import open_session

    config = launch_config(parse_args(service_args))
    if not config.online_last_layers:
        parser.error("The study requires --online-last-layers and --orthrus-head")
    if experiment.verify_gate and not experiment.smoke:
        parser.error("--verify-gate is a smoke check")
    if min(experiment.train_count, experiment.heldout_count) <= 0 or experiment.tokens < 3 or experiment.warmup_count < 0:
        parser.error("Choose positive sample counts and at least three output tokens")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(config.model_root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    def encode(text):
        return tokenizer.encode(text).ids

    tokens = min(8, experiment.tokens) if experiment.smoke else experiment.tokens
    gate_prompt = None
    if experiment.verify_gate:
        from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE

        if config.context < BLOCK_SIZE + config.verify_width + 3:
            parser.error("--verify-gate requires context for the physical page boundary and chained rejections")
        gate_prompt = encode("We can verify the mathematical result by working through each step. " * 128)
        gate_prompt = gate_prompt[:BLOCK_SIZE - 1]
    if experiment.smoke:
        train = [("smoke-train", encode("Compute seven plus five. Show each step."))]
        heldout = [("smoke-heldout", encode("Compute nine minus three. Show each step."))]
    else:
        train = load_prompts(experiment.data / "tuning.jsonl", experiment.train_count, encode, config.context, tokens)
        heldout = load_prompts(experiment.data / "quality_test.jsonl", experiment.heldout_count, encode, config.context, tokens)
    warmup = []
    if not experiment.smoke and experiment.warmup_count:
        warmup = load_prompts(experiment.data / "warmup.jsonl", experiment.warmup_count, encode, config.context, tokens)
    if {tuple(prompt) for _, prompt in train} & {tuple(prompt) for _, prompt in heldout}:
        parser.error("Online training and heldout prompts overlap")
    if any(not prompt or len(prompt) + tokens > config.context for _, prompt in (*train, *heldout)):
        parser.error("A workload exceeds the configured history")
    if not any(len(prompt) + config.draft_block <= config.context for _, prompt in train):
        parser.error("The training workload leaves no room to capture an online draft")

    predictions = experiment.output.with_suffix(".jsonl")
    if predictions == experiment.output or predictions.exists() or experiment.output.exists():
        parser.error("Use a fresh JSON summary path for this study")
    experiment.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with open_session(config) as session, predictions.open("x") as out:
        cold = perf_counter() - start
        runner = session.orthrus
        learner = runner.learner
        summary = dict(cold_start_seconds=cold, concurrency=1, tokens_per_request=tokens,
                       runtime=session.metadata, ignore_eos=True)
        if gate_prompt is not None:
            from models.deepseek_v4_1_flash_a3.validate_orthrus import validate_resident_verifier

            start = perf_counter()
            runner.target.audit_logits = True
            try:
                checks = validate_resident_verifier(runner.target, gate_prompt, width=config.verify_width)
            finally:
                runner.target.audit_logits = False
            session.graphs.reset()
            summary["verifier_gate"] = dict(cases=checks, seconds=perf_counter() - start)
            print(json.dumps(dict(stage="verifier_gate", passed=True, cases=len(checks))), flush=True)
        if warmup:
            runner.learner = None
            session.orthrus = None
            summary["warmup_ar"], warmup_ar = measure(session, warmup, tokens, out, name="warmup_ar")
            session.orthrus = runner
            summary["warmup_draft"], _ = measure(session, warmup, tokens, out, name="warmup_draft", baseline=warmup_ar)
            runner.learner = learner
        session.orthrus = None
        for name, rows in (("ar", train), ("ar_heldout", heldout)):
            summary[name], baseline = measure(session, rows, tokens, out, name=name)
            if name == "ar":
                train_ar = baseline
            else:
                heldout_ar = baseline
        session.orthrus = runner
        runner.learner = None
        summary["initial"], _ = measure(session, heldout, tokens, out, name="initial", baseline=heldout_ar)
        runner.learner = learner
        runner.rounds = 0
        summary["online"], _ = measure(session, train, tokens, out, name="online", baseline=train_ar)
        runner.learner = None
        summary["frozen"], _ = measure(session, heldout, tokens, out, name="frozen", baseline=heldout_ar)
        summary["passed"] = summary["online"]["updates"] > 0
        experiment.output.write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit("The study completed without an applied online update")


if __name__ == "__main__":
    main()
