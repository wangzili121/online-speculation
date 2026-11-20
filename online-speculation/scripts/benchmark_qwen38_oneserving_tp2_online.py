#!/usr/bin/env python3
"""Compare frozen and updating Orthrus TP2 on the same paired request stream."""

import argparse
import hashlib
import json
import os
from math import prod
from pathlib import Path
from queue import Empty
from statistics import median
import time


def load_pairs(path):
    texts = json.loads(path.read_text())
    if (not isinstance(texts, list) or not texts or len(texts) % 2
            or not all(isinstance(text, str) and text.strip() for text in texts)
            or len(set(text.strip() for text in texts)) != len(texts)):
        raise ValueError("expected a nonempty, even, distinct list of question strings")
    return [tuple(texts[i:i + 2]) for i in range(0, len(texts), 2)]


def run_pair(engine, pair, tokens, *, sampling=None, stop_on_eos=False):
    before = dict(engine.runner.spec_stats)
    started = time.monotonic()
    requests = [engine.submit_chat([{"role": "user", "content": question}], tokens,
                                   sampling=params, ignore_eos=not stop_on_eos)
                for question, params in zip(pair, sampling or (None, None), strict=True)]
    for req in requests:
        deadline = started + 300
        while time.monotonic() < deadline:
            try:
                event, value = req.events.get(timeout=1)
            except Empty:
                continue
            if event == "done":
                break
            if event in ("error", "cancelled"):
                raise RuntimeError(str(value)) from engine.scheduler.error
        else:
            raise TimeoutError("paired request exceeded its timeout")
    elapsed = time.monotonic() - started
    if any(not 0 < len(req.generated) <= tokens or
           (not stop_on_eos and len(req.generated) != tokens) for req in requests):
        raise ValueError("response violates the generation budget")
    stats = {key: value - before[key] for key, value in engine.runner.spec_stats.items()}
    return dict(seconds=elapsed, tokens=[list(req.generated) for req in requests],
                metrics=[req.metrics() for req in requests], stats=stats)


def compare(rows, reference):
    if [row["tokens"] for row in rows] != [row["tokens"] for row in reference]:
        raise AssertionError("Orthrus output differs from greedy AR")


def summarize(rows, *, tail_seconds=0.0, tail_stats=None):
    stats = {key: sum(row["stats"][key] for row in rows) for key in rows[0]["stats"]}
    for key, value in (tail_stats or {}).items():
        stats[key] += value
    total_tokens = sum(sum(map(len, row["tokens"])) for row in rows)
    seconds = sum(row["seconds"] for row in rows) + tail_seconds
    return dict(tokens=total_tokens, seconds=seconds, tail_seconds=tail_seconds, tok_s=total_tokens / seconds,
                tpot_median_ms=median(m["tpot_ms"] for row in rows for m in row["metrics"]),
                accepted_per_round=stats["accepted"] / stats["rounds"] if stats["rounds"] else None,
                stats=stats)


def update_policy(max_updates, *, require_continuous=False, warm_updates=0):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("an update cap must be a positive integer")
    if require_continuous and max_updates is not None:
        raise ValueError("continuous updates cannot have a step cap")
    if max_updates is None and warm_updates:
        return "continuous_two_stage"
    return "continuous_periodic" if max_updates is None else "capped_then_frozen"


def request_timeline(rows):
    """Keep every paired interval so active-learning windows can be audited."""
    return [dict(requests=[2 * i + 1, 2 * i + 2], seconds=row["seconds"],
                 output_tokens=[len(tokens) for tokens in row["tokens"]],
                 stats=row["stats"], learner=row["learner"])
            for i, row in enumerate(rows)]


def learner_snapshot(trainer):
    return dict(rounds=trainer.rounds, optimizer_version=trainer.version,
                completed_version=trainer.completed_version, published_version=trainer.published_version,
                update_limit_reached=trainer.update_limit_reached(),
                pending_update=trainer.pending_update is not None,
                queued_update=getattr(trainer, "queued_update", None) is not None)


def measure_interleaved(pairs, issue, trainer, checkpoint=None, *, ar_issue=None):
    """逐对请求交错测量；固定 bank 和在线 bank 常驻，学习状态仅随在线组前进。"""
    if trainer.background or trainer.pending_update is not None or getattr(trainer, "queued_update", None) is not None:
        raise ValueError("interleaved measurement requires synchronous learning")
    rows = {"frozen": [], "online": [], **({"ar": []} if ar_issue is not None else {})}
    orders = (("ar", "frozen", "online"), ("online", "frozen", "ar"),
              ("frozen", "online", "ar"), ("ar", "online", "frozen"),
              ("online", "ar", "frozen"), ("frozen", "ar", "online"))
    for index, pair in enumerate(pairs):
        order = (orders[index % len(orders)] if ar_issue is not None else
                 ("frozen", "online") if index % 2 == 0 else ("online", "frozen"))
        for mode in order:
            before = learner_snapshot(trainer)
            trainer.pending_evaluation = "current" if mode == "online" else "initial"
            row = (ar_issue if mode == "ar" else issue)(pair, index)
            row["learner"] = learner_snapshot(trainer)
            if row["learner"]["pending_update"] or row["learner"]["queued_update"]:
                raise RuntimeError("interleaved request pair left pending learning work")
            if mode != "online" and (row["learner"] != before or any(
                    row["stats"].get(key, 0) for key in
                    ("feedback_ns", "update_ns", "publish_ns", "updates", "publications"))):
                raise RuntimeError(f"{mode} evaluation performed learning work")
            rows[mode].append(row)
        if checkpoint is not None:
            checkpoint(rows, index)
    result = (rows["frozen"], rows["online"])
    return (*result, rows["ar"]) if ar_issue is not None else result


def exception_chain(error):
    result, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        result.append(dict(type=type(error).__name__, message=str(error)))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return result


def finish_engine(engine, summary, persist):
    try:
        if engine.scheduler.thread.ident is not None:
            engine.stop()
            summary["clean_shutdown"] = True
    except BaseException as cleanup:
        summary["completed"] = False
        summary["cleanup_error"] = repr(cleanup)
        if not summary.get("error"):
            summary["error"] = repr(cleanup)
            summary["error_chain"] = exception_chain(cleanup)
            raise
    finally:
        persist()


def main(*, guard=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "head", "prompts", "holdout", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--checkpoint", type=Path,
                   help="restore optimizer state; omit for fresh Adam on the exact served head")
    p.add_argument("--cold-head", type=Path, help="compare this head if no warm online mode beats frozen")
    p.add_argument("--learner-device", default=2, choices=(0, 2), type=int)
    p.add_argument("--kv-pages", type=int, help="cap the resident KV pool equally for all modes")
    p.add_argument("--ring-heap-mib", type=int, default=256,
                   help="minimum temporary heap per ring; identical for AR, frozen and online")
    p.add_argument("--strides", default="16,32,64")
    p.add_argument("--tokens", default=128, type=int)
    p.add_argument("--stop-on-eos", action="store_true", help="use natural EOS within the token budget")
    p.add_argument("--candidates", default=31, type=int, choices=range(1, 32),
                   help="maximum proposed tokens from the unchanged K32 draft block")
    p.add_argument("--verify-chunk-size", default=4, type=int, choices=(1, 2, 4),
                   help="rows per verification dispatch within the unchanged native C4 snapshot capacity")
    p.add_argument("--draft-convolution", choices=("sequential", "parallel64", "parallel128"), default="sequential")
    p.add_argument("--audit-convolution", action="store_true",
                   help="diagnostic: compare complete real draft logits against sequential convolution")
    p.add_argument("--last-layers", default=4, type=int, choices=(1, 4))
    p.add_argument("--parameter-scope", choices=("full", "norms", "output_norms"), default="full")
    p.add_argument("--publication-backend", choices=("auto", "cpu", "device"), default="auto")
    p.add_argument("--learning-rate", default=1e-5, type=float)
    p.add_argument("--warmup-updates", default=0, type=int)
    p.add_argument("--warm-update-stride", type=int,
                   help="initial cadence before switching to the continuing --strides interval")
    p.add_argument("--warm-update-steps", type=int, default=0)
    p.add_argument("--lr-schedule", choices=("constant", "inverse_sqrt"), default="constant")
    p.add_argument("--objective", default="tv", choices=("tv", "kl"))
    p.add_argument("--focus-weight", default=1, type=int,
                   help="TV weight on reached candidate offsets 2 and 3; one keeps the ordinary objective")
    p.add_argument("--position-weighting", choices=("uniform", "rejection_decay", "first_rejection"), default="uniform",
                   help="uniform, decay after first rejection, or boost only its verified row16x")
    p.add_argument("--feedback-mode", choices=("reached", "delayed"), default="reached")
    p.add_argument("--feedback-timing", choices=("early", "near_update"), default="early",
                   help="delayed feedback may be collected closer to the next scheduled update")
    p.add_argument("--teacher-tokens", type=int, choices=range(1, 32), default=31)
    p.add_argument("--teacher-temperature", type=float, default=1.0,
                   help="temperature of the learner's teacher distribution; does not change generation")
    p.add_argument("--teacher-scaling", choices=("host", "device"), default="host",
                   help="device scaling uses the existing PyPTO elementwise kernel at power-of-two temperatures")
    p.add_argument("--teacher-targets", choices=("distribution", "greedy"), default="distribution",
                   help="greedy learns emitted deterministic target IDs with the existing KL kernel")
    p.add_argument("--audit-replay", action="store_true",
                   help="diagnose the first feedback replay against actual draft logits; include its cost")
    p.add_argument("--max-updates", type=int, help="cap Adam steps during the measured online stream")
    p.add_argument("--continuous-updates", action="store_true",
                   help="require periodic learning through the entire stream; reject a step cap")
    p.add_argument("--replay-blocks", default=1, type=int)
    p.add_argument("--publish-every", default=1, type=int)
    p.add_argument("--inner-steps", default=1, type=int,
                   help="Adam steps per verified replay event; must divide the publication interval")
    p.add_argument("--ar-check-requests", type=int,
                   help="limit the AR correctness sample; online and frozen are still matched in full")
    p.add_argument("--replay-learned-head", action="store_true",
                   help="replay the full request stream with the final published head and updates disabled")
    p.add_argument("--export-heads", type=Path,
                   help="save each final published head outside the timed stream")
    p.add_argument("--gather-feedback", action="store_true", help="pack paged training history on the NPU")
    p.add_argument("--temperature", default=0, type=float, choices=(0, 1))
    p.add_argument("--lazy-sampling", action="store_true")
    p.add_argument("--seed", default=743, type=int)
    p.add_argument("--background", action="store_true", help="train on the learner while serving")
    p.add_argument("--overlap-sampling", action="store_true",
                   help="run colocated suffix updates during CPU lazy sampling; drain before GPU work and request completion")
    p.add_argument("--interleaved", action="store_true",
                   help="alternate frozen/online request pairs using resident banks; synchronous learning only")
    p.add_argument("--interleave-ar", action="store_true",
                   help="interleave full AR/frozen/online triplets in six balanced orders; requires --interleaved")
    p.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    if args.export_heads is not None and args.export_heads.exists():
        p.error("head export directory must be new")
    try:
        policy = update_policy(args.max_updates, require_continuous=args.continuous_updates,
                               warm_updates=args.warm_update_steps)
    except ValueError as error:
        p.error(str(error))
    if args.cold_head is not None and args.checkpoint is not None:
        p.error("cold fallback uses fresh Adam for both starts")
    if args.feedback_timing != "early" and args.feedback_mode != "delayed":
        p.error("near_update timing requires delayed feedback")
    if args.teacher_targets == "greedy" and (args.temperature != 0 or args.objective != "kl"
            or args.teacher_temperature != 1 or args.teacher_scaling != "host"):
        p.error("greedy teacher targets require greedy generation, KL, teacher temperature one and host preparation")
    strides = [int(value) for value in args.strides.split(",")]
    if args.interleaved and (args.background or len(strides) != 1):
        p.error("interleaved evaluation requires one stride and synchronous updates")
    if args.interleave_ar and (not args.interleaved or args.cold_head is not None):
        p.error("AR triplets require --interleaved and a single starting head")
    if (min(args.replay_blocks, args.publish_every) < 1 or len(set(strides)) != len(strides)
            or any(n < 2 * args.replay_blocks for n in strides) or args.tokens < 2 or args.output.exists()):
        p.error("use distinct strides >= twice replay-blocks, positive counts and a fresh result path")
    if args.warm_update_stride is not None and (
            args.warm_update_stride < 2 * args.replay_blocks or any(args.warm_update_stride >= n for n in strides)):
        p.error("initial cadence must fit replay blocks and be smaller than every continuing stride")
    prompts, holdout = load_pairs(args.prompts), load_pairs(args.holdout)
    ar_requests = 2 * len(prompts) if args.ar_check_requests is None else args.ar_check_requests
    if not 2 <= ar_requests <= 2 * len(prompts) or ar_requests % 2:
        p.error("AR sample must contain an even number of requests within the stream")
    if args.interleave_ar and ar_requests != 2 * len(prompts):
        p.error("AR triplets require the entire request stream, not an AR sample")
    if {q.strip() for pair in prompts for q in pair} & {q.strip() for pair in holdout for q in pair}:
        p.error("request stream and frozen holdout must have distinct questions")

    from pypto.runtime import configure_log
    from serving.distributed import SamplingParams, ServingConfig
    from serving.engine import InferenceEngine

    from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    configure_log("warn")
    trainer_type = TP2Online
    if args.feedback_mode == "delayed":
        from blockspec.pypto_qwen38.oneserving_tp2_delayed import TP2DelayedOnline
        trainer_type = TP2DelayedOnline
    extra = (dict(teacher_tokens=args.teacher_tokens, feedback_timing=args.feedback_timing)
             if args.feedback_mode == "delayed" else {})
    trainer = trainer_type(args.checkpoint, device=args.learner_device, last_layers=args.last_layers,
                        learning_rate=args.learning_rate, background=args.background,
                        replay_blocks=args.replay_blocks, publish_every=args.publish_every,
                        gather_feedback=args.gather_feedback, objective=args.objective, max_updates=args.max_updates,
                        focus_weight=args.focus_weight, audit_replay=args.audit_replay,
                        teacher_temperature=args.teacher_temperature, teacher_scaling=args.teacher_scaling,
                        teacher_targets=args.teacher_targets,
                        warmup_updates=args.warmup_updates,
                        lr_schedule=args.lr_schedule, parameter_scope=args.parameter_scope,
                        warm_update_stride=args.warm_update_stride, warm_update_steps=args.warm_update_steps,
                        publication_backend=args.publication_backend, inner_steps=args.inner_steps,
                        overlap_sampling=args.overlap_sampling, position_weighting=args.position_weighting, **extra)
    config = ServingConfig(args.model, (0, 1), max_model_len=512, max_num_batched_tokens=128,
                           max_num_seqs=2, speculative_tokens=0, prefix_cache=False,
                           long_prefill_token_threshold=4)
    engine = InferenceEngine(config)
    runner = engine.runner = OrthrusTP2Runner(config, args.head, draft_batch_size=2,
                                              online=trainer, kv_pages=args.kv_pages,
                                              stochastic=bool(args.temperature), lazy_sampling=args.lazy_sampling,
                                              verify_chunk_size=args.verify_chunk_size,
                                              ring_heap_min_mib=args.ring_heap_mib,
                                              on_cache_event=engine._on_cache,
                                              draft_convolution=args.draft_convolution,
                                              audit_convolution=args.audit_convolution)
    engine.scheduler.runner = runner
    summary = dict(completed=False, head_step=runner.head["step"], device_ids=list(config.devices),
                   head_payload_sha256=runner.head["source_payload_sha256"],
                   head_shard_sha256=[shard["sha256"] for shard in runner.head["rank_files"]],
                   visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                   learner_device=args.learner_device, last_layers=args.last_layers,
                   parameter_scope=args.parameter_scope,
                   publication_backend="device" if trainer.device_publication else "cpu",
                   physical_device_count=2 if trainer.colocated else 3, kv_pages=runner.pool_blocks,
                   resource_policy=runner.resource_policy,
                   learning_rate=args.learning_rate, background=args.background,
                   overlap_sampling=args.overlap_sampling,
                   benchmark_layout=("interleaved_ar_frozen_online" if args.interleave_ar else
                                     "interleaved_pairs" if args.interleaved else "separate_streams"),
                   interleaved_order=("six_permutations_within_pair" if args.interleave_ar else
                                      "alternating_within_pair" if args.interleaved else None),
                   primary_metric="continuous_online_vs_target_only_AR",
                   warmup_updates=args.warmup_updates,
                   warm_update_stride=args.warm_update_stride, warm_update_steps=args.warm_update_steps,
                   lr_schedule=args.lr_schedule,
                   objective=args.objective, max_updates=args.max_updates,
                   focus_weight=args.focus_weight,
                   position_weighting=args.position_weighting,
                   draft_convolution=args.draft_convolution, diagnostic_only=args.audit_convolution,
                   feedback_mode=args.feedback_mode,
                   teacher_tokens=args.teacher_tokens if args.feedback_mode == "delayed" else None,
                   feedback_timing=args.feedback_timing,
                   teacher_temperature=args.teacher_temperature,
                   teacher_scaling=args.teacher_scaling,
                   teacher_targets=args.teacher_targets,
                   audit_replay=args.audit_replay,
                   online_update_policy=policy,
                   temperature=args.temperature, seed=args.seed,
                   sampler=("host_full_vocab_lazy" if args.lazy_sampling else "host_full_vocab")
                           if args.temperature else "device_greedy",
                   replay_blocks=args.replay_blocks, publish_every=args.publish_every, inner_steps=args.inner_steps,
                   stop_on_eos=args.stop_on_eos, candidates=args.candidates,
                   verify_chunk_size=args.verify_chunk_size,
                   requests=2 * len(prompts), ar_checked_requests=ar_requests, output_tokens_per_request=args.tokens,
                   prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                   holdout_sha256=hashlib.sha256(args.holdout.read_bytes()).hexdigest(),
                   gather_feedback=args.gather_feedback,
                   optimizer_init="checkpoint" if args.checkpoint is not None else "fresh", results={})

    def persist():
        summary["convolution_audits"] = list(runner.convolution_audits)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n")
        tmp.replace(args.output)

    def issue(pair, tokens, index=0):
        params = None if not args.temperature else [
            SamplingParams(temperature=args.temperature, seed=args.seed + 2 * index + rank)
            for rank in range(2)
        ]
        return run_pair(engine, pair, tokens, sampling=params, stop_on_eos=args.stop_on_eos)

    def check_tokens(rows, reference):
        if not args.temperature:
            compare(rows, reference)

    def measure(pairs, *, label=None, frozen=None):
        rows = []
        previous = time.monotonic()
        for index, pair in enumerate(pairs):
            try:
                row = issue(pair, args.tokens, index)
            except Exception as error:
                raise RuntimeError(f"{label or 'holdout'} pair {index + 1} failed: {error}") from error
            finished = time.monotonic()
            # Background work between submissions counts against the same stream.
            row["seconds"] = finished - previous
            previous = finished
            row["learner"] = learner_snapshot(trainer)
            rows.append(row)
            if label is not None and index + 1 < len(pairs) and (index + 1) % 16 == 0:
                stream = summarize(rows)
                milestone = dict(requests=2 * len(rows), tokens=stream["tokens"], seconds=stream["seconds"],
                                 tok_s=stream["tok_s"], accepted_per_round=stream["accepted_per_round"],
                                 **{k: stream["stats"][k] for k in ("updates", "publications")})
                if frozen is not None:
                    frozen_tps = summarize(frozen[:len(rows)])["tok_s"]
                    milestone["net_increment"] = stream["tok_s"] / frozen_tps - 1
                summary.setdefault("progress", {}).setdefault(label, []).append(milestone)
                persist()
                print(label, milestone, flush=True)
        return rows

    if args.compile_only:
        paths = [runner.compile_draft(size).output_dir for size in (1, 2)]
        if args.gather_feedback:
            paths.append(trainer.compile_history(runner).output_dir)
        paths += [program.output_dir for program in trainer.compile(runner.runconfig)]
        summary.update(compiled=list(map(str, paths)), completed=True)
        persist()
        return
    if guard is not None:
        admission = runner._check_memory_admission
        compile_learner = trainer.compile

        def admit():
            for rank in config.devices:
                guard.release(rank)
            if trainer.colocated:
                guard.close()
            admission()

        def prepare_learner(config):
            programs = compile_learner(config)
            guard.release(args.learner_device)
            guard.close()
            return programs
        runner._check_memory_admission = admit
        if not trainer.colocated:
            trainer.compile = prepare_learner
    try:
        start = time.monotonic()
        summary["stage"] = "starting"
        persist()
        print("starting TP2 + learner", flush=True)
        engine.start()
        summary["startup_seconds"] = time.monotonic() - start
        summary["learner_memory_plan"] = trainer.memory_plan
        summary["trainable_parameters"] = sum(prod(p.shape) for p in trainer.graph.parameters)
        summary["training_dispatches"] = len(trainer.training.chunks) or 1
        summary["stage"] = "measuring"
        persist()
        print("ready TP2 + learner", summary["startup_seconds"], flush=True)
        runner.output_candidates = 0
        issue(prompts[0], args.tokens)
        summary["device_memory_after_ar_warmup"] = [list(runner.worker.device_memory_info(rank))
                                                     for rank in config.devices]
        ar = [] if args.interleave_ar else measure(prompts[:ar_requests // 2], label="ar")
        holdout_ar = measure(holdout, label="ar_holdout")
        if ar:
            summary["results"]["ar"] = dict(stream=summarize(ar), holdout=summarize(holdout_ar),
                                             request_timeline=request_timeline(ar))
        persist()

        def suite(results, suite_strides):
            nonlocal ar
            if args.interleaved:
                trainer.pending_evaluation = "initial"
                frozen, holdout_frozen = None, measure(holdout)
            else:
                frozen, holdout_frozen = measure(prompts, label=f"frozen_{runner.head['step']}"), measure(holdout)
                results["frozen"] = dict(stream=summarize(frozen), holdout=summarize(holdout_frozen),
                                         request_timeline=request_timeline(frozen))
                check_tokens(frozen[:len(ar)], ar)
            check_tokens(holdout_frozen, holdout_ar)
            persist()

            for stride in suite_strides:
                trainer.pending_reset = True
                if args.interleaved:
                    trainer.pending_evaluation = "current"
                issue(prompts[0], 1)
                if trainer.version != trainer.initial_step:
                    raise RuntimeError("online case did not restore the shared initial step")
                first = len(trainer.records)
                busy_start = runner.spec_stats["busy_updates"]
                feedback_skip_start = runner.spec_stats["feedback_skips"]
                trainer.stride = stride
                if args.interleaved:
                    def checkpoint(rows, index):
                        if index + 1 == len(prompts) or (index + 1) % 16:
                            return
                        fixed, current = summarize(rows["frozen"]), summarize(rows["online"])
                        milestone = dict(requests=2 * (index + 1), fixed_tok_s=fixed["tok_s"],
                                         tok_s=current["tok_s"], net_increment=current["tok_s"] / fixed["tok_s"] - 1,
                                         accepted_per_round=current["accepted_per_round"],
                                         updates=current["stats"]["updates"], publications=current["stats"]["publications"])
                        if args.interleave_ar:
                            ar_tps = summarize(rows["ar"])["tok_s"]
                            milestone.update(ar_tok_s=ar_tps, net_increment_vs_ar=current["tok_s"] / ar_tps - 1)
                        summary.setdefault("progress", {}).setdefault(f"paired_{stride}", []).append(milestone)
                        persist()
                        print("paired", milestone, flush=True)
                    def issue_ar(pair, index):
                        runner.output_candidates = 0
                        try:
                            return issue(pair, args.tokens, index)
                        finally:
                            runner.output_candidates = args.candidates
                    measured = measure_interleaved(
                        prompts, lambda pair, index: issue(pair, args.tokens, index), trainer, checkpoint,
                        ar_issue=issue_ar if args.interleave_ar else None)
                    frozen, stream = measured[:2]
                    if args.interleave_ar:
                        ar = measured[2]
                        results["ar"] = dict(stream=summarize(ar), holdout=summarize(holdout_ar),
                                             request_timeline=request_timeline(ar))
                    check_tokens(frozen[:len(ar)], ar)
                    results["frozen"] = dict(stream=summarize(frozen), holdout=summarize(holdout_frozen),
                                             request_timeline=request_timeline(frozen))
                    trainer.pending_evaluation = "current"
                else:
                    stream = measure(prompts, label=f"online_{runner.head['step']}_{stride}", frozen=frozen)
                trainer.stride = None
                tail = None
                if trainer.pending_update is not None:
                    trainer.pending_drain = True
                    tail = issue(prompts[0], 1)
                updates = trainer.records[first:]
                skipped = runner.spec_stats["busy_updates"] - busy_start
                feedback_skips = runner.spec_stats["feedback_skips"] - feedback_skip_start
                completed_events, incomplete_steps = divmod(len(updates), args.inner_steps)
                if (incomplete_steps or completed_events + skipped + feedback_skips != trainer.scheduled_updates
                        or not updates):
                    raise RuntimeError("online case submitted an unexpected number of updates")
                if [update["version"] for update in updates] != list(
                        range(trainer.initial_step + 1, trainer.version + 1)):
                    raise RuntimeError("updated TP2 weights did not advance through all versions")
                check_tokens(stream, frozen)
                final_version = trainer.published_version
                optimizer_version = trainer.version
                expected = trainer.initial_step + len(updates) // args.publish_every * args.publish_every
                if trainer.completed_version != optimizer_version or final_version != expected:
                    raise RuntimeError("the learner or serving version does not match its update cadence")
                learned_holdout = measure(holdout)
                check_tokens(learned_holdout, holdout_ar)
                if (trainer.version, trainer.published_version) != (optimizer_version, final_version):
                    raise RuntimeError("frozen holdout changed the optimizer or serving version")
                result = dict(stream=summarize(stream, tail_seconds=tail["seconds"] if tail else 0.0,
                                               tail_stats=tail["stats"] if tail else None),
                              holdout=summarize(learned_holdout), exact_greedy=True if not args.temperature else None,
                              updates=updates, skipped_busy=skipped,
                              skipped_feedback=feedback_skips,
                              supervised_positions=sum(step["supervised"] for step in updates
                                                       if step.get("inner_step", 1) == 1),
                              training_position_exposures=sum(step["supervised"] for step in updates),
                              paired_seconds=[[base["seconds"], trial["seconds"]]
                                              for base, trial in zip(frozen, stream, strict=True)],
                              published_version=final_version, optimizer_version=optimizer_version)
                result["update_schedule"] = dict(
                    policy=policy, stride=stride, max_updates=args.max_updates,
                    rounds=trainer.rounds, scheduled_updates=trainer.scheduled_updates,
                    warm_update_stride=args.warm_update_stride, warm_update_steps=args.warm_update_steps,
                    completed_updates=len(updates), skipped_busy=skipped,
                    completed_update_events=completed_events, inner_steps=args.inner_steps,
                    skipped_feedback=feedback_skips,
                    last_update_round=updates[-1]["round"],
                    rounds_after_last_update=trainer.rounds - updates[-1]["round"],
                    update_limit_reached=trainer.update_limit_reached(),
                )
                result["request_timeline"] = request_timeline(stream)
                if args.continuous_updates and result["update_schedule"]["update_limit_reached"]:
                    raise RuntimeError("continuous run reached an update limit")
                result["net_increment"] = result["stream"]["tok_s"] / results["frozen"]["stream"]["tok_s"] - 1
                if ar_requests == 2 * len(prompts):
                    result["net_increment_vs_ar"] = result["stream"]["tok_s"] / summarize(ar)["tok_s"] - 1
                results[str(stride)] = result
                persist()
                if args.replay_learned_head:
                    learned = measure(prompts, label=f"learned_{runner.head['step']}_{stride}", frozen=frozen)
                    check_tokens(learned, frozen)
                    replay = summarize(learned)
                    if (replay["stats"]["updates"] or replay["stats"]["publications"]
                            or (trainer.version, trainer.published_version) != (optimizer_version, final_version)):
                        raise RuntimeError("learned-head replay changed the optimizer or serving version")
                    result["learned_replay"] = dict(
                        stream=replay,
                        exact_greedy=result["exact_greedy"],
                        paired_seconds=[[base["seconds"], row["seconds"]]
                                        for base, row in zip(frozen, learned, strict=True)],
                    )
                    print("learned replay", runner.head["step"], stride, replay, flush=True)
                if args.export_heads is not None:
                    from blockspec.pypto_qwen38.tp2_head_checkpoint import export_served_head
                    destination = args.export_heads / f"head{runner.head['step']}-stride{stride}"
                    started = time.monotonic()
                    deadline = started + 5
                    while runner.requests and time.monotonic() < deadline:
                        time.sleep(.001)
                    manifest = export_served_head(trainer, destination)
                    result["exported_head"] = dict(path=str(destination), manifest=manifest,
                                                   seconds=time.monotonic() - started)
                persist()
                print("stride", runner.head["step"], stride, result["stream"], result["holdout"], flush=True)

        runner.output_candidates = args.candidates
        issue(prompts[0], args.tokens)
        suite(summary["results"], strides)
        if args.cold_head is not None and all(
            summary["results"][str(stride)]["stream"]["tok_s"]
            <= summary["results"]["frozen"]["stream"]["tok_s"] for stride in strides
        ):
            print("warm modes did not beat frozen; starting the cold head comparison", flush=True)
            trainer.pending_head = args.cold_head
            issue(prompts[0], 1)
            if runner.head["step"] != 0:
                raise ValueError("the cold comparison requires a step0 head")
            cold = summary["cold"] = {"head_step": runner.head["step"], "results": {}}
            suite(cold["results"], [strides[0]])
        summary["completed"] = True
        summary["stage"] = "completed"
    except BaseException as error:
        summary["error"] = repr(error)
        summary["error_chain"] = exception_chain(error)
        raise
    finally:
        finish_engine(engine, summary, persist)


if __name__ == "__main__":
    main()
