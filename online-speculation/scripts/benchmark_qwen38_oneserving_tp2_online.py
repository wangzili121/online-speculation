#!/usr/bin/env python3
"""Compare frozen and updating Orthrus TP2 on the same paired request stream."""

import argparse
import hashlib
import json
import os
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


def update_policy(max_updates, *, require_continuous=False):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("an update cap must be a positive integer")
    if require_continuous and max_updates is not None:
        raise ValueError("continuous updates cannot have a step cap")
    return "continuous_periodic" if max_updates is None else "capped_then_frozen"


def request_timeline(rows):
    """Keep every paired interval so active-learning windows can be audited."""
    return [dict(requests=[2 * i + 1, 2 * i + 2], seconds=row["seconds"],
                 output_tokens=[len(tokens) for tokens in row["tokens"]],
                 stats=row["stats"], learner=row["learner"])
            for i, row in enumerate(rows)]


def main(*, guard=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "head", "prompts", "holdout", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--checkpoint", type=Path,
                   help="restore optimizer state; omit for fresh Adam on the exact served head")
    p.add_argument("--cold-head", type=Path, help="compare this head if no warm online mode beats frozen")
    p.add_argument("--learner-device", default=2, choices=(0, 2), type=int)
    p.add_argument("--kv-pages", type=int, help="cap the resident KV pool equally for all modes")
    p.add_argument("--strides", default="16,32,64")
    p.add_argument("--tokens", default=128, type=int)
    p.add_argument("--stop-on-eos", action="store_true", help="use natural EOS within the token budget")
    p.add_argument("--candidates", default=31, type=int, choices=range(1, 32),
                   help="maximum proposed tokens from the unchanged K32 draft block")
    p.add_argument("--last-layers", default=4, type=int, choices=(1, 4))
    p.add_argument("--learning-rate", default=1e-5, type=float)
    p.add_argument("--objective", default="tv", choices=("tv", "kl"))
    p.add_argument("--focus-weight", default=1, type=int,
                   help="TV weight on reached candidate offsets 2 and 3; one keeps the ordinary objective")
    p.add_argument("--max-updates", type=int, help="cap Adam steps during the measured online stream")
    p.add_argument("--continuous-updates", action="store_true",
                   help="require periodic learning through the entire stream; reject a step cap")
    p.add_argument("--replay-blocks", default=1, type=int)
    p.add_argument("--publish-every", default=1, type=int)
    p.add_argument("--ar-check-requests", type=int,
                   help="limit the AR correctness sample; online and frozen are still matched in full")
    p.add_argument("--replay-learned-head", action="store_true",
                   help="replay the full request stream with the final published head and updates disabled")
    p.add_argument("--gather-feedback", action="store_true", help="pack paged training history on the NPU")
    p.add_argument("--temperature", default=0, type=float, choices=(0, 1))
    p.add_argument("--seed", default=743, type=int)
    p.add_argument("--background", action="store_true", help="train on the learner while serving")
    p.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    try:
        policy = update_policy(args.max_updates, require_continuous=args.continuous_updates)
    except ValueError as error:
        p.error(str(error))
    if args.cold_head is not None and args.checkpoint is not None:
        p.error("cold fallback uses fresh Adam for both starts")
    strides = [int(value) for value in args.strides.split(",")]
    if (min(args.replay_blocks, args.publish_every) < 1 or len(set(strides)) != len(strides)
            or any(n < 2 * args.replay_blocks for n in strides) or args.tokens < 2 or args.output.exists()):
        p.error("use distinct strides >= twice replay-blocks, positive counts and a fresh result path")
    prompts, holdout = load_pairs(args.prompts), load_pairs(args.holdout)
    ar_requests = 2 * len(prompts) if args.ar_check_requests is None else args.ar_check_requests
    if not 2 <= ar_requests <= 2 * len(prompts) or ar_requests % 2:
        p.error("AR sample must contain an even number of requests within the stream")
    if {q.strip() for pair in prompts for q in pair} & {q.strip() for pair in holdout for q in pair}:
        p.error("request stream and frozen holdout must have distinct questions")

    from pypto.runtime import configure_log
    from serving.distributed import SamplingParams, ServingConfig
    from serving.engine import InferenceEngine

    from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    configure_log("warn")
    trainer = TP2Online(args.checkpoint, device=args.learner_device, last_layers=args.last_layers,
                        learning_rate=args.learning_rate, background=args.background,
                        replay_blocks=args.replay_blocks, publish_every=args.publish_every,
                        gather_feedback=args.gather_feedback, objective=args.objective, max_updates=args.max_updates,
                        focus_weight=args.focus_weight)
    config = ServingConfig(args.model, (0, 1), max_model_len=512, max_num_batched_tokens=128,
                           max_num_seqs=2, speculative_tokens=0, prefix_cache=False,
                           long_prefill_token_threshold=4)
    engine = InferenceEngine(config)
    runner = engine.runner = OrthrusTP2Runner(config, args.head, draft_batch_size=2,
                                              online=trainer, kv_pages=args.kv_pages,
                                              stochastic=bool(args.temperature), on_cache_event=engine._on_cache)
    engine.scheduler.runner = runner
    summary = dict(completed=False, head_step=runner.head["step"], device_ids=list(config.devices),
                   head_payload_sha256=runner.head["source_payload_sha256"],
                   head_shard_sha256=[shard["sha256"] for shard in runner.head["rank_files"]],
                   visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                   learner_device=args.learner_device, last_layers=args.last_layers,
                   physical_device_count=2 if trainer.colocated else 3, kv_pages=runner.pool_blocks,
                   learning_rate=args.learning_rate, background=args.background,
                   objective=args.objective, max_updates=args.max_updates,
                   focus_weight=args.focus_weight,
                   online_update_policy=policy,
                   temperature=args.temperature, seed=args.seed,
                   sampler="host_full_vocab" if args.temperature else "device_greedy",
                   replay_blocks=args.replay_blocks, publish_every=args.publish_every,
                   stop_on_eos=args.stop_on_eos, candidates=args.candidates,
                   requests=2 * len(prompts), ar_checked_requests=ar_requests, output_tokens_per_request=args.tokens,
                   prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                   holdout_sha256=hashlib.sha256(args.holdout.read_bytes()).hexdigest(),
                   gather_feedback=args.gather_feedback,
                   optimizer_init="checkpoint" if args.checkpoint is not None else "fresh", results={})

    def persist():
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
            row["learner"] = dict(rounds=trainer.rounds, optimizer_version=trainer.version,
                                  completed_version=trainer.completed_version,
                                  published_version=trainer.published_version,
                                  update_limit_reached=trainer.update_limit_reached(),
                                  pending_update=trainer.pending_update is not None)
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
        print("starting TP2 + learner", flush=True)
        engine.start()
        summary["startup_seconds"] = time.monotonic() - start
        print("ready TP2 + learner", summary["startup_seconds"], flush=True)
        runner.output_candidates = 0
        issue(prompts[0], args.tokens)
        ar = measure(prompts[:ar_requests // 2], label="ar")
        holdout_ar = measure(holdout, label="ar_holdout")
        summary["results"]["ar"] = dict(stream=summarize(ar), holdout=summarize(holdout_ar),
                                         request_timeline=request_timeline(ar))
        persist()

        def suite(results, suite_strides):
            frozen, holdout_frozen = measure(prompts, label=f"frozen_{runner.head['step']}"), measure(holdout)
            check_tokens(frozen[:len(ar)], ar)
            check_tokens(holdout_frozen, holdout_ar)
            results["frozen"] = dict(stream=summarize(frozen), holdout=summarize(holdout_frozen),
                                     request_timeline=request_timeline(frozen))
            persist()
            print("frozen", runner.head["step"], results["frozen"], flush=True)

            for stride in suite_strides:
                trainer.pending_reset = True
                issue(prompts[0], 1)
                if trainer.version != trainer.initial_step:
                    raise RuntimeError("online case did not restore the shared initial step")
                first = len(trainer.records)
                busy_start = runner.spec_stats["busy_updates"]
                trainer.stride = stride
                stream = measure(prompts, label=f"online_{runner.head['step']}_{stride}", frozen=frozen)
                trainer.stride = None
                tail = None
                if trainer.pending_update is not None:
                    trainer.pending_drain = True
                    tail = issue(prompts[0], 1)
                updates = trainer.records[first:]
                skipped = runner.spec_stats["busy_updates"] - busy_start
                if len(updates) + skipped != trainer.rounds // stride or not updates:
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
                              supervised_positions=sum(step["supervised"] for step in updates),
                              paired_seconds=[[base["seconds"], trial["seconds"]]
                                              for base, trial in zip(frozen, stream, strict=True)],
                              published_version=final_version, optimizer_version=optimizer_version)
                result["update_schedule"] = dict(
                    policy=policy, stride=stride, max_updates=args.max_updates,
                    rounds=trainer.rounds, scheduled_updates=trainer.rounds // stride,
                    completed_updates=len(updates), skipped_busy=skipped,
                    last_update_round=updates[-1]["round"],
                    rounds_after_last_update=trainer.rounds - updates[-1]["round"],
                    update_limit_reached=trainer.update_limit_reached(),
                )
                result["request_timeline"] = request_timeline(stream)
                if args.continuous_updates and result["update_schedule"]["update_limit_reached"]:
                    raise RuntimeError("continuous run reached an update limit")
                result["net_increment"] = result["stream"]["tok_s"] / results["frozen"]["stream"]["tok_s"] - 1
                results[str(stride)] = result
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
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        if engine.scheduler.thread.ident is not None:
            engine.stop()
            summary["clean_shutdown"] = True
        persist()


if __name__ == "__main__":
    main()
