"""Cold-start serving curve with a measured time budget and fixed held-out probes."""

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import time

import numpy as np
import torch

from blockspec import reporting
from blockspec.commands.evaluate import prompt_ids, prompt_texts
from blockspec.measurement import compare
from blockspec.parallel.fitting import FitConfig, frozen_fingerprint
from blockspec.parallel.online import portable
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.training import distillation_loss, sample_anchors
from blockspec.parallel.weights import load_ar_base, load_public
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor
from blockspec_ablation.cold_start import CleanReplay, ColdStartService, FullBlockLearner, ServiceConfig, synchronize


def load_cold_model(path, *, ar_base=False, block_size=None, mask_token_id=None,
                    device="cuda", dtype=torch.bfloat16):
    if ar_base:
        if block_size is None or mask_token_id is None:
            raise ValueError("AR initialization requires a block size and a mask token")
        model = load_ar_base(path, block_size=block_size, mask_token_id=mask_token_id, device=device).to(dtype=dtype)
        model.frequencies = model._frequencies(model.embedding.weight.device)
    else:
        model = load_public(path, device=device, dtype=dtype)
        if block_size is not None:
            model.config = replace(model.config, block_size=block_size)
    return model.set_backend("sdpa")


@torch.no_grad()
def measure(service, prompts, tokens, *, seed, fixed_batches=None):
    """Research measurements: separate from delivered requests and gate decisions."""
    records, clean = [], []
    for index, prompt in enumerate(prompts):
        for name in (("ar", "candidate") if index % 2 == 0 else ("candidate", "ar")):
            if name == "candidate":
                with service.learner.candidate():
                    result = service._generate(prompt, tokens, speculative=True, seed=seed + index)
            else:
                result = service._generate(prompt, tokens, speculative=False, seed=seed + index)
                if fixed_batches is None:
                    sequence = torch.cat((prompt.cpu(), torch.tensor([result.tokens])), 1)
                    sequence = sequence[:, -service.fit.sequence_length:].long()
                    if sequence.shape[1] >= service.model.config.block_size:
                        rng = torch.Generator().manual_seed(seed + index + 1000)
                        anchors = sample_anchors(sequence, service.model.config.block_size,
                                                 service.fit.anchors_per_sequence, generator=rng)
                        clean.append((sequence, anchors))
            records.append({"request": index, "method": name, **result.summary()})
    batches = clean if fixed_batches is None else fixed_batches
    losses = []
    with service.learner.candidate(), service.learner.autocast():
        for sequence, anchors in batches:
            losses.append(float(distillation_loss(service.model, sequence.to(service.learner.device),
                                                  anchors.to(service.learner.device), chunk_rows=service.fit.chunk_rows)))
    metrics = {"step": service.learner.steps, "validation_kl": sum(losses) / len(losses) if losses else None,
               "candidate_over_ar": compare(records, "candidate", "ar", len(prompts), np.random.default_rng(seed))}
    for name in ("ar", "candidate"):
        selected = [row for row in records if row["method"] == name]
        count = sum(row["tokens"] for row in selected)
        elapsed = sum(row["seconds"] for row in selected)
        forwards = sum(row["decode_forwards"] for row in selected)
        metrics[name] = {"tokens": count, "seconds": elapsed, "tps": count / elapsed,
                         "decode_tpf": (count - sum(row["prefill_output_tokens"] for row in selected)) / forwards}
    return metrics, batches


def fit_complete_buffer(model, fit, records, steps, *, deterministic=False):
    """Matched-update offline control with access to the final delivered data."""
    if not records or type(steps) is not int or not 0 <= steps <= fit.steps:
        raise ValueError("completed records and a valid matched update count required")
    synchronize(model)
    start = time.perf_counter()
    learner = FullBlockLearner(model, fit, deterministic=deterministic)
    replay = CleanReplay(len(records), fit.sequence_length, model.config.block_size, fit.seed)
    for record in records:
        replay.append(record)
    for _ in range(steps):
        learner.step(replay.batch(fit.accumulate, fit.anchors_per_sequence, batch_size=fit.batch_size))
    synchronize(model)
    return learner, {"steps": steps, "completed_records": len(replay.records),
                     "supervised_rows": (steps * fit.accumulate * fit.batch_size * fit.anchors_per_sequence
                                         * (model.config.block_size - 1)),
                     "seconds": time.perf_counter() - start}


def paired_stream(summary, references, *, prior_tokens=0, prior_generation=0., prior_extra=0.):
    """Current-run net speed includes newly incurred setup/restoration costs."""
    tokens = summary["delivered_tokens"] - prior_tokens
    seconds = (summary["budget"]["service_seconds"] - prior_generation
               + summary["budget"]["extra_seconds"] - prior_extra)
    ar_tokens = sum(row["tokens"] for row in references)
    ar_seconds = sum(row["seconds"] for row in references)
    ar_tps = ar_tokens / ar_seconds
    return {"requests": len(references), "tokens": tokens, "seconds": seconds,
            "net_tps": tokens / seconds, "ar_tokens": ar_tokens, "ar_seconds": ar_seconds,
            "ar_tps": ar_tps, "net_over_ar": (tokens / seconds) / ar_tps}


def stream_comparison(streams, *, seed, prior_tokens=0, prior_extra=0.):
    """Paired interval conditional on the realized learning/publication trajectory."""
    pairs = []
    previous_tokens, previous_extra = prior_tokens, prior_extra
    for index, row in enumerate(streams):
        tokens, extra = row["delivered_tokens"], row["budget"]["extra_seconds"]
        pairs.extend(({"request": index, "method": "online", "tokens": tokens - previous_tokens,
                       "seconds": row["service_seconds"] + extra - previous_extra},
                      {"request": index, "method": "ar", **row["ar_reference"]}))
        previous_tokens, previous_extra = tokens, extra
    return compare(pairs, "online", "ar", len(streams), np.random.default_rng(seed))


def run(args):
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    fit = FitConfig(steps=args.steps, warmup_steps=args.warmup_steps, sequence_length=args.sequence_length,
                    anchors_per_sequence=args.anchors, accumulate=args.accumulate, learning_rate=args.learning_rate,
                    chunk_rows=args.chunk_rows, precision="bf16", seed=args.seed,
                    batch_size=args.batch_size, optimizer_impl=args.optimizer_impl)
    settings = ServiceConfig(fraction=args.fraction, replay_records=args.replay_records, probe_every=args.probe_every,
                              probe_tokens=args.probe_tokens, publish_margin=args.publish_margin, seed=args.seed,
                              initial_probe_factor=args.initial_probe_factor, reuse_ar_prefix=args.reuse_ar_prefix,
                              live_probe_requests=args.live_probe_requests, full_answer_screen=args.full_answer_screen,
                              reuse_speculative_answers=args.reuse_speculative_answers,
                              screen_requests=args.screen_requests, screen_margin=args.screen_margin,
                              initial_screen_estimate=args.initial_screen_estimate)
    training_texts = prompt_texts(args.prompts, args.requests, offset=args.offset)
    heldout_texts = prompt_texts(args.heldout_prompts, args.heldout_count, offset=args.heldout_offset)
    gate_texts = prompt_texts(args.heldout_prompts, args.gate_count, offset=args.heldout_offset + args.heldout_count)
    if (set(training_texts) & set(heldout_texts + gate_texts) or set(heldout_texts) & set(gate_texts)):
        raise ValueError("delivered, gate and held-out questions must be disjoint")
    resource = args.base if args.base is not None else args.model
    model = load_cold_model(resource, ar_base=args.base is not None, block_size=args.block_size,
                            mask_token_id=args.mask_token_id)
    prompts = prompt_ids(resource, args.requests, path=args.prompts, offset=args.offset, empty_system=True)
    heldout = prompt_ids(resource, args.heldout_count, path=args.heldout_prompts,
                         offset=args.heldout_offset, empty_system=True)
    gate = prompt_ids(resource, args.gate_count, path=args.heldout_prompts,
                      offset=args.heldout_offset + args.heldout_count, empty_system=True)
    sampling = SamplingConfig(1.)
    executor = SamplingExecutor(model.config.vocab_size, model.config.block_size, sampling, device="cuda")
    sampler = ProposalSampler(sampling, executor=executor)
    # Common model/sampling warm-up belongs to benchmark setup. The service
    # constructor separately charges its cold parameter initialization.
    from blockspec.parallel import MaskedAttentionBranch, generate_ar
    generate_ar(MaskedAttentionBranch(model), prompts[0], 8, sampler=sampler)
    base = frozen_fingerprint(model)
    service = ColdStartService(model, fit, settings, gate_prompts=gate, sampling=sampling,
                               sampler=sampler, retain_batches=args.offline_replay, retain_records=args.offline_control,
                               deterministic_updates=args.deterministic)
    if not all(torch.equal(p, dict(layer.attention.ar.named_parameters())[name])
               for layer in model.layers for name, p in layer.attention.draft.named_parameters()):
        raise AssertionError("cold initialization must equal AR attention")
    prior = dict(prior_tokens=0, prior_generation=0., prior_extra=0.)
    if args.resume is not None:
        start = time.perf_counter()
        state = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        read_seconds = time.perf_counter() - start
        service.load_state_dict(state)
        service.budget.charge("resume_io", read_seconds)
        prior = dict(prior_tokens=state["stream"]["delivered_tokens"],
                     prior_generation=state["stream"]["budget"]["service_seconds"],
                     prior_extra=state["stream"]["budget"]["extra_seconds"])
        del state
    curve, streams, references, research_seconds = [], [], [], 0.
    fixed_batches = None

    def checkpoint_curve():
        nonlocal fixed_batches, research_seconds
        synchronize(model)
        start = time.perf_counter()
        row, fixed_batches = measure(service, heldout, args.heldout_tokens,
                                      seed=args.seed + 200000, fixed_batches=fixed_batches)
        synchronize(model)
        research_seconds += time.perf_counter() - start
        curve.append(row)
        print(reporting.dumps({"curve": row, "service": service.summary()}), flush=True)

    def reference_ar(prompt, request_seed):
        nonlocal research_seconds
        synchronize(model)
        start = time.perf_counter()
        reference = service._generate(prompt, args.tokens, speculative=False, seed=request_seed)
        synchronize(model)
        research_seconds += time.perf_counter() - start
        return reference

    checkpoint_curve()
    for index, prompt in enumerate(prompts):
        ar_result = None
        request_seed = args.seed + args.offset + index
        if service.speculating and index % 2 == 0:
            ar_result = reference_ar(prompt, request_seed)
        delivered, row = service.serve(prompt, args.tokens, seed=request_seed)
        if row["mode"] == "ar":
            reference = {"tokens": len(delivered.tokens), "seconds": row["service_seconds"]}
        else:
            if ar_result is None:
                ar_result = reference_ar(prompt, request_seed)
            reference = {"tokens": len(ar_result.tokens), "seconds": ar_result.seconds}
        references.append(reference)
        row["ar_reference"] = reference
        row["paired_run"] = paired_stream(service.summary(), references, **prior)
        streams.append(row)
        if row["update"] or row["probe"] or (index + 1) % args.log_every == 0:
            print(reporting.dumps(row), flush=True)
        if service.learner.steps and service.learner.steps % args.curve_every == 0 and curve[-1]["step"] != service.learner.steps:
            checkpoint_curve()
    if curve[-1]["step"] != service.learner.steps:
        checkpoint_curve()
    result = {"method": "cold-start-full-block-kl", "initialization": "AR attention copy",
              "deterministic": args.deterministic,
              "block_size": model.config.block_size,
              "fit": asdict(fit), "controller": asdict(settings), "sampling": asdict(sampling),
              "stream": service.summary(), "paired_run": paired_stream(service.summary(), references, **prior),
              "curve": curve, "requests": streams,
              "research_measurement_seconds": research_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "frozen_ar_unchanged": base == frozen_fingerprint(model)}
    result["paired_run"].update(stream_comparison(streams, seed=args.seed,
                                                 prior_tokens=prior["prior_tokens"], prior_extra=prior["prior_extra"]))
    if args.checkpoint is not None:
        # Private continuation state stays outside publication summaries.
        start = time.perf_counter()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = service.state_dict()
        with args.checkpoint.open("xb") as handle:
            torch.save(state, handle)
        result["shutdown_checkpoint_seconds"] = time.perf_counter() - start
        del state
    if args.offline_replay:
        # The same realized microbatches distinguish implementation parity from
        # a separate offline data-access or optimization-budget comparison.
        reference_weights = portable(service.learner.master)
        transcript = service.transcript
        del service.learner.optimizer
        service.learner.master.clear()
        synchronize(model)
        start = time.perf_counter()
        offline = FullBlockLearner(model, fit, deterministic=args.deterministic)
        for batches in transcript:
            offline.step(batches)
        synchronize(model)
        result["offline_replay"] = {"steps": offline.steps, "seconds": time.perf_counter() - start,
                                     "identical_parameters": all(torch.equal(p.cpu(), reference_weights[name])
                                                                 for name, p in offline.master.items()),
                                     "frozen_ar_unchanged": base == frozen_fingerprint(model)}
        service.learner = offline
        del reference_weights
    if args.offline_control:
        steps = service.learner.steps
        del service.learner.optimizer
        service.learner.master.clear()
        service.learner, control = fit_complete_buffer(model, fit, service.completed_records, steps,
                                                       deterministic=args.deterministic)
        start = time.perf_counter()
        control["metric"], _ = measure(service, heldout, args.heldout_tokens,
                                       seed=args.seed + 200000, fixed_batches=fixed_batches)
        synchronize(model)
        control["measurement_seconds"] = time.perf_counter() - start
        control["frozen_ar_unchanged"] = base == frozen_fingerprint(model)
        result["offline_control"] = control
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        reporting.dump(result, handle, indent=2)
    print(reporting.dumps({key: value for key, value in result.items() if key != "requests"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--base", type=Path)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--heldout-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--block-size", type=int)
    for name, default in (("requests", 128), ("offset", 0), ("tokens", 256), ("steps", 64), ("warmup-steps", 4),
                           ("sequence-length", 256), ("anchors", 4), ("batch-size", 1), ("accumulate", 1), ("chunk-rows", 32),
                           ("replay-records", 128), ("probe-every", 8), ("probe-tokens", 32), ("gate-count", 2),
                           ("live-probe-requests", 1), ("screen-requests", 2),
                           ("heldout-count", 4), ("heldout-tokens", 128), ("heldout-offset", 64),
                           ("curve-every", 4), ("log-every", 8), ("seed", 743)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--fraction", type=float, default=.01)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--publish-margin", type=float, default=1.10)
    parser.add_argument("--initial-probe-factor", type=float, default=2.)
    parser.add_argument("--reuse-ar-prefix", action="store_true")
    parser.add_argument("--full-answer-screen", action="store_true")
    parser.add_argument("--reuse-speculative-answers", action="store_true")
    parser.add_argument("--screen-margin", type=float, default=1.02)
    parser.add_argument("--initial-screen-estimate", type=float, default=.4)
    parser.add_argument("--optimizer-impl", choices=("single", "fused"), default="single")
    parser.add_argument("--offline-replay", action="store_true")
    parser.add_argument("--offline-control", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.output.exists() or (args.checkpoint is not None and args.checkpoint.exists()):
        parser.error("new result and checkpoint destinations required")
    if args.resume is not None and (args.offline_replay or args.offline_control):
        parser.error("full cold-start offline comparisons use a fresh learning stream")
    if args.offline_replay and not args.deterministic:
        parser.error("elementwise replay auditing requires --deterministic; use --offline-control for service-quality comparisons")
    if args.block_size is not None and not 2 <= args.block_size <= args.sequence_length:
        parser.error("block size must lie between 2 and the sequence window")
    if args.base is not None and (args.block_size is None or args.mask_token_id is None):
        parser.error("--base requires --block-size and --mask-token-id")
    if args.base is None and args.mask_token_id is not None:
        parser.error("--model supplies its own mask token")
    if any(getattr(args, key) < 1 for key in ("requests", "tokens", "curve_every", "log_every", "heldout_count",
                                               "heldout_tokens", "gate_count")):
        parser.error("positive request, output and measurement sizes required")
    run(args)


if __name__ == "__main__":
    main()
