"""Full-answer work/timing audit against ordinary corrected generation."""

import argparse
import json

import torch

from blockspec import reporting
from blockspec.commands.evaluate import prompt_ids
from blockspec.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.weights import source_identity
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor
from blockspec.state import trim_cache
from blockspec_ablation.cold_experiment import load_cold_model
from blockspec_ablation.virtual_work import estimate_trace, expected_work, inverse_acceptance, score_trace


@torch.no_grad()
def serial_audit(branch, prompt, outputs, cap, eos_id):
    """Compare every used batched row with an ordinary single-anchor draft.

    This audit uses the same clean AR cache on both paths, isolating the masked
    batch's floating-point effects from the inverse-coupling calculation.
    """
    scores = score_trace(branch.model, prompt, outputs, output_budget=cap, eos_id=eos_id)
    clean = torch.cat((prompt, outputs[None]), 1)
    teacher = branch.model(clean, compute_logits=False)
    single = scores.acceptance.clone()
    differences = []
    for j in range(1, outputs.numel() - 1):
        anchor = prompt.shape[1] + j - 1
        length = min(branch.default_block_size, cap - j + 1)
        inputs = prompt.new_full((1, length), branch.model.config.mask_token_id)
        inputs[:, :1] = clean[:, anchor:anchor + 1]
        output = branch.model(inputs, view="draft", cache=trim_cache(teacher.cache, anchor))
        count = min(length - 1, outputs.numel() - j - 1)
        q = output.logits[0, :count].float().log_softmax(-1).gather(1, outputs[j:j + count, None])[:, 0].double()
        single[j - 1, :count] = inverse_acceptance(q, scores.target_logp[j - 1, :count])
        differences.append((q - scores.proposal_logp[j - 1, :count]).abs())
    base = float(scores.work().decode_forwards)
    reference = float(expected_work(single.cpu(), output_tokens=outputs.numel(), output_budget=cap,
                                     block_size=branch.default_block_size).decode_forwards)
    errors = torch.cat(differences) if differences else torch.zeros(1)
    return {"batched_forwards": base, "single_anchor_forwards": reference,
            "relative_forward_error": base / reference - 1 if reference else 0.,
            "max_logp_error": float(errors.max()), "mean_logp_error": float(errors.mean())}


def summarize(records):
    def totals(name):
        rows = [r[name] for r in records]
        tokens, seconds = sum(x["tokens"] for x in rows), sum(x["seconds"] for x in rows)
        forwards = sum(x["decode_forwards"] for x in rows)
        return {"tokens": tokens, "seconds": seconds, "tps": tokens / seconds,
                "decode_forwards": forwards, "decode_tpf": (tokens - len(rows)) / forwards if forwards else None}
    ar, candidate = totals("ar"), totals("candidate")
    tokens = ar["tokens"]
    predicted_seconds = sum(r["predicted_seconds"] for r in records)
    forwards = sum(r["virtual_forwards"] for r in records)
    result = {"requests": len(records), "ar": ar, "candidate": candidate,
            "candidate_over_ar": candidate["tps"] / ar["tps"],
            "predicted_over_ar": (tokens / predicted_seconds) / ar["tps"],
            "virtual_decode_tpf": (tokens - len(records)) / forwards if forwards else None,
            "mean_screen_seconds": sum(r["screen_seconds"] for r in records) / len(records),
            "mean_score_seconds": sum(r["score_seconds"] for r in records) / len(records),
            "mean_calibration_seconds": sum(r["calibration_seconds"] for r in records) / len(records)}
    if all("prefix_seconds" in r for r in records):
        result["mean_prefix_seconds"] = sum(r["prefix_seconds"] for r in records) / len(records)
        result["screen_over_prefix_cost"] = result["mean_screen_seconds"] / result["mean_prefix_seconds"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Cold-start full-attention learning state")
    source.add_argument("--cold-copy", action="store_true")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1211)
    parser.add_argument("--anchors-per-pass", type=int, default=128)
    parser.add_argument("--calibration-points", type=int, default=3)
    parser.add_argument("--serial-audits", type=int, default=0)
    parser.add_argument("--prefix-compare", action="store_true", help="Time a matched 48-token candidate prefix")
    args = parser.parse_args(argv)
    if args.count < 1 or args.tokens < 2 or not 0 <= args.serial_audits <= args.count:
        parser.error("positive request count, at least two output tokens and bounded audit count required")
    torch.set_num_threads(1)
    model = load_cold_model(args.model, block_size=args.block_size)
    if args.checkpoint:
        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
        weights = {name: p for name, p in model.named_parameters() if ".attention.draft." in name}
        saved = state.get("master", {})
        if (state.get("format") != "cold-start-research-v1" or not source_identity(state.get("source", {}))
                or source_identity(state["source"]) != source_identity(model.source)
                or state.get("model_config") != model.config.to_dict() or saved.keys() != weights.keys()
                or any(p.shape != weights[name].shape or not torch.isfinite(p).all() for name, p in saved.items())):
            raise ValueError("matching complete cold-start candidate required")
        with torch.no_grad():
            for name, p in weights.items():
                p.copy_(saved[name])
        del saved, state
    elif args.cold_copy:
        model.initialize_draft_from_ar()
    frozen = [(p, p._version) for p in model.parameters()]
    branch = MaskedAttentionBranch(model)
    sampling = SamplingConfig(1.)
    sampler = ProposalSampler(sampling, executor=SamplingExecutor(model.config.vocab_size, args.block_size,
                                                                 sampling, device="cuda"))
    prompts = prompt_ids(args.model, args.count, path=args.prompts, offset=args.offset, empty_system=True)
    rng = lambda seed: torch.Generator(device="cuda").manual_seed(seed)
    eos = model.config.eos_token_id
    warm = generate_ar(branch, prompts[0], 16, sampler=sampler, generator=rng(args.seed), prefix_tokens=1)
    estimate_trace(branch, prompts[0], prompts[0].new_tensor(warm.tokens), output_budget=16,
                    prefill_seconds=warm.prefix_seconds, sampler=sampler, generator=rng(args.seed + 1),
                    anchors_per_pass=args.anchors_per_pass, calibration_points=args.calibration_points)
    records = []
    for index, prompt in enumerate(prompts):
        results = {}
        for kind in (("ar", "candidate") if index % 2 == 0 else ("candidate", "ar")):
            fn = generate_ar if kind == "ar" else generate
            options = {"prefix_tokens": 1} if kind == "ar" else {}
            results[kind] = fn(branch, prompt, args.tokens, sampler=sampler, generator=rng(args.seed + 11 + index),
                               eos_id=eos, **options)
        ar = results["ar"]
        outputs = prompt.new_tensor(ar.tokens)
        prefix_seconds = None
        checks = (("screen", "prefix") if index % 2 == 0 else ("prefix", "screen")) if args.prefix_compare else ("screen",)
        for kind in checks:
            if kind == "screen":
                estimate = estimate_trace(branch, prompt, outputs, output_budget=args.tokens,
                                           prefill_seconds=ar.prefix_seconds, sampler=sampler, eos_id=eos,
                                           generator=rng(args.seed + 1011 + index), anchors_per_pass=args.anchors_per_pass,
                                           calibration_points=args.calibration_points)
            else:
                prefix_seconds = generate(branch, prompt, min(48, args.tokens), sampler=sampler,
                                            eos_id=eos, generator=rng(args.seed + 11 + index)).seconds
        compact = lambda result: {k: result.summary()[k] for k in ("tokens", "seconds", "decode_forwards", "decode_tpf")}
        row = {"request": index, **{k: compact(v) for k, v in results.items()},
               "virtual_forwards": float(estimate.work.decode_forwards),
               "predicted_seconds": estimate.predicted_seconds, "screen_seconds": estimate.measured_seconds,
               "score_seconds": estimate.score_seconds, "calibration_seconds": estimate.calibration_seconds,
               "round_seconds": estimate.round_seconds}
        if prefix_seconds is not None:
            row["prefix_seconds"] = prefix_seconds
        if index < args.serial_audits:
            row["serial_audit"] = serial_audit(branch, prompt, outputs, args.tokens, eos)
        records.append(row)
        print(reporting.dumps(row), flush=True)
    summary = summarize(records)
    summary["frozen_parameters"] = all(p._version == version for p, version in frozen)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
