"""Paired public-weight suffix continuation, including online replay and backward."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from blockspec.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.audit import AuditSampler, SuffixAudit, audit_summary, paired_audit_intervals
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.weights import file_sha256, load_public
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor
from blockspec.measurement import compare, parameter_digest
from .evaluate import prompt_ids, prompt_texts


def argument_parser(*, loss_choices=("tv", "forward_kl")):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--learning-prompts", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--shuffle-requests", action="store_true")
    parser.add_argument("--learn-requests", type=int, default=16)
    parser.add_argument("--learn-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--empty-system", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.)
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--replay-blocks", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--loss", choices=loss_choices, default=loss_choices[0])
    parser.add_argument("--audit-requests", type=int, default=0)
    parser.add_argument("--audit-tokens", type=int, default=128)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_experiment(args, *, config=None, learner_factory=SuffixLearner, feedback_factory=None):
    """Paired controls sharing weights, sampling, setup and request accounting."""
    parser = argument_parser()
    if (args.output.exists() or min(args.requests, args.tokens, args.repeats, args.learn_requests,
                                   args.learn_tokens, args.audit_tokens) < 1 or args.block_size < 2
            or args.prompt_offset < 0 or not 0 <= args.audit_requests <= args.requests
            or (args.audit_only and args.audit_requests == 0)):
        parser.error("new output, positive request budgets and block >= 2 required")
    config = (SuffixConfig(args.last_layers, args.stride, args.replay_blocks, args.learning_rate, loss=args.loss)
              if config is None else config)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    if set(prompt_texts(args.prompts, args.requests, offset=args.prompt_offset)) & set(
            prompt_texts(args.learning_prompts, args.learn_requests)):
        parser.error("learning and evaluation questions overlap")
    torch.set_num_threads(1)
    model = load_public(args.model, device="cuda", dtype=torch.bfloat16)
    before = parameter_digest(model)
    branch = MaskedAttentionBranch(model)
    prompts = prompt_ids(args.model, args.requests, path=args.prompts, thinking=args.thinking,
                         offset=args.prompt_offset, empty_system=args.empty_system)
    training = prompt_ids(args.model, args.learn_requests, path=args.learning_prompts,
                          thinking=args.thinking, empty_system=args.empty_system)
    executor = (SamplingExecutor(model.config.vocab_size, args.block_size, sampling, use_cuda_graph=False) if args.temperature > 0 else None)
    sampler = ProposalSampler(sampling, executor=executor)

    def run(method, prompt, budget, seed, owner=None, sampler_override=None):
        feedback = (None if owner is None else OnlineFeedback(learner=owner) if feedback_factory is None
                    else feedback_factory(learner=owner, output_budget=budget))
        options = {"sampling": sampling, "eos_id": model.config.eos_token_id, "sampler": sampler_override or sampler,
                   "generator": torch.Generator(device="cuda").manual_seed(seed)}
        output = (generate_ar(branch, prompt, budget, **options) if method == "ar" else
                  generate(branch, prompt, budget, block_size=args.block_size,
                           feedback=feedback, **options))
        return output.summary() | {"token_ids": output.tokens}

    for method in ("ar", "static"):
        run(method, training[0], 32, args.seed)
    started = time.perf_counter()
    continued = learner_factory(model, config)
    initial_state = continued.state_dict()
    original = {name: p.detach().clone() for name, p in continued.execution.items()}
    learner_setup = time.perf_counter() - started
    with torch.no_grad():
        ar_probe = model(prompts[0], logits_to_keep=1)
    learning_records = []
    for index, prompt in enumerate(training):
        row = run("learning", prompt, args.learn_tokens, args.seed + 10000 + index, continued)
        learning_records.append(row)
        print(report.dumps({"learning_request": index + 1, "updates": continued.updates,
                          "last_loss": continued.last_loss}), flush=True)
    learned_state = continued.state_dict()
    learned_weights = {name: p.detach().clone() for name, p in continued.execution.items()}
    changed = any(not torch.equal(original[name], p) for name, p in learned_weights.items())
    with torch.no_grad():
        ar_after = model(prompts[0], logits_to_keep=1)
    ar_equal = torch.equal(ar_probe.logits, ar_after.logits) and all(
        torch.equal(a, b) for old, new in zip(ar_probe.cache, ar_after.cache) for a, b in zip(old, new))
    del ar_probe, ar_after
    started = time.perf_counter()
    online = learner_factory(model, config)
    learner_setup += time.perf_counter() - started
    methods = ["ar", "static", "online", "learned", "continued"]
    owners = {"online": online, "continued": continued}
    records, streams = [], []
    state_switch_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    for repeat in range(0 if args.audit_only else args.repeats):
        online.load_state_dict(initial_state)
        continued.load_state_dict(learned_state)
        order = (np.random.default_rng(args.seed + repeat).permutation(len(prompts))
                 if args.shuffle_requests else range(len(prompts)))
        for completed, index in enumerate(order):
            index, prompt = int(index), prompts[int(index)]
            offset = (index + repeat) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    if method in owners:
                        owners[method].publish()
                    else:
                        weights = learned_weights if method == "learned" else original
                        for name, value in weights.items():
                            continued.execution[name].copy_(value)
                torch.cuda.synchronize()
                state_switch_seconds += time.perf_counter() - started
                row = run(method, prompt, args.tokens, args.seed + repeat * 1000 + index, owners.get(method))
                row.update(method=method, request=index, source_request=index + args.prompt_offset,
                           repeat=repeat, input_tokens=prompt.numel())
                records.append(row)
            print(report.dumps({"repeat": repeat, "requests_complete": completed + 1}), flush=True)
        stream_tps = {method: sum(r["tokens"] for r in records if r["repeat"] == repeat and r["method"] == method)
                      / sum(r["seconds"] for r in records if r["repeat"] == repeat and r["method"] == method)
                      for method in methods}
        streams.append({"repeat": repeat, "tps": stream_tps,
                        "online_vs_static": stream_tps["online"] / stream_tps["static"],
                        "continued_vs_learned": stream_tps["continued"] / stream_tps["learned"],
                        "learners": {name: {"updates": owner.updates, "last_loss": owner.last_loss,
                                           "version": owner.version} for name, owner in owners.items()}})
    aggregate = {}
    for method in methods if records else ():
        rows = [r for r in records if r["method"] == method]
        tokens, seconds = sum(r["tokens"] for r in rows), sum(r["seconds"] for r in rows)
        aggregate[method] = {"tokens": tokens, "seconds": seconds, "tps": tokens / seconds,
                             "decode_tpf": (tokens - len(rows)) / sum(r["decode_forwards"] for r in rows),
                             "updates": sum(r["updates"] for r in rows),
                             "update_seconds": sum(r["update_seconds"] for r in rows)}
    rng = np.random.default_rng(args.seed)
    comparisons = ({f"{a}_vs_{b}": compare(records, a, b, args.requests, rng) for a, b in (
        ("static", "ar"), ("online", "static"), ("learned", "static"), ("continued", "static"), ("continued", "learned"))}
        if records else {})
    with torch.no_grad():
        for name, value in original.items():
            continued.execution[name].copy_(value)
    peak = torch.cuda.max_memory_allocated()
    audit_records, combined_audit, audit_seconds = [], None, 0.
    for index, prompt in enumerate(prompts[:args.audit_requests]):
        audit_sampler = AuditSampler(sampling, executor=executor)
        inspector = SuffixAudit(model, continued.capture_layer, learned_weights, args.block_size, sampling,
                                recorded_logits=lambda owner=audit_sampler: owner.last_logits)
        row = run("audit", prompt, args.audit_tokens, args.seed + 20000 + index, inspector, audit_sampler)
        audit_seconds += row["seconds"]
        values = inspector.totals.cpu()
        combined_audit = values.clone() if combined_audit is None else combined_audit + values
        audit_records.append({"request": index + args.prompt_offset,
                              "max_replay_logit_error": inspector.max_replay_logit_error, **audit_summary(values)})
        print(report.dumps({"audit_requests_complete": index + 1}), flush=True)
    restored = parameter_digest(model) == before
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters())
    result = {"script_sha256": file_sha256(__file__),
              "torch": str(torch.__version__), "device": torch.cuda.get_device_name(),
              "sampling": asdict(sampling), "learner": asdict(config),
              "config": {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
              "prompt_sha256": file_sha256(args.prompts), "learning_prompt_sha256": file_sha256(args.learning_prompts),
              "trainable_parameters": continued.trainable_parameters,
              "learner_setup_seconds": learner_setup,
              "sampling_setup_seconds": 0. if executor is None else executor.setup_seconds,
              "comparison_state_switch_seconds": state_switch_seconds,
              "learning": {"tokens": sum(r["tokens"] for r in learning_records),
                           "seconds": sum(r["seconds"] for r in learning_records),
                           "updates": learned_state["updates"], "update_seconds": learned_state["update_seconds"],
                           "last_loss": learned_state["last_loss"]},
              "aggregate": aggregate, "comparisons": comparisons, "streams": streams,
              "common_prefix_audit": {"seconds": audit_seconds, "records": audit_records,
                                      "summary": None if combined_audit is None else audit_summary(combined_audit),
                                      "learned_minus_original": (paired_audit_intervals(audit_records, seed=args.seed)
                                                                 if any(r["positions"] for r in audit_records) else None)},
              "ar_logits_and_kv_unchanged": ar_equal, "learned_execution_weights_changed": changed,
              "original_restored": restored, "inference_parameters_frozen": frozen,
              "frozen_fingerprint": continued.frozen_fingerprint,
              "resident_bytes": resident, "peak_allocated_bytes": peak,
              "records": records, "pass": ar_equal and changed and restored and frozen}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    displayed = {key: value for key, value in result.items() if key not in ("records", "common_prefix_audit")}
    if combined_audit is not None:
        displayed["audit_mean"] = result["common_prefix_audit"]["summary"]["mean"]
        displayed["audit_learned_minus_original"] = result["common_prefix_audit"]["learned_minus_original"]
    print(report.dumps(displayed, indent=2))
    if not result["pass"]:
        raise SystemExit(1)

    return result


def main():
    run_experiment(argument_parser().parse_args())


if __name__ == "__main__":
    main()
