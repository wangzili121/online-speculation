"""Paired execution and online-calibration controls for a frozen dual-view model."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
from pathlib import Path
from blockspec.measurement import compare, parameter_digest
import time

import numpy as np
import torch

from blockspec_ablation.calibration import OverlapMix
from blockspec_ablation.checkpoint import implementation_fingerprint
from blockspec_ablation.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec_ablation.parallel.feedback import OnlineFeedback
from blockspec_ablation.parallel.sampling import ProposalSampler
from blockspec.parallel.weights import file_sha256, load_public
from blockspec.sampling import SamplingConfig
from blockspec_ablation.sampling_execution import SamplingExecutor
from blockspec.commands.evaluate import prompt_ids, prompt_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--learning-prompts", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--learn-requests", type=int, default=16)
    parser.add_argument("--learn-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=.8)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or min(args.requests, args.repeats, args.tokens, args.learn_requests,
                                   args.learn_tokens, args.top_k, args.interval) < 1
            or args.block_size < 2 or args.temperature <= 0):
        parser.error("new output, positive sampling/budgets and block >= 2 required")
    evaluation_texts = prompt_texts(args.prompts, args.requests)
    learning_texts = prompt_texts(args.learning_prompts, args.learn_requests)
    if set(evaluation_texts) & set(learning_texts):
        parser.error("learning and evaluation questions overlap")
    torch.set_num_threads(1)
    model = load_public(args.model, device="cuda", dtype=torch.bfloat16)
    before = parameter_digest(model)
    branch = MaskedAttentionBranch(model)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    prompts = prompt_ids(args.model, args.requests, path=args.prompts, thinking=args.thinking)
    training = prompt_ids(args.model, args.learn_requests, path=args.learning_prompts, thinking=args.thinking)
    engines = {"eager": SamplingExecutor(model.config.vocab_size, args.block_size, sampling, temperatures=(),
                                         use_cuda_graph=False, protected_rows=0),
               "graph": SamplingExecutor(model.config.vocab_size, args.block_size, sampling, protected_rows=0)}

    def mixer(adaptive):
        return OverlapMix(args.block_size, args.top_k, device="cuda", protected_rows=0,
                          interval=args.interval, learning_rate=args.learning_rate, adaptive=adaptive)

    def run(method, prompt, budget, seed, mix=None):
        engine = engines["eager" if method.endswith("eager") else "graph"]
        engine.validate(model, sampling, args.block_size, mix)
        sampler = ProposalSampler(sampling, executor=engine, calibrator=mix)
        options = {"sampling": sampling, "eos_id": model.config.eos_token_id, "sampler": sampler,
                   "generator": torch.Generator(device="cuda").manual_seed(seed)}
        output = (generate_ar(branch, prompt, budget, **options) if method.startswith("ar_") else
                  generate(branch, prompt, budget, block_size=args.block_size,
                           feedback=None if mix is None else OnlineFeedback(calibrator=mix), **options))
        return output.summary() | {"token_ids": output.tokens}

    methods = ["ar_eager", "ar_graph", "static_eager", "static_graph", "identity_graph", "online_graph",
               "learned_graph", "continued_graph"]
    for method in methods[:5]:
        run(method, training[0], 32, args.seed, mixer(False) if method == "identity_graph" else None)
    learned = mixer(True)
    learning_records = []
    for index, prompt in enumerate(training):
        learning_records.append(run("learning_graph", prompt, args.learn_tokens, args.seed + 10000 + index, learned))
    learned_state = learned.state_dict()
    records, final_mixtures = [], []
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    started = time.perf_counter()
    for repeat in range(args.repeats):
        mixes = {"identity_graph": mixer(False), "online_graph": mixer(True),
                 "learned_graph": mixer(False), "continued_graph": mixer(True)}
        for method in ("learned_graph", "continued_graph"):
            mixes[method].load_state_dict(learned_state)
        for index, prompt in enumerate(prompts):
            offset = (index + repeat) % len(methods)
            ordered = methods[offset:] + methods[:offset]
            for method in ordered:
                row = run(method, prompt, args.tokens, args.seed + repeat * 1000 + index, mixes.get(method))
                row.update(method=method, request=index, repeat=repeat, input_tokens=prompt.numel())
                records.append(row)
            print(report.dumps({"repeat": repeat, "requests_complete": index + 1,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        final_mixtures.append({name: mix.metrics() for name, mix in mixes.items()})
    aggregate = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        total, seconds = sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)
        forwards = sum(row["decode_forwards"] for row in rows)
        aggregate[method] = {"tokens": total, "seconds": seconds, "tps": total / seconds,
                             "decode_tpf": (total - len(rows)) / forwards,
                             "updates": sum(row["updates"] for row in rows),
                             "update_seconds": sum(row["update_seconds"] for row in rows)}
    rng = np.random.default_rng(args.seed)
    comparisons = {f"{a}_vs_{b}": compare(records, a, b, args.requests, rng) for a, b in (
        ("ar_graph", "ar_eager"), ("static_graph", "static_eager"), ("static_graph", "ar_graph"),
        ("identity_graph", "static_graph"), ("online_graph", "static_graph"),
        ("learned_graph", "static_graph"), ("continued_graph", "static_graph"), ("continued_graph", "learned_graph"))}
    identical = True
    for repeat in range(args.repeats):
        for index in range(args.requests):
            rows = {row["method"]: row for row in records if row["repeat"] == repeat and row["request"] == index}
            for left, right in (("ar_eager", "ar_graph"), ("static_eager", "static_graph"), ("static_graph", "identity_graph")):
                identical &= all(rows[left][key] == rows[right][key]
                                 for key in ("token_ids", "accepted_per_round", "decode_forwards"))
    frozen = before == parameter_digest(model) and all(not p.requires_grad and p.grad is None for p in model.parameters())
    result = {"parameter_digest": before,
              "implementation": implementation_fingerprint(), "script_sha256": file_sha256(__file__),
              "torch": str(torch.__version__), "device": torch.cuda.get_device_name(),
              "prompt_sha256": file_sha256(args.prompts), "learning_prompt_sha256": file_sha256(args.learning_prompts),
              "sampling": asdict(sampling), "config": {key: str(value) if isinstance(value, Path) else value
                                                         for key, value in vars(args).items()},
              "setup_seconds": {key: value.setup_seconds for key, value in engines.items()},
              "learning": {"tokens": sum(row["tokens"] for row in learning_records),
                           "seconds": sum(row["seconds"] for row in learning_records), "state": learned.metrics()},
              "aggregate": aggregate, "comparisons": comparisons, "final_mixtures": final_mixtures,
              "execution_and_identity_outputs_equal": identical, "all_parameters_frozen": frozen,
              "resident_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "records": records, "pass": identical and frozen}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps({key: value for key, value in result.items() if key not in ("records", "final_mixtures")}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
