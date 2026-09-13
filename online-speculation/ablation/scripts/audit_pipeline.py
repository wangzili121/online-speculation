"""Paired public-adapter regression against a named pre-refactor Git revision."""

import argparse

from blockspec import reporting as report
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

from blockspec_ablation.adapter_io import load_peft_adapter, peft_config
from blockspec_ablation.benchmark import continuation_prompts
from blockspec_ablation.checkpoint import (adapter_fingerprint, base_fingerprint,
                                  implementation_fingerprint, load_hf_base)
from blockspec.data import load_sequences
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec.parallel.weights import file_sha256
from blockspec.sampling import SamplingConfig
from blockspec_ablation.sampling_execution import SamplingExecutor


def reference_decoder(revision, root):
    resolved = subprocess.run(["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root,
                              check=True, text=True, capture_output=True).stdout.strip()
    source = subprocess.run(["git", "show", f"{resolved}:online-speculation/src/blockspec/decoding.py"],
                            cwd=root, check=True, text=True, capture_output=True).stdout
    for name in ("sampling", "state", "feedback", "data", "parallel.generation"):
        source = source.replace(f"from .{name} import", f"from blockspec.{name} import")
    module = types.ModuleType("blockspec_ablation._migration_reference")
    module.__package__ = "blockspec_ablation"
    sys.modules[module.__name__] = module
    exec(compile(source, f"git:{resolved}:decoding.py", "exec"), module.__dict__)
    return module, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True, help="caller-selected local Git ref")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.requests, args.prompt_length, args.tokens, args.repeats) < 1:
        parser.error("positive budgets and a new output file required")
    root = Path(__file__).resolve().parents[3]
    source, commit = reference_decoder(args.reference_commit, root)
    torch.set_num_threads(4)
    config = peft_config(args.adapter)
    model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device="cuda",
                         dtype=torch.bfloat16).eval().requires_grad_(False)
    load_peft_adapter(args.adapter, model)
    model.set_attention_backend("grouped")
    before = base_fingerprint(model), adapter_fingerprint(model)
    prompts = continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                   count=args.requests, length=args.prompt_length)
    executor = FixedShapeExecutor(model, capacity=args.prompt_length + max(args.tokens, 32),
                                   max_query=args.block_size)
    executor.prepare([(length, False, None) for length in range(1, args.block_size + 1)]
                     + [(length, True, None) for length in range(2, args.block_size + 1)])
    sampling = SamplingConfig(1., 50, .95)
    sampler = SamplingExecutor(model.config.vocab_size, args.block_size, sampling, temperatures=())
    methods = {"old_ar": source.generate_ar, "new_ar": generate_ar,
               "old_static": source.generate_speculative, "new_static": generate_speculative}

    def run(method, prompt, count, seed):
        options = {"executor": executor, "sampler_executor": sampler, "sampling": sampling,
                   "generator": torch.Generator(device="cuda").manual_seed(seed)}
        if method.endswith("static"):
            options["block_size"] = args.block_size
        return methods[method](model, prompt.cuda(), count, **options)

    for method in methods:
        run(method, prompts[0], 32, args.seed)
    records = []
    identical = True
    for repeat in range(args.repeats):
        for index, prompt in enumerate(prompts):
            order = list(methods) if (repeat + index) % 2 == 0 else list(reversed(methods))
            results = {}
            for method in order:
                result = run(method, prompt, args.tokens, args.seed + repeat * args.requests + index)
                results[method] = result
                records.append(result.summary() | {"method": method, "repeat": repeat, "request": index})
            for mode in ("ar", "static"):
                old, new = results["old_" + mode], results["new_" + mode]
                identical &= (old.tokens == new.tokens and old.decode_forwards == new.decode_forwards
                              and old.accepted_per_round == new.accepted_per_round)
    aggregate = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        total, seconds = sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)
        aggregate[method] = {"tokens": total, "seconds": seconds, "tps": total / seconds}
    random = np.random.default_rng(args.seed)
    for mode in ("ar", "static"):
        values = np.asarray([[sum(row["seconds"] for row in records if row["request"] == index
                                 and row["method"] == version + "_" + mode)
                              for version in ("old", "new")] for index in range(args.requests)])
        draws = random.integers(0, len(values), size=(2000, len(values)))
        summed = values[draws].sum(1)
        aggregate["new_" + mode]["paired_ratio_ci95"] = np.quantile(summed[:, 0] / summed[:, 1], [.025, .975]).tolist()
        aggregate["new_" + mode]["relative_to_old"] = aggregate["new_" + mode]["tps"] / aggregate["old_" + mode]["tps"]
    unchanged = before == (base_fingerprint(model), adapter_fingerprint(model))
    result = {"reference_commit": commit, "implementation": implementation_fingerprint(),
              "script_sha256": file_sha256(__file__), "data_sha256": file_sha256(args.data),
              "sampling": {"temperature": 1, "top_k": 50, "top_p": .95},
              "dtype": "bfloat16", "backend": "grouped+cuda_graph", "config": vars(args) | {
                  "base": str(args.base), "adapter": str(args.adapter), "data": str(args.data), "output": str(args.output)},
              "base_fingerprint": before[0], "adapter_fingerprint": before[1],
              "all_outputs_and_counts_identical": identical, "weights_unchanged": unchanged,
              "aggregate": aggregate, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if not identical or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
