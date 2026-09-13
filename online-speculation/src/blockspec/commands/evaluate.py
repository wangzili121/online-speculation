"""Caller-selected weight audit and paired dual-view throughput measurement."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from blockspec.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.weights import file_sha256, load_public
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor
from blockspec.state import cache_length


PROMPTS = (
    "Explain how binary search works. Give a Python implementation and discuss its time complexity.",
    "Solve step by step: A train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed?",
    "Write a Python function that merges two sorted lists. Explain the invariant used in its loop.",
    "Explain the difference between a process and a thread, including memory sharing and synchronization.",
    "Find the sum of the first 100 positive integers and derive the general formula for the first n integers.",
    "Describe the water cycle, including evaporation, condensation, precipitation, and collection.",
    "Write a Python function to check whether brackets in a string are balanced, with examples and an explanation.",
    "Compare breadth-first search with depth-first search, including their data structures and typical uses.",
)


def load_reference(folder, dtype, backend, manifest):
    from blockspec.reference import checked_reference

    checked = checked_reference(folder, manifest)
    entry = checked["entrypoint"]
    source = (Path(folder) / entry["file"]).resolve()
    if source.parent != Path(folder).resolve() or entry["file"] not in checked["reference_lf_sha256"]:
        raise ValueError("reference entrypoint must be a checked local source file")
    spec = importlib.util.spec_from_file_location("blockspec_local_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, entry["class"]).from_pretrained(
        folder, dtype=dtype, attn_implementation=backend, local_files_only=True).cuda().eval().requires_grad_(False)


def prompt_texts(path, count, *, offset=0):
    if type(offset) is not int or offset < 0 or type(count) is not int or count < 1:
        raise ValueError("nonnegative prompt offset and positive count required")
    if path is None:
        texts = list(PROMPTS)
    else:
        texts = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    value = row.get("question", row.get("prompt"))
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("each prompt record needs a nonempty question or prompt string")
                    texts.append(value)
                    if len(texts) == count + offset:
                        break
    if not count + offset <= len(texts):
        raise ValueError("the requested number of prompts must fit the input data")
    return texts[offset:offset + count]


def render_prompt(tokenizer, text, *, thinking=False, empty_system=False):
    messages = ([{"role": "system", "content": ""}] if empty_system else [])
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)


def prompt_ids(folder, count, *, path=None, thinking=False, offset=0, empty_system=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    result = []
    for text in prompt_texts(path, count, offset=offset):
        rendered = render_prompt(tokenizer, text, thinking=thinking, empty_system=empty_system)
        result.append(tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.cuda())
    return result


def compare_logits(actual, reference):
    a, b = actual.float(), reference.float()
    tv = 0.5 * (a.softmax(-1) - b.softmax(-1)).abs().sum(-1)
    return {"max_logit_error": float((a - b).abs().max()),
            "mean_logit_error": float((a - b).abs().mean()),
            "max_tv": float(tv.max()), "mean_tv": float(tv.mean()),
            "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


def compare_cache(actual, reference):
    error, scaled_error = 0.0, 0.0
    for (key, value), layer in zip(actual, reference.layers, strict=True):
        if key.shape != layer.keys.shape or value.shape != layer.values.shape:
            raise AssertionError("reference cache shapes differ")
        tolerance = 0.0002 if key.dtype == torch.float32 else 0.05
        for actual_tensor, expected_tensor in ((key, layer.keys), (value, layer.values)):
            delta = (actual_tensor.float() - expected_tensor.float()).abs()
            error = max(error, float(delta.max()))
            scale = tolerance * (1 + expected_tensor.float().abs())
            scaled_error = max(scaled_error, float((delta / scale).max()))
    return {"max_kv_error": error, "max_kv_tolerance_ratio": scaled_error,
            "kv_atol_rtol": tolerance}


@torch.inference_mode()
def audit(args, own, reference):
    records = []
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    sequence = torch.randint(100, own.config.vocab_size - 1000, (1, 80), device="cuda", generator=rng)
    for prefix_length in (7, 29):
        prefix = sequence[:, :prefix_length]
        for block in args.blocks:
            ar = own(prefix)
            oracle = reference(input_ids=prefix, use_cache=True)
            records.append({"stage": "ar_prefill", "prefix": prefix_length, "block": block,
                            **compare_logits(ar.logits, oracle.logits),
                            **compare_cache(ar.cache, oracle.past_key_values)})
            draft_ids = prefix.new_full((1, block), own.config.mask_token_id)
            draft_ids[:, 0] = sequence[:, prefix_length]
            positions = torch.arange(prefix_length, prefix_length + block, device="cuda")[None]
            before = [(layer.keys.clone(), layer.values.clone()) for layer in oracle.past_key_values.layers]
            draft = own(draft_ids, view="draft", cache=ar.cache, positions=positions)
            oracle_draft = reference(input_ids=draft_ids, past_key_values=oracle.past_key_values,
                                     position_ids=positions, use_cache=False,
                                     is_diffusion_pass=True, ar_seq_len=prefix_length)
            for (key, value), layer in zip(before, oracle.past_key_values.layers, strict=True):
                if not torch.equal(key, layer.keys) or not torch.equal(value, layer.values):
                    raise AssertionError("reference draft mutated the persistent cache")
            records.append({"stage": "draft", "prefix": prefix_length, "block": block,
                            **compare_logits(draft.logits, oracle_draft.logits)})
            candidates = torch.cat((draft_ids[:, :1], draft.logits[:, :-1].argmax(-1)), 1)
            verifier = own(candidates, cache=ar.cache)
            oracle_verifier = reference(input_ids=candidates, past_key_values=oracle.past_key_values,
                                        position_ids=positions, use_cache=True)
            records.append({"stage": "ar_verify", "prefix": prefix_length, "block": block,
                            **compare_logits(verifier.logits, oracle_verifier.logits),
                            **compare_cache(verifier.cache, oracle_verifier.past_key_values)})
    limit = 0.0005 if args.dtype == "float32" else 0.5
    tv_limit = 0.0001 if args.dtype == "float32" else 0.02
    numerical_pass = all(row["max_logit_error"] <= limit and row["max_tv"] <= tv_limit
                         and row.get("max_kv_tolerance_ratio", 0) <= 1 for row in records)
    prompt = prompt_ids(args.model, 1, path=args.prompts, thinking=args.thinking, empty_system=args.empty_system)[0]
    own_ids = generate(MaskedAttentionBranch(own), prompt, args.tokens, block_size=args.blocks[-1],
                       eos_id=own.config.eos_token_id).tokens
    reference.config.block_size = args.blocks[-1]
    reference_ids = reference.generate(input_ids=prompt, max_new_tokens=args.tokens,
                                       temperature=0, top_k=0, top_p=1,
                                       eos_token_id=own.config.eos_token_id)[0, prompt.shape[1]:].tolist()
    ar_ids = generate_ar(MaskedAttentionBranch(own), prompt, args.tokens,
                         eos_id=own.config.eos_token_id).tokens
    greedy_pass = own_ids == reference_ids == ar_ids
    return {"records": records, "thresholds": {"max_logit_error": limit, "max_tv": tv_limit},
            "numerical_pass": numerical_pass, "greedy_pass": greedy_pass,
            "greedy_tokens": len(own_ids), "own_tokens": own_ids,
            "reference_tokens": reference_ids, "ar_tokens": ar_ids,
            "pass": numerical_pass and greedy_pass}


def reference_generate(model, prompt, tokens, block, sampling, seed):
    counts = {"prefill": 0, "ar": 0, "draft": 0}

    def count(module, positional, kwargs):
        if sum(counts.values()) == 0:
            counts["prefill"] += 1
        else:
            counts["draft" if kwargs.get("is_diffusion_pass", False) else "ar"] += 1

    hook = model.register_forward_pre_hook(count, with_kwargs=True)
    model.config.block_size = max(block, 2)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        output = model.generate(input_ids=prompt, max_new_tokens=tokens,
                                temperature=sampling.temperature, top_k=sampling.top_k,
                                top_p=sampling.top_p, use_diffusion_mode=block > 1,
                                do_sample=sampling.temperature > 0,
                                eos_token_id=model.config.eos_token_id)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    elapsed = time.perf_counter() - start
    token_ids = output[0, prompt.shape[1]:].tolist()
    return {"tokens": len(token_ids), "seconds": elapsed,
            "prefill_forwards": counts["prefill"], "prefill_output_tokens": 1,
            "decode_forwards": counts["ar"] + counts["draft"], "token_ids": token_ids}


class ProfileSampler(ProposalSampler):
    def sample_ar(self, logits, generator):
        with torch.profiler.record_function("blockspec.sample_ar"):
            return super().sample_ar(logits, generator)

    def propose(self, logits, generator, **kwargs):
        with torch.profiler.record_function("blockspec.propose"):
            return super().propose(logits, generator, **kwargs)

    def verify(self, proposal, target_logits, generator):
        with torch.profiler.record_function("blockspec.correct"):
            return super().verify(proposal, target_logits, generator)


@torch.inference_mode()
def profile_generation(model, prompt, *, tokens, block_size, sampling, seed):
    """Separate operator attribution from uninstrumented throughput measurements."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if prompt.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    versions = {name: value._version for name, value in model.named_parameters()}
    branch, records = MaskedAttentionBranch(model), []

    def run(method, sampler):
        options = {"sampler": sampler, "sampling": sampling, "eos_id": model.config.eos_token_id,
                   "generator": torch.Generator(device=prompt.device).manual_seed(seed)}
        return (generate_ar(branch, prompt, tokens, **options) if method == "ar" else
                generate(branch, prompt, tokens, block_size=block_size, **options))

    for method in ("ar", "speculative"):
        expected = run(method, ProposalSampler(sampling))
        scopes = []

        def before(module, inputs, kwargs, active=scopes):
            stage = ("draft" if kwargs.get("view") == "draft" else
                     "prefill" if kwargs.get("cache") is None else "ar_forward")
            scope = torch.profiler.record_function(f"blockspec.{stage}")
            scope.__enter__()
            active.append(scope)

        def after(module, inputs, output, active=scopes):
            active.pop().__exit__(None, None, None)

        pre = model.register_forward_pre_hook(before, with_kwargs=True)
        post = model.register_forward_hook(after, always_call=True)
        try:
            with torch.profiler.profile(activities=activities, acc_events=True) as profiler:
                actual = run(method, ProfileSampler(sampling))
        finally:
            pre.remove()
            post.remove()
        events = [{"name": event.key, "count": event.count,
                   "self_cpu_us": event.self_cpu_time_total, "self_device_us": event.self_device_time_total,
                   "inclusive_device_us": event.device_time_total} for event in profiler.key_averages()
                  if event.device_type == torch.autograd.DeviceType.CPU]
        records.append({"method": method, "tokens": len(actual.tokens), "decode_forwards": actual.decode_forwards,
                        "tokens_unchanged": actual.tokens == expected.tokens,
                        "counters_unchanged": actual.accepted_per_round == expected.accepted_per_round
                        and actual.decode_forwards == expected.decode_forwards,
                        "regions": [event for event in events if event["name"].startswith("blockspec.")],
                        "operators": sorted((event for event in events if not event["name"].startswith("blockspec.")),
                                            key=lambda event: event["self_device_us"], reverse=True)[:20]})
    unchanged = versions == {name: value._version for name, value in model.named_parameters()}
    return {"profiles": records, "parameters_unchanged": unchanged,
            "pass": unchanged and all(row["tokens_unchanged"] and row["counters_unchanged"] for row in records)}


@torch.inference_mode()
def trace_decision(args, own):
    """Compare one greedy decision, including a same-cache one-row/block replay."""
    prompt = prompt_ids(args.model, args.request_index + 1, path=args.prompts, thinking=args.thinking,
                        empty_system=args.empty_system)[args.request_index]
    branch, captures, output_ids = MaskedAttentionBranch(own), {}, {}
    for mode in ("ar", "speculative"):
        selected = {}

        def capture(module, positional, kwargs, output, captured=selected):
            if kwargs.get("view", "ar") == "draft":
                return
            first = cache_length(kwargs.get("cache")) + positional[0].shape[1] - output.logits.shape[1] + 1
            row = prompt.shape[1] + args.token_index - first
            if 0 <= row < output.logits.shape[1]:
                captured.update(logits=output.logits[0, row].float().clone(), cache=kwargs.get("cache"),
                                inputs=positional[0].clone(), row=row)

        hook = own.register_forward_hook(capture, with_kwargs=True)
        try:
            output = (generate_ar(branch, prompt, args.tokens) if mode == "ar" else
                      generate(branch, prompt, args.tokens, block_size=args.blocks[-1]))
        finally:
            hook.remove()
        output_ids[mode] = output.tokens
        captures[mode] = selected
    if not all(captures.values()):
        raise ValueError("decision was outside the generated token range")
    first_difference = next((index for index, (left, right) in enumerate(zip(
        output_ids["ar"], output_ids["speculative"])) if left != right), None)
    ar, spec = captures["ar"]["logits"], captures["speculative"]["logits"]
    ar_id, spec_id = int(ar.argmax()), int(spec.argmax())
    saved = captures["ar"]
    future = prompt.new_full((1, args.blocks[-1]), own.config.mask_token_id)
    future[:, :1] = saved["inputs"][:, -1:]
    single = own(future[:, :1], cache=saved["cache"]).logits[0, 0]
    block = own(future, cache=saved["cache"]).logits[0, 0]
    return {"request_index": args.request_index, "token_index": args.token_index,
            "first_difference": first_difference, "prefix_identical":
                output_ids["ar"][:args.token_index] == output_ids["speculative"][:args.token_index],
            "actual_path": compare_logits(ar[None], spec[None]),
            "ar_choice": ar_id, "speculative_choice": spec_id,
            "ar_gap_between_choices": float(ar[ar_id] - ar[spec_id]),
            "block_gap_between_choices": float(spec[ar_id] - spec[spec_id]),
            "same_cache_single_vs_block": compare_logits(single[None], block[None]),
            "same_cache_single_choice": int(single.argmax()), "same_cache_block_choice": int(block.argmax())}


@torch.inference_mode()
def benchmark(args, own, reference):
    prompts = prompt_ids(args.model, args.requests, path=args.prompts, thinking=args.thinking,
                         empty_system=args.empty_system)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    engines = ({("own" if len(args.sampling_executions) == 1 else f"own_{mode}"): mode
                for mode in args.sampling_executions} if own is not None else {})
    samplers, preparation = {}, {}
    for implementation, mode in engines.items():
        for block in args.blocks:
            executor = (None if mode == "scalar" else SamplingExecutor(
                own.config.vocab_size, block, sampling, use_cuda_graph=mode == "graph"))
            samplers[implementation, block] = ProposalSampler(sampling, executor=executor)
            preparation[f"{implementation}_k{block}"] = 0. if executor is None else executor.setup_seconds
        samplers[implementation, 1] = samplers[implementation, args.blocks[0]]
    methods = [(implementation, block) for implementation in list(engines) + (["reference"] if reference is not None else [])
               for block in [1] + args.blocks]
    branch = None if own is None else MaskedAttentionBranch(own)

    def run(implementation, block, prompt, count, seed):
        if implementation == "reference":
            return reference_generate(reference, prompt, count, block, sampling, seed)
        options = {"sampling": sampling, "eos_id": own.config.eos_token_id,
                   "sampler": samplers[implementation, block],
                   "generator": torch.Generator(device="cuda").manual_seed(seed)}
        result = (generate_ar(branch, prompt, count, **options) if block == 1 else
                  generate(branch, prompt, count, block_size=block, **options))
        return result.summary() | {"token_ids": result.tokens}

    for implementation, block in methods:
        run(implementation, block, prompts[0], min(32, args.tokens), args.seed)
    records = []
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    for repeat in range(args.repeats):
        for index, prompt in enumerate(prompts):
            ordered = methods if (repeat + index) % 2 == 0 else list(reversed(methods))
            for implementation, block in ordered:
                row = run(implementation, block, prompt, args.tokens, args.seed + repeat * 1000 + index)
                row.update(implementation=implementation, block=block, request=index, repeat=repeat,
                           input_tokens=prompt.numel())
                records.append(row)
            print(report.dumps({"repeat": repeat, "requests_complete": index + 1}), flush=True)
    aggregate = {}
    for implementation, block in methods:
        key = f"{implementation}_k{block}"
        rows = [row for row in records if (row["implementation"], row["block"]) == (implementation, block)]
        total, seconds = sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)
        forwards = sum(row["decode_forwards"] for row in rows)
        aggregate[key] = {"tokens": total, "seconds": seconds, "tps": total / seconds,
                          "decode_tpf": (total - len(rows)) / forwards if forwards else None}
    random = np.random.default_rng(args.seed)
    for implementation, block in methods:
        key, base = f"{implementation}_k{block}", f"{implementation}_k1"
        aggregate[key]["speedup_vs_matched_ar"] = aggregate[key]["tps"] / aggregate[base]["tps"]
        if block == 1:
            continue
        grouped = {}
        identical = True
        for index in range(len(prompts)):
            values = []
            for size in (1, block):
                rows = [row for row in records if row["implementation"] == implementation
                        and row["block"] == size and row["request"] == index]
                values.extend((sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)))
            grouped[index] = values
            if sampling.temperature == 0:
                for repeat in range(args.repeats):
                    rows = [row for row in records if row["implementation"] == implementation
                            and row["block"] in (1, block) and row["request"] == index and row["repeat"] == repeat]
                    identical &= rows[0]["token_ids"] == rows[1]["token_ids"]
        values = np.asarray(list(grouped.values()))
        selected = random.integers(0, len(values), size=(2000, len(values)))
        summed = values[selected].sum(1)
        ratios = (summed[:, 2] / summed[:, 3]) / (summed[:, 0] / summed[:, 1])
        aggregate[key]["paired_request_ci95"] = np.quantile(ratios, [.025, .975]).tolist()
        aggregate[key]["greedy_identical"] = identical if sampling.temperature == 0 else None
    equivalence = {}
    for block in [1] + args.blocks:
        baseline = next(iter(engines), "reference")
        base_rows = {(row["request"], row["repeat"]): row for row in records
                     if row["implementation"] == baseline and row["block"] == block}
        for implementation, size in methods:
            if size != block or implementation == baseline:
                continue
            selected = [row for row in records if row["implementation"] == implementation and row["block"] == block]
            key = f"{implementation}_vs_{baseline}_k{block}"
            equivalence[key] = {"requests": len(selected), **{
                field + "_identical": (all(row[field] == base_rows[row["request"], row["repeat"]][field]
                                           for row in selected)
                                       if all(field in row and field in base_rows[row["request"], row["repeat"]]
                                              for row in selected) else None)
                for field in ("token_ids", "accepted_per_round", "decode_forwards")}}
    return {"sampling": asdict(sampling), "sampling_execution": engines, "sampling_setup_seconds": preparation,
            "execution_equivalence": equivalence,
            "workload": "local question JSONL" if args.prompts else "fixed smoke prompts",
            "thinking": args.thinking, "empty_system": args.empty_system,
            "prompt_file_sha256": None if args.prompts is None else file_sha256(args.prompts),
            "prompts": prompt_texts(args.prompts, args.requests), "records": records, "aggregate": aggregate,
            "resident_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "benchmark", "trace", "profile"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, help="required when importing an external reference")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--backend", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--implementation", choices=("own", "reference", "both"), default="own")
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--empty-system", action="store_true", help="include the public quickstart's empty system message")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--sampling-executions", nargs="+", choices=("scalar", "tensor", "graph"))
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--request-index", type=int, default=6)
    parser.add_argument("--token-index", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sampling_executions is None:
        args.sampling_executions = ["tensor" if args.mode == "benchmark" and args.temperature > 0
                                    and args.implementation != "reference" else "scalar"]
    if (args.tokens < 1 or args.requests < 1 or (args.prompts is None and args.requests > len(PROMPTS)) or args.repeats < 1
            or any(block < 2 for block in args.blocks) or len(set(args.blocks)) != len(args.blocks)):
        parser.error("positive budgets, unique blocks >= 2, and prompts within the input data required")
    if args.output is not None and args.output.exists():
        parser.error("choose a new output file")
    if len(set(args.sampling_executions)) != len(args.sampling_executions) or (args.sampling_executions != ["scalar"] and (
            args.mode != "benchmark" or args.implementation == "reference" or args.temperature <= 0)):
        parser.error("distinct prepared sampling modes require an own/both positive-temperature benchmark")
    if args.mode == "profile" and (args.implementation != "own" or args.tokens > 256):
        parser.error("profile uses the independent implementation and at most 256 tokens")
    if args.mode == "trace" and (args.implementation == "reference" or args.request_index < 0
                                  or (args.prompts is None and args.request_index >= len(PROMPTS))
                                  or not 0 <= args.token_index < args.tokens):
        parser.error("trace requires the independent model and valid request/token indices")
    if (args.mode == "audit" or args.implementation in ("reference", "both")) and args.reference_manifest is None:
        parser.error("external execution requires a caller-supplied --reference-manifest")
    torch.set_num_threads(1)
    own = reference = None
    dtype = getattr(torch, args.dtype)
    if args.mode == "audit" or args.implementation in ("own", "both"):
        own = load_public(args.model, device="cuda", dtype=dtype).set_backend(args.backend)
    if args.mode == "audit" or args.implementation in ("reference", "both"):
        reference = load_reference(args.model, dtype, args.backend, args.reference_manifest)
    result = {"mode": args.mode, "dtype": args.dtype, "backend": args.backend,
              "python": platform.python_version(), "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "seed": args.seed,
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    if args.mode == "audit":
        result.update(audit(args, own, reference))
    elif args.mode == "trace":
        result.update(trace_decision(args, own))
    elif args.mode == "profile":
        prompt = prompt_ids(args.model, 1, path=args.prompts, thinking=args.thinking,
                            empty_system=args.empty_system)[0]
        result.update(profile_generation(own, prompt, tokens=args.tokens, block_size=args.blocks[-1],
                                         sampling=SamplingConfig(args.temperature, args.top_k, args.top_p), seed=args.seed))
    else:
        result.update(benchmark(args, own, reference))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            report.dump(result, handle, indent=2)
            handle.write("\n")
    print(report.dumps(result if args.mode == "audit" else {key: value for key, value in result.items()
                                                        if key != "records"}, indent=2))
    if args.mode in ("audit", "profile") and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
