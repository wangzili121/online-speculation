"""Train only PrefixRelay heads on a frozen, hash-checked published adapter."""

import argparse

from blockspec import reporting as report
import hashlib
from pathlib import Path
import time
from dataclasses import asdict

import torch

from blockspec_ablation.benchmark import aggregate, continuation_prompts
from blockspec_ablation.adapter_io import load_peft_adapter, peft_config
from blockspec_ablation.checkpoint import (adapter_fingerprint, base_fingerprint, implementation_fingerprint, load_hf_base)
from blockspec.data import assert_split_files_disjoint, load_sequences
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.diffusion import UniformNoise
from blockspec_ablation.relay import (RelayConfig, RelayHead, RelayLearner, generate_relay, load_relay, save_relay)
from blockspec_ablation.relay import relay_candidates
from blockspec_ablation.relay_execution import RelayExecutor
from blockspec.sampling import SamplingConfig, probabilities


def emit(row):
    print(report.dumps(row), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_head_metadata(metadata, *, train_sha256, block_size, serving_contract=None):
    """Apply the same data and block contract to every evaluated head."""
    if metadata.get("train_sha256") != train_sha256:
        raise ValueError("head training-file SHA must match the checked training split")
    if metadata.get("config", {}).get("block_size") != block_size:
        raise ValueError("evaluation block size must match the trained head")
    if serving_contract is not None and metadata.get("serving_contract") != serving_contract:
        raise ValueError("head serving contract must match backbone, precision, sampling and noise")


def assert_frozen(model, expected):
    actual = {"base": base_fingerprint(model), "adapter": adapter_fingerprint(model)}
    if actual != expected or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("base and published adapter must retain their frozen execution tensors")
    return {"frozen_base_unchanged": True, "frozen_adapter_unchanged": True}


def paired_bootstrap(rows, reference, method, prompts, *, seed=271828, resamples=10000):
    """Resample request clusters, retaining all repeated runs of each request."""
    a, b = rows[reference], rows[method]
    if (prompts < 1 or resamples < 200 or len(a) != len(b) or len(a) % prompts
            or any(x["tokens"] != y["tokens"] for x, y in zip(a, b))):
        raise ValueError("paired equal-token repetitions and >=200 bootstrap draws required")
    groups = [(sum(a[j]["seconds"] for j in range(i, len(a), prompts)),
               sum(b[j]["seconds"] for j in range(i, len(b), prompts))) for i in range(prompts)]
    state, ratios = seed, []
    for _ in range(resamples):
        numerator = denominator = 0.
        for _ in range(prompts):
            state = (1664525 * state + 1013904223) & 0xffffffff
            x, y = groups[(state * prompts) >> 32]
            numerator += x
            denominator += y
        ratios.append(numerator / denominator)
    ratios.sort()
    return {"method": "paired_request_cluster_bootstrap", "clusters": prompts, "resamples": resamples,
            "seed": seed, "speed_ratio_95_interval": [ratios[int(.025 * resamples)], ratios[int(.975 * resamples) - 1]]}


class HeadAudit:
    """Paired TV on the serving head's same reached prefixes, without updates."""
    def __init__(self, head, sampling, *, evaluated_head=None):
        self.head, self.sampling = head, sampling
        self.evaluated_head = evaluated_head or head
        self.updates = self.feedback_blocks = 0
        self.update_seconds = 0.
        self.totals = torch.zeros(6, device=next(head.parameters()).device)

    def clear_replay(self):
        pass

    @torch.no_grad()
    def observe(self, feedback, *, may_update=True):
        count = len(feedback.previous)
        if not count:
            return
        p = probabilities(feedback.teacher, self.sampling)
        original = probabilities(feedback.logits, self.sampling)
        corrected = probabilities(self.evaluated_head(feedback.logits, feedback.previous), self.sampling)
        reference = probabilities(self.head(feedback.logits, feedback.previous), self.sampling)
        tv0 = .5 * (p - original).abs().sum(-1)
        tv1 = .5 * (p - corrected).abs().sum(-1)
        reference_tv = .5 * (p - reference).abs().sum(-1)
        reference_confidence = self.head.confidence_logits(feedback.hidden, feedback.previous).sigmoid()
        confidence = self.evaluated_head.confidence_logits(feedback.hidden, feedback.previous).sigmoid()
        self.totals += torch.stack((tv0.sum(), tv1.sum(), (confidence - (1 - tv1)).square().sum(),
                                    tv0.new_tensor(count), reference_tv.sum(),
                                    (reference_confidence - (1 - reference_tv)).square().sum()))
        self.feedback_blocks += 1


@torch.no_grad()
def profile_head(head, block_size, sampling, proposal):
    logits = torch.randn(block_size, head.config.vocab_size, device="cuda")
    hidden = torch.randn(block_size, head.config.hidden_size, device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(98472)
    methods = {"eager": lambda: relay_candidates(head, logits, hidden, sampling=sampling, generator=rng)}
    if proposal:
        methods["cuda_graph"] = lambda: proposal(logits, hidden, generator=rng)
    result = {}
    for name, method in methods.items():
        for _ in range(8):
            method()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(128):
            method()
        torch.cuda.synchronize()
        result[name + "_ms_per_block"] = (time.perf_counter() - start) * 1000 / 128
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "benchmark", "audit"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-sha256", help="optional local adapter integrity check")
    parser.add_argument("--reference-manifest", type=Path, help="local validation metadata for HF execution")
    parser.add_argument("--backbone", choices=["independent_graph", "hf_sdpa"], default="independent_graph")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--data", type=Path, required=True, help="evaluation file; always checked against training data")
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True, help="exclusive train output or benchmark input")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--train-requests", type=int, default=64)
    parser.add_argument("--train-tokens", type=int, default=128)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--confidence-lr", type=float, help="separate confidence rate; omission uses --lr")
    parser.add_argument("--embedding-init", choices=["random", "base_projected"], default="random")
    parser.add_argument("--audit-reference", type=Path, help="fixed serving head for common-prefix counterfactual TV")
    parser.add_argument("--compare-head", type=Path, help="additional fixed head in the paired throughput benchmark")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--noise-low", type=int, default=1)
    parser.add_argument("--noise-high", type=int)
    parser.add_argument("--threshold", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--online", action="store_true", help="include an additional head-continuation arm")
    parser.add_argument("--scheduled", action="store_true", help="include the confidence-truncated diagnostic arm")
    parser.add_argument("--head-execution", choices=["eager", "cuda_graph"], default="eager")
    args = parser.parse_args()
    if not args.adapter.is_dir():
        parser.error("provide the published PEFT adapter directory")
    if args.mode == "train" and args.head.exists():
        parser.error("select a new checkpoint output path")
    if args.repeats < 2 or args.repeats % 2 or min(args.tokens, args.train_tokens, args.train_requests) < 1:
        parser.error("positive budgets and a positive even repeat count required")
    assert_split_files_disjoint(args.train_data, args.data)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.backbone == "hf_sdpa":
        if args.reference_manifest is None:
            parser.error("HF execution requires --reference-manifest")
        from blockspec_ablation.hf_execution import load_frozen_hf
        model, adapter_source = load_frozen_hf(args.base, args.adapter, reference_manifest=args.reference_manifest,
                                               expected_sha256=args.reference_sha256,
                                               dtype=getattr(torch, args.dtype))
    else:
        config = peft_config(args.adapter)
        model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device="cuda",
                             dtype=getattr(torch, args.dtype))
        adapter_source = load_peft_adapter(args.adapter, model, expected_sha256=args.reference_sha256)
        model.set_attention_backend("grouped")
    model.eval().requires_grad_(False)
    frozen = {"base": base_fingerprint(model), "adapter": adapter_fingerprint(model)}
    binding = {"base": frozen["base"], "adapter": adapter_source["sha256"]}
    sampling = SamplingConfig(args.temperature, args.sampling_top_k, args.top_p)
    training_sampling = SamplingConfig(args.temperature if args.temperature > 0 else 1.,
                                        args.sampling_top_k, args.top_p)
    noise = UniformNoise(args.noise_low, model.config.vocab_size if args.noise_high is None else args.noise_high)
    contract = {"backbone": args.backbone, "base_dtype": args.dtype, "head_dtype": "float32",
                "block_size": args.block_size, "sampling": asdict(sampling), "noise": asdict(noise),
                "torch": str(torch.__version__), "transformers": adapter_source.get("transformers"),
                "backbone_reference": adapter_source.get("reference_lf_sha256")}
    head_training = None
    if args.mode == "train":
        head = RelayHead(RelayConfig(model.config.vocab_size, model.config.hidden_size, args.rank)).to("cuda")
        if args.embedding_init == "base_projected":
            head.initialize_from_embedding(model.model.embed_tokens.weight, seed=args.seed)
    else:
        head, metadata = load_relay(args.head, binding=binding, device="cuda")
        validate_head_metadata(metadata, train_sha256=sha(args.train_data), block_size=args.block_size,
                               serving_contract=contract)
        head_training = {key: metadata.get(key) for key in
                         ("config", "generated_tokens", "updates", "examples", "training_seconds")}
    source = implementation_fingerprint()
    provenance = {"implementation_sha256": source, "script_sha256": sha(__file__), "binding": binding,
                  "evaluation_sha256": sha(args.data), "train_sha256": sha(args.train_data),
                  "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "head_training": head_training,
                  "serving_contract": contract, "adapter_source": adapter_source,
                  "frozen_execution_fingerprints": frozen,
                  "head_parameters": sum(p.numel() for p in head.parameters()), "precision": args.dtype,
                  "device": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    engine, setups = None, {"ar": 0., "parallel": 0., "relay": 0.}
    if args.backbone == "independent_graph":
        engine = FixedShapeExecutor(model, capacity=args.prompt_length + max(args.tokens, args.train_tokens, 32),
                                    max_query=args.block_size)
        clean = [(n, False, None) for n in range(1, args.block_size + 1)]
        ordinary = [(n, True, None) for n in range(2, args.block_size + 1)]
        relay = [(n, True, model.config.num_hidden_layers) for n in range(2, args.block_size + 1)]
        engine.prepare(clean + ordinary + relay)
        setups = {"ar": engine.signature_seconds[(1, False, None)],
                  "parallel": sum(engine.signature_seconds[k] for k in clean + ordinary),
                  "relay": sum(engine.signature_seconds[k] for k in clean + relay)}
    emit({"stage": "start", **provenance, "engine_setup_seconds": setups})
    options = {"block_size": args.block_size, "sampling": sampling, "noise": noise, "executor": engine}
    proposal = (RelayExecutor(head, block_size=args.block_size, sampling=options["sampling"])
                if args.head_execution == "cuda_graph" else None)
    scheduled_proposal = (RelayExecutor(head, block_size=args.block_size, sampling=options["sampling"],
                                        threshold=args.threshold) if proposal and args.scheduled else None)
    head_setup = proposal.setup_seconds if proposal else 0.
    scheduled_setup = scheduled_proposal.setup_seconds if scheduled_proposal else 0.
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    prompts = continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                   count=args.prompts, length=args.prompt_length)
    # Warm all paths using a training-record prefix, including every scheduled query shape.
    sequences = load_sequences(args.train_data, model.config.vocab_size)
    warm = sequences[0][:args.prompt_length].reshape(1, -1).cuda()
    generate_ar(model, warm, 32, sampling=options["sampling"], executor=engine, generator=rng)
    generate_speculative(model, warm, 32, **options, generator=rng)
    generate_relay(model, head, warm, 32, **options, generator=rng, proposal_executor=proposal)
    if args.scheduled:
        generate_relay(model, head, warm, 32, **options, generator=rng, threshold=args.threshold,
                       proposal_executor=scheduled_proposal)
    if args.mode == "audit":
        reference = head
        if args.audit_reference:
            reference, reference_metadata = load_relay(args.audit_reference, binding=binding, device="cuda")
            validate_head_metadata(reference_metadata, train_sha256=sha(args.train_data), block_size=args.block_size,
                                   serving_contract=contract)
        audit = HeadAudit(reference, training_sampling,
                          evaluated_head=head)
        reference_executor = (RelayExecutor(reference, block_size=args.block_size, sampling=options["sampling"])
                              if proposal else None)
        rng.manual_seed(args.seed)
        trace = hashlib.sha256()
        for prompt in prompts:
            result = generate_relay(model, reference, prompt.cuda(), args.tokens, **options, generator=rng,
                                    proposal_executor=reference_executor, learner=audit)
            trace.update(torch.tensor(result.tokens, dtype=torch.int64).numpy().tobytes())
        original, corrected, error, count, reference_tv, reference_error = audit.totals.tolist()
        emit({"stage": "audit", **provenance, "positions": count, "original_tv": original / count,
              "corrected_tv": corrected / count, "confidence_mse": error / count,
              "reference_tv": reference_tv / count, "trace_sha256": trace.hexdigest(),
              "reference_confidence_mse": reference_error / count,
              "head_sha256": sha(args.head), "reference_head_sha256": sha(args.audit_reference or args.head),
              "microbenchmark": profile_head(head, args.block_size, options["sampling"], proposal),
              **assert_frozen(model, frozen)})
        return
    if args.mode == "train":
        learner = RelayLearner(head, lr=args.lr, interval=args.interval,
                               sampling=training_sampling,
                               confidence_lr=args.confidence_lr)
        crop_rng = torch.Generator().manual_seed(args.seed)
        eligible = [s for s in sequences if len(s) >= args.prompt_length]
        start = time.perf_counter()
        total = 0
        for request in range(args.train_requests):
            sequence = eligible[int(torch.randint(len(eligible), (), generator=crop_rng))]
            offset = int(torch.randint(len(sequence) - args.prompt_length + 1, (), generator=crop_rng))
            prompt = sequence[offset:offset + args.prompt_length].reshape(1, -1).cuda()
            result = generate_relay(model, head, prompt, args.train_tokens, **options, learner=learner,
                                    generator=rng, proposal_executor=proposal)
            total += len(result.tokens)
            if request % 8 == 0 or request + 1 == args.train_requests:
                emit({"stage": "training", "request": request + 1, "updates": learner.updates,
                      "examples": learner.examples, "seconds": time.perf_counter() - start, **learner.last_metrics})
        metadata = {**provenance, "training_seconds": time.perf_counter() - start, "generated_tokens": total,
                    "updates": learner.updates, "examples": learner.examples,
                    "update_seconds": learner.update_seconds, "last_metrics": learner.last_metrics}
        metadata.update(assert_frozen(model, frozen))
        save_relay(args.head, head, binding=binding, metadata=metadata)
        emit({"stage": "trained", "head_sha256": sha(args.head), **metadata})
        return
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    reference_head = reference_proposal = None
    if args.compare_head:
        reference_head, reference_metadata = load_relay(args.compare_head, binding=binding, device="cuda")
        validate_head_metadata(reference_metadata, train_sha256=sha(args.train_data), block_size=args.block_size,
                               serving_contract=contract)
        reference_proposal = (RelayExecutor(reference_head, block_size=args.block_size, sampling=options["sampling"])
                              if proposal else None)
        generate_relay(model, reference_head, warm, 32, **options, generator=rng, proposal_executor=reference_proposal)
        setups["reference"] = setups["relay"] + (reference_proposal.setup_seconds if reference_proposal else 0.)
    arms = ["ar", "parallel"] + (["reference"] if reference_head else []) + ["relay"]
    arms += ["scheduled"] if args.scheduled else []
    arms += ["online"] if args.online else []
    rows = {arm: [] for arm in arms}
    repeats = []
    learner_setup_seconds = 0.
    greedy_matches = {arm: 0 for arm in arms if arm != "ar"}
    for repeat in range(args.repeats):
        head.load_state_dict(initial)
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        learner = (RelayLearner(head, lr=args.lr, interval=args.interval,
                                sampling=training_sampling,
                                confidence_lr=args.confidence_lr)
                   if args.online else None)
        torch.cuda.synchronize()
        learner_setup_seconds += time.perf_counter() - setup_start if learner else 0.
        online_state = {k: v.clone() for k, v in initial.items()}
        per_repeat = {arm: [] for arm in arms}
        for index, prompt in enumerate(prompts):
            order = arms if (repeat + index) % 2 == 0 else arms[::-1]
            outputs = {}
            for arm in order:
                head.load_state_dict(online_state if arm == "online" else initial)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + len(prompts) * repeat + index)
                if arm == "ar":
                    result = generate_ar(model, prompt.cuda(), args.tokens, sampling=options["sampling"],
                                         executor=engine, generator=generator)
                elif arm == "parallel":
                    result = generate_speculative(model, prompt.cuda(), args.tokens, **options, generator=generator)
                elif arm == "reference":
                    result = generate_relay(model, reference_head, prompt.cuda(), args.tokens, **options,
                                            generator=generator, proposal_executor=reference_proposal)
                else:
                    result = generate_relay(model, head, prompt.cuda(), args.tokens, **options, generator=generator,
                                            threshold=args.threshold if arm == "scheduled" else 0.,
                                            learner=learner if arm == "online" else None,
                                            proposal_executor=scheduled_proposal if arm == "scheduled" else proposal)
                if arm == "online":
                    online_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                row = result.summary()
                rows[arm].append(row)
                per_repeat[arm].append(row)
                outputs[arm] = result.tokens
            if args.temperature == 0:
                for arm in greedy_matches:
                    greedy_matches[arm] += outputs[arm] == outputs["ar"]
            emit({"stage": "benchmark", "repeat": repeat, "request": index,
                  "tps": {arm: per_repeat[arm][-1]["tps"] for arm in arms}})
        repeats.append({arm: aggregate(per_repeat[arm]) for arm in arms})
    summary = {}
    for arm in arms:
        setup = setups.get(arm, setups["relay"])
        setup += scheduled_setup if arm == "scheduled" else head_setup if arm in ("relay", "online") else 0.
        summary[arm] = aggregate(rows[arm], engine_setup_seconds=setup,
                                 setup_seconds=learner_setup_seconds if arm == "online" else 0.)
        if arm not in ("ar", "parallel"):
            summary[arm]["verified_tokens"] = sum(r["verified_tokens"] for r in rows[arm])
            for key in ("depth_proposed", "depth_accepted"):
                summary[arm][key] = [sum(r[key][i] for r in rows[arm]) for i in range(args.block_size - 1)]
    emit({"stage": "complete", **provenance, "head_sha256": sha(args.head), "summary": summary,
          "repetitions": repeats, "greedy_matches": greedy_matches if args.temperature == 0 else None,
          "paired_uncertainty": {arm: paired_bootstrap(rows, "parallel", arm, len(prompts))
                                 for arm in arms if arm not in ("ar", "parallel")},
          "reference_head_sha256": sha(args.compare_head) if args.compare_head else None,
          "relay_vs_reference": paired_bootstrap(rows, "reference", "relay", len(prompts)) if reference_head else None,
          "online_vs_fixed": paired_bootstrap(rows, "relay", "online", len(prompts)) if args.online else None,
          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          **assert_frozen(model, frozen)})


if __name__ == "__main__":
    main()
