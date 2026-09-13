"""Frozen-weight audits and paired benchmarks for cheap online proposal mixing."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
from pathlib import Path
import time

import torch

from blockspec_ablation.adapter_io import load_peft_adapter, peft_config
from blockspec_ablation.benchmark import aggregate, continuation_prompts
from blockspec_ablation.calibration import OverlapMix
from blockspec_ablation.continuation import ContinuationMix
from blockspec_ablation.checkpoint import adapter_fingerprint, base_fingerprint, implementation_fingerprint, load_hf_base
from blockspec.data import assert_split_files_disjoint, load_sequences
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.diffusion import UniformNoise
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec.sampling import SamplingConfig
from blockspec_ablation.sampling_execution import SamplingExecutor
from prefix_relay import assert_frozen, paired_bootstrap, sha


def emit(value):
    print(report.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "benchmark"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-sha256", help="optional local adapter integrity check")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--split-role", choices=["validation", "test"], default="validation")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[.5, .75, 1., 1.25, 1.5])
    parser.add_argument("--learning-rate", type=float, default=.5)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--sampling-execution", choices=["eager", "cuda_graph"], default="eager")
    parser.add_argument("--compare-eager", action="store_true")
    parser.add_argument("--learn-requests", type=int, default=0)
    parser.add_argument("--learn-tokens", type=int, default=128)
    parser.add_argument("--audit-learned", action="store_true")
    parser.add_argument("--method", choices=("temperatures", "continuation"), default="temperatures")
    parser.add_argument("--copy-start-depth", type=int, default=1)
    parser.add_argument("--fixed", type=float, nargs="*", default=[])
    parser.add_argument("--audit-online", action="store_true")
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.method == "continuation" and args.fixed:
        parser.error("fixed temperatures apply to the temperature mixture")
    if (args.repeats < 2 or args.repeats % 2 or args.top_k < 1 or args.temperature <= 0
            or args.learn_requests < 0 or args.learn_tokens < 1):
        parser.error("positive temperature/top-k and a positive even repeat count required")
    assert_split_files_disjoint(args.train_data, args.data)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = peft_config(args.adapter)
    model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device="cuda",
                        dtype=getattr(torch, args.dtype))
    source = load_peft_adapter(args.adapter, model, expected_sha256=args.reference_sha256)
    model.eval().requires_grad_(False).set_attention_backend("grouped")
    frozen = {"base": base_fingerprint(model), "adapter": adapter_fingerprint(model)}
    prompts = [p.cuda() for p in continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                                    count=args.prompts, length=args.prompt_length)]
    sampling, noise = SamplingConfig(args.temperature, args.top_k, args.top_p), UniformNoise(1, model.config.vocab_size)
    engine = FixedShapeExecutor(model, capacity=args.prompt_length + max(args.tokens, args.learn_tokens, 32), max_query=args.block_size)
    clean = [(n, False, None) for n in range(1, args.block_size + 1)]
    draft = [(n, True, None) for n in range(2, args.block_size + 1)]
    engine.prepare(clean + draft)
    setups = {"ar": engine.signature_seconds[(1, False, None)], "base": engine.setup_seconds}
    setups["ar_eager"], setups["base_eager"] = setups["ar"], setups["base"]
    sampler = None
    if args.sampling_execution == "cuda_graph":
        sampler = SamplingExecutor(model.config.vocab_size, args.block_size, sampling,
                                   temperatures=tuple(args.temperatures) if args.method == "temperatures" else (),
                                   continuation=args.method == "continuation", device="cuda")
        setups["ar"] += sampler.signature_seconds[1, "plain"]
        setups["base"] += sum(value for (_, kind), value in sampler.signature_seconds.items()
                              if kind not in ("mixed", "continuation"))
        setups["identity"] = setups["online"] = engine.setup_seconds + sampler.setup_seconds
    options = dict(block_size=args.block_size, sampling=sampling, noise=noise, executor=engine, sampler_executor=sampler)
    provenance = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "implementation_sha256": implementation_fingerprint(), "script_sha256": sha(__file__),
                  "data_sha256": sha(args.data), "adapter": source, "frozen_fingerprints": frozen,
                  "training_data_sha256": sha(args.train_data),
                  "sampling": asdict(sampling), "noise": asdict(noise),
                  "retention_threshold": .97}

    def mixer(adaptive=False, fixed=1., diagnostics=False):
        if args.method == "continuation":
            return ContinuationMix(args.block_size, args.top_k, learning_rate=args.learning_rate,
                                   interval=args.interval, adaptive=adaptive, diagnostics=diagnostics, device="cuda",
                                   start_depth=args.copy_start_depth)
        return OverlapMix(args.block_size, args.top_k, temperatures=args.temperatures,
                          learning_rate=args.learning_rate, interval=args.interval, adaptive=adaptive,
                          fixed_temperature=fixed, diagnostics=diagnostics, device="cuda")

    warm = load_sequences(args.train_data, model.config.vocab_size)[0][:args.prompt_length].reshape(1, -1).cuda()
    generate_ar(model, warm, 32, sampling=sampling, executor=engine, sampler_executor=sampler)
    generate_speculative(model, warm, 32, **options)
    generate_speculative(model, warm, 32, calibrator=mixer(True), **options)
    if args.compare_eager:
        generate_ar(model, warm, 32, sampling=sampling, executor=engine)
        generate_speculative(model, warm, 32, **dict(options, sampler_executor=None))
    emit({"stage": "start", **provenance, "engine_setup_seconds": setups})
    learned_state, learning = None, None
    if args.learn_requests:
        learning_prompts = continuation_prompts(load_sequences(args.train_data, model.config.vocab_size),
                                                count=args.learn_requests, length=args.prompt_length)
        torch.cuda.synchronize()
        start = time.perf_counter()
        learner = mixer(True)
        torch.cuda.synchronize()
        initialization = time.perf_counter() - start
        learning_rows = []
        for index, prompt in enumerate(learning_prompts):
            result = generate_speculative(model, prompt.cuda(), args.learn_tokens, calibrator=learner, **options,
                                          generator=torch.Generator(device="cuda").manual_seed(args.seed + 10000 + index))
            learning_rows.append(result.summary())
            if (index + 1) % 8 == 0 or index + 1 == len(learning_prompts):
                emit({"stage": "learning_progress", "requests": index + 1, "updates": learner.updates})
        learned_state = learner.state_dict()
        learning = {**aggregate(learning_rows, setup_seconds=initialization), "final_state": learner.metrics()}
        emit({"stage": "learning_complete", "learning": learning})
    if args.mode == "audit":
        calibration = mixer(args.audit_online, diagnostics=True)
        if learned_state is not None:
            calibration.load_state_dict(learned_state)
        for index, prompt in enumerate(prompts):
            result = generate_speculative(model, prompt, args.tokens, calibrator=calibration, **options,
                                          generator=torch.Generator(device="cuda").manual_seed(args.seed + index))
            emit({"stage": "audit_progress", "request": index, "tokens_per_round": len(result.tokens) / result.rounds})
        emit({"stage": "audit_complete", **provenance, "learning": learning, **calibration.metrics(), **assert_frozen(model, frozen)})
        return

    arms = ["ar", "base", "identity", "online"] + [f"fixed_{value:g}" for value in args.fixed]
    if learned_state is not None:
        arms += ["learned", "continued"]
    if args.compare_eager:
        arms += ["ar_eager", "base_eager"]
    if len(set(arms)) != len(arms):
        parser.error("distinct fixed temperatures required")
    rows = {arm: [] for arm in arms}
    initialization = {arm: 0. for arm in arms}
    repetitions, final_states = [], []
    for repeat in range(args.repeats):
        mixers = {}
        for arm in arms[2:]:
            if arm.endswith("_eager"):
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            mixers[arm] = mixer(arm in ("online", "continued"), float(arm[6:]) if arm.startswith("fixed_") else 1.)
            if arm in ("learned", "continued"):
                mixers[arm].load_state_dict(learned_state)
            torch.cuda.synchronize()
            initialization[arm] += time.perf_counter() - start
        current = {arm: [] for arm in arms}
        for index, prompt in enumerate(prompts):
            order = arms if (repeat + index) % 2 == 0 else arms[::-1]
            for arm in order:
                generator = torch.Generator(device="cuda").manual_seed(args.seed + len(prompts) * repeat + index)
                current_sampler = None if arm.endswith("_eager") else sampler
                if arm in ("ar", "ar_eager"):
                    result = generate_ar(model, prompt, args.tokens, sampling=sampling, executor=engine, sampler_executor=current_sampler,
                                         generator=generator)
                else:
                    result = generate_speculative(model, prompt, args.tokens, calibrator=mixers.get(arm),
                                                  generator=generator, **dict(options, sampler_executor=current_sampler))
                row = result.summary()
                rows[arm].append(row)
                current[arm].append(row)
            emit({"stage": "progress", "repeat": repeat, "request": index,
                  "tps": {arm: current[arm][-1]["tps"] for arm in arms}})
        repetitions.append({arm: aggregate(current[arm]) for arm in arms})
        final_states.append({arm: mixers[arm].metrics() for arm in ("online", "continued") if arm in mixers})
    learned_audit = None
    if args.audit_learned and learned_state is not None:
        evaluated = mixer(False, diagnostics=True)
        evaluated.load_state_dict(learned_state)
        for index, prompt in enumerate(prompts):
            generate_speculative(model, prompt, args.tokens, calibrator=evaluated, **options,
                                 generator=torch.Generator(device="cuda").manual_seed(args.seed + index))
        learned_audit = evaluated.metrics()
    summary = {arm: aggregate(rows[arm], setup_seconds=initialization[arm],
                              engine_setup_seconds=setups.get(arm, setups.get("online", setups["base"]))) for arm in arms}
    vs_base = {arm: {"ratio": summary[arm]["tps"] / summary["base"]["tps"],
                     **paired_bootstrap(rows, "base", arm, len(prompts))} for arm in arms if arm != "base"}
    emit({"stage": "complete", **provenance, "summary": summary, "repetitions": repetitions,
          "final_online_states": final_states, "learning": learning, "learned_audit": learned_audit,
          "vs_base": vs_base,
          "retention_pass": vs_base["online"]["speed_ratio_95_interval"][0] >= .97,
          "vs_learned": ({arm: {"ratio": summary[arm]["tps"] / summary["learned"]["tps"],
                                **paired_bootstrap(rows, "learned", arm, len(prompts))} for arm in ("base", "continued")}
                         if learned_state is not None else None),
          "vs_ar": {arm: {"ratio": summary[arm]["tps"] / summary["ar"]["tps"],
                         **paired_bootstrap(rows, "ar", arm, len(prompts))} for arm in arms if arm != "ar"},
          **assert_frozen(model, frozen)})


if __name__ == "__main__":
    main()
