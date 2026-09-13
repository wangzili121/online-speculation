"""Small-model offline reproduction: own decoder with trained or reference weights."""

import argparse

from blockspec import reporting as report
import hashlib
from pathlib import Path

import torch

from blockspec_ablation.adapter_io import load_peft_adapter, peft_config
from blockspec_ablation.benchmark import BenchmarkConfig, benchmark_offline, continuation_prompts
from blockspec_ablation.checkpoint import implementation_fingerprint, load_checkpoint, load_hf_base
from blockspec.data import load_sequences
from blockspec_ablation.diffusion import UniformNoise
from blockspec.sampling import SamplingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-sha256", help="optional local adapter integrity check")
    parser.add_argument("--reference-manifest", type=Path, help="local validation metadata for HF execution")
    parser.add_argument("--backbone", choices=["independent_graph", "hf_sdpa"], default="independent_graph")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-role", choices=["validation", "test"], default="validation")
    parser.add_argument("--prompts", type=int, default=17)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--sampler", choices=["linear", "tree"], default="tree")
    parser.add_argument("--top-k", type=int, default=8, help="candidate width for tree construction")
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy; positive: target sampling temperature")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="execution base precision; local checkpoints are validated against the FP32 source first")
    parser.add_argument("--sampling-top-k", type=int, default=0, help="target probability filter; 0 keeps the vocabulary")
    parser.add_argument("--top-p", type=float, default=1.0, help="target nucleus mass")
    parser.add_argument("--noise-low", type=int, default=0, help="inclusive uniform draft-noise bound")
    parser.add_argument("--noise-high", type=int, help="exclusive draft-noise bound; defaults to vocabulary size")
    parser.add_argument("--prefix-budget", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    source_sha = implementation_fingerprint()
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.backbone == "hf_sdpa":
        if not args.adapter.is_dir() or args.reference_manifest is None or args.sampler != "linear":
            parser.error("HF execution requires a local reference manifest, PEFT directory and linear sampling")
        from blockspec_ablation.hf_execution import load_frozen_hf
        model, provenance = load_frozen_hf(args.base, args.adapter, reference_manifest=args.reference_manifest,
                                               expected_sha256=args.reference_sha256,
                                           dtype=getattr(torch, args.dtype))
    elif args.adapter.is_dir():
        config = peft_config(args.adapter)
        model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device="cuda",
                             dtype=getattr(torch, args.dtype))
        provenance = load_peft_adapter(args.adapter, model, expected_sha256=args.reference_sha256)
    else:
        payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
        if not payload.get("adapter_only"):
            parser.error("provide an adapter-only checkpoint")
        if data_sha == payload.get("metadata", {}).get("train_data_sha256"):
            parser.error("select a validation or test file")
        config = payload["config"]
        del payload
        model = load_hf_base(args.base, rank=config["adapter_rank"], alpha=config["adapter_alpha"], device="cuda")
        model, metadata = load_checkpoint(args.adapter, model=model, device="cuda")
        provenance = {"kind": "local_training", "sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                      "training": metadata}
        # Validate the original checkpoint before explicitly changing base precision.
        # Adapter masters retain their stored dtype and every original value.
        model.set_base_dtype(getattr(torch, args.dtype))
    config = BenchmarkConfig(tokens=args.tokens, block_size=args.block_size, repeats=args.repeats,
                             warmup_tokens=32, sampler=args.sampler, top_k=args.top_k,
                             prefix_budget=args.prefix_budget,
                             execution="eager" if args.backbone == "hf_sdpa" else "cuda_graph",
                             attention_backend="sdpa" if args.backbone == "hf_sdpa" else "grouped", seed=args.seed,
                             sampling=SamplingConfig(args.temperature, args.sampling_top_k, args.top_p),
                             noise=UniformNoise(args.noise_low, args.noise_high))
    prompts = continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                   count=args.prompts, length=args.prompt_length)
    print(report.dumps({"stage": "start", "implementation_sha256": source_sha,
                      "data_sha256": data_sha, "split_role": args.split_role,
                      "adapter": provenance, "backbone": args.backbone,
                      "dtype": args.dtype, "device": torch.cuda.get_device_name()}), flush=True)
    result = benchmark_offline(model, prompts, config,
                               progress=(lambda row: print(report.dumps(row), flush=True)) if args.progress else None)
    print(report.dumps({"stage": "complete", "implementation_sha256": source_sha,
                      "data_sha256": data_sha, "adapter_sha256": provenance["sha256"],
                      "backbone": args.backbone, "dtype": args.dtype, **result}), flush=True)


if __name__ == "__main__":
    main()
