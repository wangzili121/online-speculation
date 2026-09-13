"""Small executable end-to-end pipeline; no imported research implementation."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .checkpoint import (adapter_state, base_fingerprint, implementation_fingerprint,
                         load_checkpoint, load_hf_base, save_checkpoint)
from blockspec.data import assert_split_files_disjoint, load_sequences
from .decoding import generate_ar, generate_speculative
from .diffusion import UniformNoise
from .distillation import LOSS_KINDS, offline_step, paired_loss
from .model import Decoder, ModelConfig
from .online import OnlineConfig, OnlineLearner
from blockspec.sampling import SamplingConfig
from .training import TrainingConfig, train_adapter
from .tree import generate_tree


def toy_sequences(count, length, vocab, *, device, generator):
    """A learnable periodic stream, not a natural-language speed benchmark."""
    starts = torch.randint(1, vocab - 1, (count, 1), device=device, generator=generator)
    positions = torch.arange(length - 1, device=device)[None]
    # Token 0 is BOS; 1..V-2 form a deterministic cycle; V-1 is unused EOS.
    body = 1 + ((starts - 1 + positions) % (vocab - 2))
    return torch.cat((torch.zeros(count, 1, device=device, dtype=torch.long), body), dim=1)


def demo(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    config = ModelConfig(vocab_size=16, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, adapter_rank=args.rank,
                         adapter_alpha=float(args.rank))
    model = Decoder(config).to(device)
    train = toy_sequences(64, 33, config.vocab_size, device=device, generator=rng)
    model.train_base_only()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=0.003, weight_decay=0)
    for step in range(args.base_steps):
        batch = train[(step % 8) * 8:(step % 8 + 1) * 8]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        optimizer.step()
    print(report.dumps({"stage": "base", "steps": args.base_steps, "loss": float(loss.detach())}), flush=True)
    model.train_adapters_only()
    frozen = base_fingerprint(model)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=0.003, weight_decay=0)
    fixed_noise = torch.randint(config.vocab_size, train[:8].shape, device=device, generator=rng)
    with torch.no_grad():
        before = float(paired_loss(model, train[:8], args.block_size, noisy=fixed_noise))
    blocks = [2, 4, args.block_size]
    for step in range(args.adapter_steps):
        block = blocks[min(2, 3 * step // args.adapter_steps)]
        stats = offline_step(model, optimizer, train[(step % 8) * 8:(step % 8 + 1) * 8],
                             block, kind=args.loss, generator=rng)
    with torch.no_grad():
        after = float(paired_loss(model, train[:8], args.block_size, noisy=fixed_noise))
    assert base_fingerprint(model) == frozen, "offline training changed base weights"
    print(report.dumps({"stage": "offline", "steps": args.adapter_steps,
                      "fixed_noise_l1_before": before, "fixed_noise_l1_after": after,
                      "last_train": stats}), flush=True)
    if args.checkpoint:
        save_checkpoint(args.checkpoint, model, metadata={"task": "synthetic-cycle", "seed": args.seed})
        loaded, _ = load_checkpoint(args.checkpoint, device=device)
        assert base_fingerprint(loaded) == frozen
        for name, value in adapter_state(model).items():
            assert torch.equal(value, adapter_state(loaded)[name])
        model = loaded.train_adapters_only()
    prompt = train[:1, :5]
    generate = generate_tree if args.sampler == "tree" else generate_speculative
    ar = generate_ar(model, prompt, args.tokens)
    static = generate(model, prompt, args.tokens, block_size=args.block_size,
                                  generator=torch.Generator(device=device).manual_seed(args.seed + 2))
    learner = OnlineLearner(model, OnlineConfig(stride=args.update_stride, replay_blocks=2,
                                               learning_rate=0.001, loss=args.loss))
    original_adapter = adapter_state(model)
    online = generate(model, prompt, args.tokens, block_size=args.block_size,
                                  generator=torch.Generator(device=device).manual_seed(args.seed + 2),
                                  learner=learner)
    assert ar.tokens == static.tokens == online.tokens, "greedy outputs differ"
    assert base_fingerprint(model) == frozen, "online training changed base weights"
    changed = any(not torch.equal(v, adapter_state(model)[k]) for k, v in original_adapter.items())
    print(report.dumps({"stage": "decode", "sampler": args.sampler, "ar": ar.summary(), "static": static.summary(),
                      "online": online.summary(), "greedy_identical": True,
                      "base_unchanged": True, "online_adapter_changed": changed,
                      "scope": "synthetic correctness smoke; not a speed claim"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("demo", help="base training -> offline adapter -> checkpoint -> online decode")
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--seed", type=int, default=314159)
    run.add_argument("--base-steps", type=int, default=120)
    run.add_argument("--adapter-steps", type=int, default=240)
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--block-size", type=int, default=4)
    run.add_argument("--sampler", choices=["linear", "tree"], default="linear")
    run.add_argument("--tokens", type=int, default=128)
    run.add_argument("--update-stride", type=int, default=8)
    run.add_argument("--loss", choices=LOSS_KINDS, default="l1")
    run.add_argument("--checkpoint", type=Path)
    train = sub.add_parser("train", help="independently train a fresh adapter on local JSONL sequences")
    train.add_argument("--base", type=Path, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--initial-adapter", type=Path,
                       help="continue a matching local adapter; initializes a fresh optimizer")
    train.add_argument("--device", default="cuda")
    train.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--alpha", type=float,
                       help="adapter scaling numerator; alpha/rank multiplies each low-rank branch (default: rank)")
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--sequence-length", type=int, default=128)
    train.add_argument("--blocks", default="2,4,6,8")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--bos-id", type=int, default=0)
    train.add_argument("--seed", type=int, default=314159)
    train.add_argument("--text-data", action="store_true", help="use local HF tokenizer for text records")
    train.add_argument("--loss", choices=LOSS_KINDS, default="l1")
    train.add_argument("--warmup-loss", choices=["forward_kl", "reverse_kl", "reverse_kl_l1"], default="reverse_kl_l1")
    train.add_argument("--validation-data", type=Path)
    train.add_argument("--validation-every", type=int, default=100)
    train.add_argument("--validation-batches", type=int, default=4)
    train.add_argument("--threads", type=int, default=4)
    prepare = sub.add_parser("prepare", help="fetch a bounded public dataset subset with question-group splits")
    prepare.add_argument("--base", type=Path, required=True, help="local tokenizer and chat template")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset", required=True, help="caller-selected conversation dataset")
    prepare.add_argument("--dataset-config", default="default")
    prepare.add_argument("--source-split", default="train")
    prepare.add_argument("--offsets", required=True, help="comma-separated source row offsets")
    prepare.add_argument("--page-size", type=int, default=8)
    prepare.add_argument("--max-tokens", type=int, default=8192)
    prepare.add_argument("--seed", type=int, default=314159)
    bench = sub.add_parser("benchmark", help="balanced AR/static/online continuation streams; stdout only")
    bench.add_argument("--base", type=Path, required=True)
    bench.add_argument("--adapter", type=Path, required=True)
    bench.add_argument("--data", type=Path, required=True, help="tokenized development or held-out JSONL")
    bench.add_argument("--split-role", choices=["validation", "test"], default="validation")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                       help="source base precision for checkpoint fingerprint validation")
    bench.add_argument("--execution-dtype", choices=["float32", "bfloat16"],
                       help="explicit base precision after loading; preserves adapter master values")
    bench.add_argument("--prompts", type=int, default=4)
    bench.add_argument("--prompt-length", type=int, default=128)
    bench.add_argument("--tokens", type=int, default=128)
    bench.add_argument("--block-size", type=int, default=4)
    bench.add_argument("--sampler", choices=["linear", "tree"], default="linear")
    bench.add_argument("--top-k", type=int, default=4, help="candidate width for tree construction")
    bench.add_argument("--temperature", type=float, default=0.0, help="0: greedy; positive: target sampling temperature")
    bench.add_argument("--sampling-top-k", type=int, default=0, help="target probability filter; 0 keeps the vocabulary")
    bench.add_argument("--top-p", type=float, default=1.0, help="target nucleus mass")
    bench.add_argument("--noise-low", type=int, default=0, help="inclusive uniform draft-noise bound")
    bench.add_argument("--noise-high", type=int, help="exclusive draft-noise bound; defaults to vocabulary size")
    bench.add_argument("--prefix-budget", type=int, default=16)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--warmup-tokens", type=int, default=8)
    bench.add_argument("--update-stride", type=int, default=32)
    bench.add_argument("--replay-blocks", type=int, default=1)
    bench.add_argument("--online-last-layers", type=int, help="optional exact suffix replay; omit for full-adapter updates")
    bench.add_argument("--learning-rate", type=float, default=1e-4)
    bench.add_argument("--loss", choices=LOSS_KINDS, default="l1")
    bench.add_argument("--optimizer", choices=["auto", "standard", "fused"], default="auto",
                       help="online AdamW execution; auto selects fused on CUDA")
    bench.add_argument("--feedback-execution", choices=["windowed", "all"], default="windowed",
                       help="collect decoder feedback in the update window, or on every round")
    bench.add_argument("--update-policy", choices=["periodic", "coverage"], default="periodic",
                       help="periodic updates, or hold weights when every replay block is fully covered")
    bench.add_argument("--eos-id", type=int, help="omit for a declared fixed-token-budget measurement")
    bench.add_argument("--seed", type=int, default=271828)
    bench.add_argument("--threads", type=int, default=4)
    bench.add_argument("--progress", action="store_true", help="also print per-request counters")
    bench.add_argument("--execution", choices=["eager", "cuda_graph"], default="eager",
                       help="shared inference executor for AR/static/online; FP32 or BF16 CUDA graphs")
    bench.add_argument("--online-execution", choices=["eager", "cuda_graph"], default="eager",
                       help="prepared FP32 suffix forward/loss/gradient graphs, or eager online training")
    bench.add_argument("--attention-backend", choices=["sdpa", "grouped"], default="sdpa",
                       help="shared AR/draft/verifier/training attention; grouped specializes short FP32/FP64 queries")
    args = parser.parse_args()
    if args.command == "benchmark":
        from .benchmark import BenchmarkConfig, benchmark_streams, continuation_prompts
        implementation_sha = implementation_fingerprint()
        config = BenchmarkConfig(tokens=args.tokens, block_size=args.block_size, repeats=args.repeats,
                                 warmup_tokens=args.warmup_tokens, seed=args.seed, sampler=args.sampler,
                                 top_k=args.top_k, prefix_budget=args.prefix_budget, eos_id=args.eos_id,
                                 execution=args.execution, online_execution=args.online_execution,
                                 attention_backend=args.attention_backend,
                                 sampling=SamplingConfig(args.temperature, args.sampling_top_k, args.top_p),
                                 noise=UniformNoise(args.noise_low, args.noise_high))
        online_config = OnlineConfig(stride=args.update_stride, replay_blocks=args.replay_blocks,
                                     learning_rate=args.learning_rate, loss=args.loss,
                                     train_last_layers=args.online_last_layers, optimizer=args.optimizer,
                                     feedback_execution=args.feedback_execution, update_policy=args.update_policy)
        if args.threads < 1:
            parser.error("positive threads required")
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
        if not payload.get("adapter_only"):
            parser.error("benchmark expects an adapter-only checkpoint and its local base")
        saved_config = ModelConfig.from_dict(payload["config"])
        data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
        if data_sha == payload.get("metadata", {}).get("train_data_sha256"):
            parser.error("benchmark data is the checkpoint's training file")
        del payload
        model = load_hf_base(args.base, rank=saved_config.adapter_rank, alpha=saved_config.adapter_alpha,
                             device=args.device, dtype=getattr(torch, args.dtype))
        model, metadata = load_checkpoint(args.adapter, model=model, device=args.device)
        execution_dtype = args.execution_dtype or args.dtype
        model.set_base_dtype(getattr(torch, execution_dtype))
        sequences = load_sequences(args.data, model.config.vocab_size)
        prompts = continuation_prompts(sequences, count=args.prompts, length=args.prompt_length)
        progress = (lambda row: print(report.dumps(row), flush=True)) if args.progress else None
        result = benchmark_streams(model, prompts, config, online_config, progress=progress)
        result.update(data_sha256=data_sha, split_role=args.split_role,
                      adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                      implementation_sha256_at_start=implementation_sha, dtype=execution_dtype,
                      checkpoint_base_dtype=args.dtype,
                      offline_training_config=metadata.get("training_config"),
                      device=torch.cuda.get_device_name() if str(args.device).startswith("cuda") else str(args.device))
        print(report.dumps(result), flush=True)
        return
    if args.command == "prepare":
        from blockspec.corpus import prepare_snapshot
        from blockspec.tokenizer import LocalTokenizer
        manifest = prepare_snapshot(args.output, LocalTokenizer(args.base),
                                     dataset=args.dataset, dataset_config=args.dataset_config, source_split=args.source_split,
                                     offsets=tuple(map(int, args.offsets.split(","))),
                                     page_size=args.page_size, max_tokens=args.max_tokens, seed=args.seed,
                                     progress=lambda row: print(report.dumps(row), flush=True))
        print(report.dumps({"output": str(args.output), "splits": manifest["splits"],
                          "questions": manifest["unique_questions"], "domains": manifest["domains"]}), flush=True)
        return
    if args.command == "train":
        if args.output.exists():
            parser.error("output already exists; choose a new checkpoint path")
        config = TrainingConfig(steps=args.steps, batch_size=args.batch_size,
                                sequence_length=args.sequence_length,
                                blocks=tuple(int(b) for b in args.blocks.split(",")),
                                learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                                warmup_loss=args.warmup_loss, loss=args.loss, seed=args.seed,
                                validation_every=args.validation_every)
        if args.threads < 1 or args.validation_batches < 1:
            parser.error("threads and validation batches must be positive")
        if args.validation_data:
            assert_split_files_disjoint(args.data, args.validation_data)
        data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
        validation_sha = hashlib.sha256(args.validation_data.read_bytes()).hexdigest() if args.validation_data else None
        implementation_sha = implementation_fingerprint()
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        model = load_hf_base(args.base, rank=args.rank, alpha=args.alpha,
                             device=args.device, dtype=getattr(torch, args.dtype))
        initial_sha = None
        if args.initial_adapter:
            model, _ = load_checkpoint(args.initial_adapter, model=model, device=args.device)
            initial_sha = hashlib.sha256(args.initial_adapter.read_bytes()).hexdigest()
        fingerprint = base_fingerprint(model)
        tokenizer = None
        if args.text_data:
            from blockspec.tokenizer import LocalTokenizer
            tokenizer = LocalTokenizer(args.base)
        data = load_sequences(args.data, model.config.vocab_size, tokenizer=tokenizer)
        validation = None
        if args.validation_data:
            from .validation import FixedValidation
            validation_data = load_sequences(args.validation_data, model.config.vocab_size, tokenizer=tokenizer)
            validation = FixedValidation(validation_data, vocab_size=model.config.vocab_size,
                                           blocks=config.blocks, batches=args.validation_batches,
                                           length=config.sequence_length, bos_id=args.bos_id)
        def progress(stats):
            if "validation" in stats or stats["step"] == 1 or stats["step"] % 25 == 0 or stats["step"] == args.steps:
                print(report.dumps(stats), flush=True)
        result = train_adapter(model, data, config, bos_id=args.bos_id, progress=progress, validation=validation)
        if base_fingerprint(model) != fingerprint:
            raise RuntimeError("offline training changed frozen base weights")
        save_checkpoint(args.output, model, adapter_only=True,
                        metadata={"training": result, "training_config": asdict(config),
                                  "base_source": str(args.base), "seed": args.seed,
                                  "train_data_sha256": data_sha, "validation_data_sha256": validation_sha,
                                  "initial_adapter_sha256": initial_sha,
                                  "implementation_sha256_at_start": implementation_sha})
        print(report.dumps({"checkpoint": str(args.output), **result}), flush=True)
        return
    if min(args.base_steps, args.adapter_steps, args.rank, args.tokens, args.threads) < 1:
        parser.error("steps, rank, tokens and threads must be positive")
    if args.block_size < 2:
        parser.error("block size must be >=2")
    demo(args)


if __name__ == "__main__":
    main()
