"""Bounded real-weight integration check, not a publication-speed benchmark.

Uses only the base checkpoint. The offline and online adapters are initialized
and trained here, without loading a published draft adapter. Emits stdout only.
"""

import argparse

from blockspec import reporting as report

import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint, load_hf_base
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.distillation import paired_batch
from blockspec_ablation.online import OnlineConfig, OnlineLearner
from blockspec.tokenizer import LocalTokenizer
from blockspec_ablation.training import TrainingConfig, train_adapter
from blockspec_ablation.tree import generate_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--sampler", choices=["linear", "tree"], default="linear")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260905)
    model = load_hf_base(args.base, rank=8, device=args.device, dtype=getattr(torch, args.dtype))
    model.train_adapters_only()
    tokenizer = LocalTokenizer(args.base)
    frozen = base_fingerprint(model)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    text = "The following is a clear explanation of how a computer works. A computer"
    prompt = torch.tensor([[0] + tokenizer.encode(text)], device=args.device)
    # Force nonempty independent sequences; generated trajectories are only a smoke dataset.
    teacher = generate_ar(model, prompt, max(48, args.tokens))
    sequence = torch.tensor(prompt[0].tolist() + teacher.tokens)
    clean = sequence[:32][None].to(args.device)
    paired = paired_batch(clean, 4, noisy=torch.randint(model.config.vocab_size, clean.shape,
                                                       device=args.device))
    with torch.no_grad():
        clean_logits = model(clean)
        packed_logits = model(paired.tokens, positions=paired.positions, allowed=paired.allowed,
                              adapter_mask=paired.adapter_mask)[:, :clean.shape[1]]
        teacher_max_error = float((clean_logits.float() - packed_logits.float()).abs().max())
        teacher_argmax_agreement = float((clean_logits.argmax(-1) == packed_logits.argmax(-1)).float().mean())
    print(report.dumps({"stage": "loaded", "base_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                                   if not n.endswith((".lora_A", ".lora_B"))),
                      "adapter_parameters": sum(p.numel() for p in model.adapter_parameters()),
                      "paired_teacher_max_abs_error": teacher_max_error,
                      "paired_teacher_argmax_agreement": teacher_argmax_agreement,
                      "base_sample": tokenizer.decode(teacher.tokens[:24])}), flush=True)
    training = train_adapter(model, [sequence],
                              TrainingConfig(steps=args.train_steps, sequence_length=32, blocks=(2, 4),
                                             warmup_steps=min(2, args.train_steps)))
    assert base_fingerprint(model) == frozen
    initial = adapter_state(model)
    generate = generate_tree if args.sampler == "tree" else generate_speculative
    ar = generate_ar(model, prompt, args.tokens)
    static = generate(model, prompt, args.tokens, block_size=4,
                                  generator=torch.Generator(device=args.device).manual_seed(43))
    learner = OnlineLearner(model, OnlineConfig(stride=2, replay_blocks=1, learning_rate=1e-4,
                                               loss="forward_kl"))
    online = generate(model, prompt, args.tokens, block_size=4, learner=learner,
                                  generator=torch.Generator(device=args.device).manual_seed(43))
    assert base_fingerprint(model) == frozen
    changed = any(not torch.equal(value, adapter_state(model)[key]) for key, value in initial.items())
    print(report.dumps({"stage": "integration", "sampler": args.sampler, "training": training, "ar": ar.summary(),
                      "static": static.summary(), "online": online.summary(),
                      "greedy_identical": ar.tokens == static.tokens == online.tokens,
                      "base_unchanged": True, "online_adapter_changed": changed,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                      if args.device.startswith("cuda") else None,
                      "scope": "single-prompt integration, not a trained-adapter speed result"}), flush=True)


if __name__ == "__main__":
    main()
