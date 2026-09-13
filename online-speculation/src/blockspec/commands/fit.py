"""Local Qwen3 draft fitting, exact-boundary resume, and a synthetic pipeline check."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from blockspec.data import assert_split_files_disjoint
from blockspec.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.fitting import FitConfig, TokenDataset, Trainer, frozen_fingerprint
from blockspec.parallel.weights import file_sha256, load_ar_base, public_key_map


def fit_config(args, *, demo=False):
    defaults = {"steps": 48 if demo else 1000, "batch_size": 2 if demo else 1,
                "sequence_length": 24 if demo else 256, "anchors_per_sequence": 2 if demo else 4,
                "accumulate": 2 if demo else 1, "chunk_rows": 8 if demo else 32,
                "learning_rate": .002 if demo else .0001, "warmup_steps": 4 if demo else 50,
                "precision": "fp32", "backend": "sdpa", "seed": 731, "optimizer_impl": "single"}
    return FitConfig(**{key: default if getattr(args, key) is None else getattr(args, key)
                        for key, default in defaults.items()})


def synthetic_demo(args):
    config = fit_config(args, demo=True)
    torch.manual_seed(config.seed)
    architecture = DualViewConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=8, block_size=4, mask_token_id=1)
    teacher = DualViewDecoder(architecture).to(args.device)
    for layer in teacher.layers:
        layer.attention.draft.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad], lr=.006, foreach=False)
    rng = torch.Generator().manual_seed(config.seed + 10)
    for _ in range(args.teacher_steps):
        offsets = torch.randint(8, (8, 1), generator=rng)
        tokens = ((torch.arange(config.sequence_length)[None] + offsets) % 8 + 2).to(args.device)
        loss = F.cross_entropy(teacher(tokens[:, :-1]).logits.flatten(0, 1), tokens[:, 1:].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    ntp_loss = float(loss.detach())
    with tempfile.TemporaryDirectory(prefix="blockspec-dual-fit-") as temporary:
        folder = Path(temporary)
        base = folder / "ar"
        base.mkdir()
        public_config = architecture.to_dict() | {"model_type": "qwen3"}
        for key in ("block_size", "mask_token_id"):
            public_config.pop(key)
        (base / "config.json").write_text(json.dumps(public_config))
        weights = teacher.state_dict()
        save_file({public: weights[own].detach().cpu().clone().contiguous()
                   for own, public in public_key_map(architecture, include_draft=False).items()},
                  base / "model.safetensors")
        paths = []
        for name, extra, count in (("training", 8, 16), ("validation", 40, 8)):
            path = folder / (name + ".jsonl")
            rows = [{"input_ids": [(j + i) % 8 + 2 for j in range(config.sequence_length + extra + i)]}
                    for i in range(count)]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            paths.append(path)
        assert_split_files_disjoint(*paths)
        training, validation = [TokenDataset(path, architecture.vocab_size, config.sequence_length) for path in paths]

        def initialize():
            own = load_ar_base(base, block_size=4, mask_token_id=1, device=args.device)
            return Trainer(own, training, config)

        full = initialize()
        initial = full.evaluate(validation)
        full_records = full.run()
        final = full.evaluate(validation)
        split = initialize()
        part = split.run(config.steps // 2)
        split.save(folder / "interrupted.pt")
        resumed = Trainer.resume(folder / "interrupted.pt", paths[0], device=args.device)
        part += resumed.run()
        equal_updates = all(all(a[key] == b[key] for key in ("loss", "gradient_norm", "learning_rate", "step"))
                            for a, b in zip(full_records, part, strict=True))
        equal_parameters = all(torch.equal(value, resumed.model.state_dict()[key])
                               for key, value in full.model.state_dict().items())
        frozen = full.base_fingerprint == frozen_fingerprint(full.model) == frozen_fingerprint(resumed.model)
        branch = MaskedAttentionBranch(full.model.eval())
        prompt = torch.tensor([[2, 3, 4, 5]], device=args.device)
        ar = generate_ar(branch, prompt, 24)
        speculative = generate(branch, prompt, 24, block_size=4, audit_cache=True)
        passed = equal_updates and equal_parameters and frozen and ar.tokens == speculative.tokens and final < initial
        full.save(args.output)
        return {"mode": "synthetic-demo", "config": asdict(config), "teacher_ntp_loss": ntp_loss,
                "validation_kl_before": initial, "validation_kl_after": final,
                "updates_identical_after_resume": equal_updates, "parameters_identical_after_resume": equal_parameters,
                "frozen_parameters_unchanged": frozen, "greedy_matches_ar": ar.tokens == speculative.tokens,
                "full_run_seconds": sum(row["seconds"] for row in full_records),
                "training_data_sha256": training.fingerprint, "validation_data_sha256": validation.fingerprint,
                "pass": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "resume", "demo"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--teacher-steps", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    for name in ("steps", "batch-size", "sequence-length", "anchors-per-sequence", "accumulate", "chunk-rows", "warmup-steps", "seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--backend", choices=("eager", "sdpa"))
    parser.add_argument("--optimizer-impl", choices=("single", "fused"))
    args = parser.parse_args()
    if args.output.exists() or (args.summary is not None and args.summary.exists()):
        parser.error("checkpoint and summary paths must be new")
    if args.threads < 1 or args.log_every < 1 or args.teacher_steps < 1:
        parser.error("positive thread count, log cadence and teacher steps required")
    torch.set_num_threads(args.threads)
    if args.mode == "demo":
        if any(value is not None for value in (args.base, args.checkpoint, args.data, args.validation, args.stop_after)):
            parser.error("demo constructs its own temporary synthetic model and data")
        result = synthetic_demo(args)
    else:
        if args.data is None:
            parser.error("tokenized training JSONL required")
        if args.validation is not None:
            assert_split_files_disjoint(args.data, args.validation)
        if args.mode == "train":
            if args.base is None or args.mask_token_id is None or args.checkpoint is not None:
                parser.error("train requires an AR base and explicit mask token")
            config = fit_config(args)
            model = load_ar_base(args.base, block_size=args.block_size, mask_token_id=args.mask_token_id, device=args.device)
            data = TokenDataset(args.data, model.config.vocab_size, config.sequence_length)
            trainer = Trainer(model, data, config)
        else:
            if args.checkpoint is None or args.base is not None:
                parser.error("resume requires a complete training checkpoint")
            schedule = ("steps", "batch_size", "sequence_length", "anchors_per_sequence", "accumulate", "chunk_rows",
                        "learning_rate", "warmup_steps", "precision", "backend", "seed", "optimizer_impl")
            if any(getattr(args, key) is not None for key in schedule):
                parser.error("resume keeps the saved full schedule; --stop-after sets an interruption boundary")
            trainer = Trainer.resume(args.checkpoint, args.data, device=args.device)
        validation = (TokenDataset(args.validation, trainer.model.config.vocab_size, trainer.config.sequence_length)
                      if args.validation is not None else None)
        initial = trainer.evaluate(validation) if validation is not None else None

        def progress(row):
            if row["step"] % args.log_every == 0:
                print(report.dumps(row), flush=True)

        records = trainer.run(args.stop_after, progress)
        final = trainer.evaluate(validation) if validation is not None else None
        frozen = frozen_fingerprint(trainer.model) == trainer.base_fingerprint
        if not frozen:
            raise RuntimeError("frozen parameters changed during fitting")
        trainer.save(args.output)
        result = {"mode": args.mode, "step": trainer.step, "config": asdict(trainer.config),
                  "updates_this_run": len(records), "training_seconds": sum(row["seconds"] for row in records),
                  "validation_kl_before": initial, "validation_kl_after": final,
                  "training_data_sha256": trainer.data.fingerprint,
                  "validation_data_sha256": None if validation is None else validation.fingerprint,
                  "frozen_parameters_unchanged": frozen, "source": trainer.model.source, "pass": frozen}
    result.update(script_sha256=file_sha256(__file__), output=str(args.output))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("x") as handle:
            report.dump(result, handle, indent=2)
            handle.write("\n")
    print(report.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
