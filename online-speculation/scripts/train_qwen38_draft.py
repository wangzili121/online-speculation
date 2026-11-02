#!/usr/bin/env python3
"""Offline all-layer Qwen draft distillation on the native TP-sharded backbone."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def main():
    prepare_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="tokenized JSONL: input_ids, optional anchor_ranges")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--audit", action="store_true", help="hash frozen target and draft before/after fitting")
    parser.add_argument("--timing", action="store_true", help="measure synchronized update phases in validation workers")
    args = arguments(parser, defaults=dict(optimizer_device="cpu", checkpoint_layers=True))
    if args.online or args.steps < 1 or not args.block_size <= args.sequence_length <= args.max_model_len:
        parser.error("offline fitting requires positive steps and block_size <= sequence_length <= max_model_len")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import AutoConfig
    from vllm import LLM
    from blockspec.parallel.fitting import TokenDataset, BatchStream

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    text = getattr(config, "text_config", config)
    dataset = TokenDataset(args.data, text.vocab_size, args.sequence_length)
    stream = BatchStream(dataset, args.seed)
    args.online, args.all_draft_layers = False, True
    args.checkpoint_dir = str(args.output / "head")
    settings = native_options(args) | speculative_options(args)
    settings["additional_config"]["qwen38_orthrus"]["offline"] = True
    if args.audit or args.timing:
        settings["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    llm = LLM(**settings)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    records = []
    frozen_before = llm.collective_rpc("orthrus_target_fingerprint") if args.audit else None
    head_before = llm.collective_rpc("orthrus_head_fingerprint") if args.audit else None
    with (args.output / "loss.jsonl").open("w") as log:
        for step in range(args.steps):
            tokens, candidates = stream.batch_with_anchor_candidates(1, args.block_size)
            choices = candidates[0]
            anchor = int(choices[int(torch.randint(len(choices), (), generator=stream.rng))])
            start = time.perf_counter()
            method = "orthrus_timed_offline_step" if args.timing else "orthrus_offline_step"
            ranks = llm.collective_rpc(method, args=(tokens[0].tolist(), anchor))
            record = {"step": step + 1, "anchor": anchor, "seconds": time.perf_counter() - start,
                      "sequence_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), "ranks": ranks}
            if any(row["layers"] != text.num_hidden_layers
                   or row["gradient_layers"] != list(range(text.num_hidden_layers)) for row in ranks):
                raise RuntimeError("offline learner did not select every draft layer")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            records.append(record)
    head = llm.collective_rpc("orthrus_checkpoint")
    report = {"mode": "offline_all_draft", "data_sha256": dataset.fingerprint, "steps": args.steps,
              "eligible_records": len(dataset), "skipped_records": dataset.skipped,
              "seed": args.seed, "loss": "forward_kl", "optimizer_device": args.optimizer_device,
              "checkpoint_layers": args.checkpoint_layers, "head": head, "records": records,
              "stats": llm.collective_rpc("orthrus_stats")}
    if args.audit:
        frozen_after = llm.collective_rpc("orthrus_target_fingerprint")
        head_after = llm.collective_rpc("orthrus_head_fingerprint")
        report["audit"] = dict(target_unchanged=frozen_before == frozen_after,
                               draft_changed=head_before != head_after,
                               target_before=frozen_before, target_after=frozen_after,
                               head_before=head_before, head_after=head_after)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if frozen_before != frozen_after or head_before == head_after:
            raise AssertionError("offline fitting changed target weights or failed to publish draft updates")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
