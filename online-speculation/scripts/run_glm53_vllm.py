#!/usr/bin/env python3
"""Bounded native GLM-5.3-Flash baseline or experimental Orthrus smoke run."""

import argparse
import json
import os
from pathlib import Path
import time


def settings(args):
    values = dict(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
                  quantization="ascend", enforce_eager=True, max_num_seqs=1,
                  max_model_len=2048, max_num_batched_tokens=256,
                  gpu_memory_utilization=.90, enable_expert_parallel=True,
                  enable_prefix_caching=False, async_scheduling=False,
                  language_model_only=True, trust_remote_code=False, seed=1024)
    if args.mode == "orthrus":
        if not args.experimental_hybrid_policy:
            raise ValueError("GLM hybrid draft policy requires explicit experimental opt-in")
        if bool(args.head_dir) == bool(args.initialize_from_target):
            raise ValueError("supply a head directory or explicit smoke initialization, not both")
        if args.mask_token_id is None or not 2 <= args.block_size <= 8:
            raise ValueError("explicit mask ID and native KDA block size 2..8 required")
        from blockspec.vllm_ascend.glm53_native import HYBRID_POLICY

        values.update(speculative_config={
            "method": "custom_class",
            "model": "blockspec.vllm_ascend.glm53_proposer.GLM53OrthrusProposer",
            "num_speculative_tokens": args.block_size - 1,
            "draft_sample_method": "probabilistic", "disable_padded_drafter_batch": True,
        }, additional_config={"glm53_orthrus": dict(
            mask_token_id=args.mask_token_id, hybrid_policy=HYBRID_POLICY,
            head_dir=args.head_dir, initialize_from_target=args.initialize_from_target,
        )}, worker_extension_cls="blockspec.vllm_ascend.glm53_proposer.GLM53WorkerExtension")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "orthrus"), default="baseline")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--head-dir")
    parser.add_argument("--initialize-from-target", action="store_true")
    parser.add_argument("--experimental-hybrid-policy", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")
    options = settings(args)
    from vllm import LLM, SamplingParams

    start = time.monotonic()
    engine = LLM(**options)
    prompts = ["请用一句话解释什么是缓存。", "What is 17 times 23?",
               "Summarize the following notes in one sentence.\n" + "The server caches previously computed states. " * 80]
    rows = []
    for prompt in prompts:
        result = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        rows.append(dict(prompt=prompt, tokens=list(result.outputs[0].token_ids), text=result.outputs[0].text))
    report = dict(mode=args.mode, greedy=rows, elapsed_seconds=time.monotonic() - start,
                  settings=options, status="passed")
    if args.mode == "orthrus":
        stats = engine.llm_engine.engine_core.collective_rpc("orthrus_stats")
        if not stats or any(not s.get("verified") or not s.get("history_reads") for s in stats):
            raise AssertionError(f"no real native propose/verify execution: {stats}")
        report["workers"] = stats
        baseline = json.loads(args.output.with_name("baseline.json").read_text())
        if [r["tokens"] for r in baseline["greedy"]] != [r["tokens"] for r in rows]:
            report["status"] = "greedy_mismatch"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"mode": args.mode, "status": report["status"], "output": str(args.output)}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
