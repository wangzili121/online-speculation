#!/usr/bin/env python3
"""Bounded native GLM-5.3-Flash baseline or experimental Orthrus smoke run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def settings(args):
    values = dict(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
                  quantization="ascend", enforce_eager=True, max_num_seqs=1,
                  max_model_len=args.max_model_len, max_num_batched_tokens=args.max_num_batched_tokens,
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
        from blockspec.vllm_ascend.glm53_native import BIDIRECTIONAL_POLICY, HYBRID_POLICY

        extra = dict(mask_token_id=args.mask_token_id,
                     hybrid_policy=BIDIRECTIONAL_POLICY if args.kda_mode == "bidirectional" else HYBRID_POLICY,
                     head_dir=args.head_dir, initialize_from_target=args.initialize_from_target,
                     audit_cache=args.audit_cache)
        if args.kda_backward_weight is not None:
            if args.kda_mode != "bidirectional" or not args.initialize_from_target:
                raise ValueError("backward weight is only valid for bidirectional smoke initialization")
            extra["kda_backward_weight"] = args.kda_backward_weight
        values.update(speculative_config={
            "method": "custom_class",
            "model": "blockspec.vllm_ascend.glm53_proposer.GLM53OrthrusProposer",
            "num_speculative_tokens": args.block_size - 1,
            "draft_sample_method": "probabilistic", "disable_padded_drafter_batch": True,
        }, hf_overrides={"orthrus_glm53": extra},
            worker_extension_cls="blockspec.vllm_ascend.glm53_proposer.GLM53WorkerExtension")
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
    parser.add_argument("--kda-mode", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--kda-backward-weight", type=float)
    parser.add_argument("--audit-cache", action="store_true")
    parser.add_argument("--save-head", type=Path)
    parser.add_argument("--baseline", type=Path)
    # TP1 uses 4352-token MLA pages and 2048-token KDA pages in this runtime.
    parser.add_argument("--max-model-len", type=int, default=34816)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--long-prompt-repeats", type=int, default=80)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    args = parser.parse_args()
    identity = {name: hashlib.sha256((Path(args.model) / name).read_bytes()).hexdigest() for name in
                ("config.json", "quant_model_weights.safetensors.index.json", "quant_model_description.json")}
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")
    options = settings(args)
    from vllm import LLM, SamplingParams

    start = time.monotonic()
    engine = LLM(**options)
    prompts = ["请用一句话解释什么是缓存。", "What is 17 times 23?",
               "Summarize the following notes in one sentence.\n"
               + "The server caches previously computed states. " * args.long_prompt_repeats]
    rows = []
    for prompt in prompts:
        result = engine.generate([prompt], SamplingParams(temperature=args.temperature,
            max_tokens=args.max_output_tokens, ignore_eos=args.ignore_eos, seed=1024), use_tqdm=False)[0]
        rows.append(dict(prompt=prompt, tokens=list(result.outputs[0].token_ids), text=result.outputs[0].text))
    report = dict(mode=args.mode, greedy=rows, temperature=args.temperature,
                  model_identity=identity, generation=dict(max_tokens=args.max_output_tokens,
                      ignore_eos=args.ignore_eos, seed=1024, temperature=args.temperature),
                  elapsed_seconds=time.monotonic() - start,
                  settings=options, status="passed")
    root = Path(__file__).resolve().parents[1]
    report["code_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "scripts/run_glm53_vllm.py", "scripts/run_in_runtime.py",
        "src/blockspec/vllm_ascend/glm53_proposer.py", "src/blockspec/vllm_ascend/glm53_native.py",
        "src/blockspec/vllm_ascend/glm53_ops.py", "src/blockspec/vllm_ascend/qwen38_runtime.py",
        "src/blockspec/vllm_ascend/qwen38_proposer.py")}
    if args.mode == "orthrus":
        stats = engine.llm_engine.engine_core.collective_rpc("orthrus_stats")
        if not stats or any(not s.get("verified") or not s.get("history_reads") for s in stats):
            raise AssertionError(f"no real native propose/verify execution: {stats}")
        if args.kda_mode == "bidirectional" and any(not s.get("kda_reverse_calls") for s in stats):
            raise AssertionError("bidirectional KDA was not actually executed")
        if args.audit_cache and any(s.get("cache_audits") != s.get("proposals") for s in stats):
            raise AssertionError("not every proposal passed a target-cache audit")
        report["workers"] = stats
        if args.save_head:
            report["head_exports"] = engine.llm_engine.engine_core.collective_rpc(
                "orthrus_save_head", args=(str(args.save_head),))
        if args.head_dir:
            report["head_digests"] = engine.llm_engine.engine_core.collective_rpc("orthrus_head_digest")
        if args.temperature == 0:
            baseline = json.loads((args.baseline or args.output.with_name("baseline.json")).read_text())
            if (baseline.get("temperature", 0) != 0 or baseline["model_identity"] != identity
                    or baseline["generation"] != report["generation"]
                    or [r["prompt"] for r in baseline["greedy"]] != prompts):
                raise ValueError("baseline does not match this target and greedy workload")
            if [r["tokens"] for r in baseline["greedy"]] != [r["tokens"] for r in rows]:
                report["status"] = "greedy_mismatch"
        else:
            report["sampling_scope"] = "native propose/verify smoke only, not a distribution-equivalence test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"mode": args.mode, "status": report["status"], "output": str(args.output)}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
