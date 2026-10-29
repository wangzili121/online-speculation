#!/usr/bin/env python3
"""Real native-engine smoke: run target-only and Orthrus in separate processes."""

import json
from pathlib import Path
import time

from run_qwen38_vllm import arguments, native_options, speculative_options


def compare_greedy(baseline, candidate):
    expected = {item["case"]: item["tokens"] for item in baseline["outputs"] if item["temperature"] == 0}
    actual = {item["case"]: item["tokens"] for item in candidate["outputs"] if item["temperature"] == 0}
    if not expected or expected != actual:
        raise AssertionError(f"greedy mismatch: baseline={expected}, candidate={actual}")


def main():
    import os
    from vllm import LLM, SamplingParams

    args = arguments()
    options = native_options(args)
    target_only = os.environ.get("ORTHRUS_VALIDATE_TARGET_ONLY") == "1"
    if not target_only:
        options.update(speculative_options(args))
    if os.environ.get("ORTHRUS_VALIDATE_AUDIT") == "1":
        options["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    start = time.perf_counter()
    llm = LLM(**options)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    if os.environ.get("ORTHRUS_VALIDATE_AUDIT") == "1":
        llm.collective_rpc("orthrus_enable_numeric_audit")
    cases = [("short-greedy", "Hello", 0., 8), ("short-sampling", "Hello", 1., 8)]
    if os.environ.get("ORTHRUS_VALIDATE_EXTENDED") == "1":
        cases += [("reject-rounds", "Count the integers from 1 to 20:", 0., 48),
                  ("chunked-prefill", "A tree has green leaves. " * 80 + "Summarize:", 0., 16),
                  ("sampling-topk", "The capital of France is", .7, 16)]
    outputs = []
    for name, prompt, temperature, length in cases:
        parameters = SamplingParams(temperature=temperature, seed=7, max_tokens=length,
                                    ignore_eos=True, top_p=1., top_k=-1)
        if name == "sampling-topk":
            parameters = SamplingParams(temperature=temperature, seed=7, max_tokens=length,
                                        ignore_eos=True, top_p=.9, top_k=20)
        result = llm.generate([prompt], parameters)[0].outputs[0]
        if len(result.token_ids) != length:
            raise AssertionError(f"{name}: expected {length} generated tokens, got {len(result.token_ids)}")
        outputs.append({"case": name, "temperature": temperature,
                        "tokens": list(result.token_ids), "text": result.text})
    stats = llm.collective_rpc("orthrus_stats")
    evidence = {"model": args.model, "target_only": target_only, "tp": args.tp,
                "block_size": args.block_size, "gdn_prefill_backend": args.gdn_prefill_backend,
                "outputs": outputs, "workers": stats,
                "seconds": time.perf_counter() - start}
    baseline = os.environ.get("ORTHRUS_VALIDATE_BASELINE")
    if baseline:
        reference = json.loads(Path(baseline).read_text())
        if reference["gdn_prefill_backend"] != args.gdn_prefill_backend:
            raise AssertionError("baseline and candidate must use the same GDN prefill backend")
        compare_greedy(reference, evidence)
        evidence["greedy_matches_baseline"] = True
    if not target_only:
        assert all(item["orthrus"] and item["verified"] > 0 for item in stats), stats
        if args.online:
            assert all(item["updates"] > 0 for item in stats), stats
        assert all(item["max_verified_tokens"] == args.block_size - 1 for item in stats), stats
        if len({(item["proposals"], item["verified"], item["accepted"], item["updates"]) for item in stats}) != 1:
            raise AssertionError(f"TP worker counters diverged: {stats}")
    destination = os.environ.get("ORTHRUS_VALIDATE_OUTPUT")
    if destination:
        Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
