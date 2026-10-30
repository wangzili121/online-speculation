#!/usr/bin/env python3
"""Real native-engine smoke: run target-only and Orthrus in separate processes."""

import json
from pathlib import Path
import time

from run_qwen38_vllm import arguments, native_options, speculative_options


def validation_cases(*, extended=False, soak=False, max_model_len=2048):
    cases = [("short-greedy", "Hello", 0., 8), ("short-sampling", "Hello", 1., 8)]
    if extended:
        cases += [("reject-rounds", "Count the integers from 1 to 20:", 0., 48),
                  ("chunked-prefill", "A tree has green leaves. " * 80 + "Summarize:", 0., 16),
                  ("sampling-topk", "The capital of France is", .7, 16)]
    if soak:
        if max_model_len < 1024:
            raise ValueError("soak validation requires max_model_len >= 1024")
        cases += [("long-decode", "Count the integers from 1 to 100:", 0., 128),
                  # Token IDs make the remaining context budget exact, independent of tokenization.
                  ("context-tail", {"prompt_token_ids": [3] * (max_model_len - 4)}, 0., 4),
                  ("after-context-tail", "Hello", 0., 8)]
    return cases


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
    ngram_control = os.environ.get("ORTHRUS_VALIDATE_NGRAM_CONTROL") == "1"
    if ngram_control:
        if target_only or args.online:
            raise ValueError("native ngram control cannot enable Orthrus or online updates")
        options["speculative_config"] = dict(method="ngram", num_speculative_tokens=args.block_size - 1,
                                             prompt_lookup_min=2, prompt_lookup_max=4)
    elif not target_only:
        options.update(speculative_options(args))
    audit = os.environ.get("ORTHRUS_VALIDATE_AUDIT") == "1"
    trace = os.environ.get("ORTHRUS_VALIDATE_LOGITS") == "1"
    if audit or trace:
        options["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    start = time.perf_counter()
    llm = LLM(**options)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    if trace:
        llm.collective_rpc("orthrus_enable_logits_audit")
    if audit:
        llm.collective_rpc("orthrus_enable_numeric_audit")
        if not target_only and not ngram_control:
            llm.collective_rpc("orthrus_enable_cache_audit")
    soak = os.environ.get("ORTHRUS_VALIDATE_SOAK") == "1"
    cases = validation_cases(extended=os.environ.get("ORTHRUS_VALIDATE_EXTENDED") == "1",
                             soak=soak, max_model_len=args.max_model_len)
    selected = os.environ.get("ORTHRUS_VALIDATE_CASES")
    if selected:
        names = set(selected.split(","))
        if names - {case[0] for case in cases}:
            raise ValueError("unknown validation cases")
        cases = [case for case in cases if case[0] in names]
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
    if soak and any(case[0] == "short-greedy" for case in cases):
        stop = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True,
                              stop_token_ids=[outputs[0]["tokens"][0]])
        stopped = llm.generate(["Hello"], stop)[0].outputs[0]
        if list(stopped.token_ids) != outputs[0]["tokens"][:1] or stopped.finish_reason != "stop":
            raise AssertionError(f"stop token was not respected: {stopped}")
        recovered = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=8,
                                                           ignore_eos=True))[0].outputs[0]
        if list(recovered.token_ids) != outputs[0]["tokens"]:
            raise AssertionError("request after stop inherited stale history")
        outputs.append({"case": "after-stop", "temperature": 0.,
                        "tokens": list(recovered.token_ids), "text": recovered.text})
    replay = None
    if os.environ.get("ORTHRUS_VALIDATE_REPLAY_STEPS"):
        if target_only or not args.online or os.environ.get("ORTHRUS_VALIDATE_AUDIT") != "1":
            raise ValueError("fixed replay validation requires the online validation worker")
        replay = llm.collective_rpc("orthrus_validate_online_replay",
                                    args=(int(os.environ["ORTHRUS_VALIDATE_REPLAY_STEPS"]),))
    stats = llm.collective_rpc("orthrus_stats")
    evidence = {"model": args.model, "target_only": target_only, "tp": args.tp,
                "block_size": args.block_size, "gdn_prefill_backend": args.gdn_prefill_backend,
                "native_ngram_control": ngram_control,
                "outputs": outputs, "workers": stats,
                "seconds": time.perf_counter() - start}
    if replay is not None:
        evidence["fixed_replay"] = replay
    baseline = os.environ.get("ORTHRUS_VALIDATE_BASELINE")
    destination = os.environ.get("ORTHRUS_VALIDATE_OUTPUT")
    if baseline:
        reference = json.loads(Path(baseline).read_text())
        if reference["gdn_prefill_backend"] != args.gdn_prefill_backend:
            raise AssertionError("baseline and candidate must use the same GDN prefill backend")
        evidence["greedy_matches_baseline"] = False
        # Preserve outputs and diagnostics even when a comparison fails.
        if destination:
            Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        compare_greedy(reference, evidence)
        evidence["greedy_matches_baseline"] = True
    if not target_only and not ngram_control:
        assert all(item["orthrus"] and item["verified"] > 0 for item in stats), stats
        assert all(item["history_backend"] == "shared_target_snapshot"
                   and item["history_reads"] == item["proposals"] for item in stats), stats
        if os.environ.get("ORTHRUS_VALIDATE_AUDIT") == "1":
            assert all(item["cache_readonly_checks"] > 0 for item in stats), stats
        if args.online:
            assert all(item["updates"] > 0 for item in stats), stats
        if not selected:
            assert all(item["max_verified_tokens"] == args.block_size - 1 for item in stats), stats
        if len({(item["proposals"], item["verified"], item["accepted"], item["updates"]) for item in stats}) != 1:
            raise AssertionError(f"TP worker counters diverged: {stats}")
        if args.online and args.checkpoint_dir:
            evidence["checkpoint"] = llm.collective_rpc("orthrus_checkpoint")
    if destination:
        Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
