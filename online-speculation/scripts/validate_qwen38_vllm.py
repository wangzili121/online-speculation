#!/usr/bin/env python3
"""Real native-engine smoke: run target-only and Orthrus in separate processes."""

import json
import math
from pathlib import Path
import time

from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def validation_cases(*, extended=False, soak=False, comprehensive=False, max_model_len=2048):
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
    if comprehensive:
        cases += [("length-1", "Hello", 0., 1), ("length-2", "Hello", 0., 2),
                  ("length-31", "Count from 1:", 0., 31),
                  ("length-32", "Count from 1:", 0., 32),
                  ("length-33", "Count from 1:", 0., 33),
                  ("code", "def add(a, b):\n    return", 0., 16),
                  ("json", 'Return JSON with keys "name" and "age":', 0., 16),
                  ("unicode", "\u8bf7\u7528\u4e2d\u6587\u7b80\u77ed\u4ecb\u7ecd\u5317\u4eac\u3002", 0., 16),
                  ("page-edge", {"prompt_token_ids": [3] * 1535}, 0., 33)]
    return cases


def check_calibration(observed, expected, variance):
    """Six-sigma bounded check against per-prefix target probabilities."""
    failures = []
    for token in expected:
        if any(not math.isfinite(value) or value < 0
               for value in (observed.get(token, 0), expected[token], variance[token])):
            raise AssertionError("nonfinite or negative calibration statistics")
        deviation = abs(observed.get(token, 0) - expected[token])
        limit = 6 * math.sqrt(variance[token]) + 1
        if deviation > limit:
            failures.append((token, deviation, limit))
    if failures:
        raise AssertionError(f"sampling calibration failed: {failures}")


def sampling_calibration(llm, trials, *, check_repeat=False):
    from vllm import SamplingParams

    if not 16 <= trials <= 4096:
        raise ValueError("sampling calibration requires 16..4096 trials")
    tokenizer = llm.get_tokenizer()
    allowed = [tokenizer.encode(text, add_special_tokens=False) for text in (" A", " B", " C")]
    if any(len(ids) != 1 for ids in allowed):
        raise ValueError("calibration choices must each be one token")
    allowed = [ids[0] for ids in allowed]
    llm.collective_rpc("orthrus_enable_sampling_calibration", args=(allowed,))
    observed, expected, variance = ({token: 0. for token in allowed} for _ in range(3))
    rows = 0
    repeated = None
    for seed in range(trials):
        params = SamplingParams(temperature=1., seed=10000 + seed, max_tokens=3, ignore_eos=True,
                                allowed_token_ids=allowed)
        output = llm.generate(["Choose A, B, or C:"], params, use_tqdm=False)[0].outputs[0]
        traces = llm.collective_rpc("orthrus_sampling_rows")
        traces = [trace[:3] for trace in traces]  # Engine output limits can discard surplus bonus tokens.
        if (len(output.token_ids) != 3 or not traces or any(trace != traces[0] for trace in traces)
                or [row["token"] for row in traces[0]] != list(output.token_ids)):
            raise AssertionError("native sampling rows do not match returned tokens or TP ranks")
        if seed == 0 and check_repeat:
            again = llm.generate(["Choose A, B, or C:"], params, use_tqdm=False)[0].outputs[0]
            if list(again.token_ids) != list(output.token_ids):
                raise AssertionError("fixed-head requests with the same seed are not repeatable")
            llm.collective_rpc("orthrus_sampling_rows")
            repeated = list(output.token_ids)
        # The first token is sampled by AR before drafting; only later rows test speculation.
        for row in traces[0][1:]:
            token = row["token"]
            p = dict(zip(allowed, row["probabilities"], strict=True))
            if token not in p:
                raise AssertionError("sampling escaped the allowed vocabulary")
            rows += 1
            observed[token] += 1
            for choice, probability in p.items():
                expected[choice] += probability
                variance[choice] += probability * (1 - probability)
    check_calibration(observed, expected, variance)
    return dict(trials=trials, speculative_rows=rows, allowed=allowed,
                observed=observed, expected=expected, variance=variance,
                probability_source="native_pre_sampling_logits",
                repeated_seed_tokens=repeated, passed=True)


def compare_greedy(baseline, candidate):
    expected = {item["case"]: item["tokens"] for item in baseline["outputs"] if item["temperature"] == 0}
    actual = [item for item in candidate["outputs"] if item["temperature"] == 0]
    references = {item.get("reference_case", item["case"]) for item in actual}
    if (not expected or references != expected.keys()
            or len({item["case"] for item in actual}) != len(actual)
            or any(item["tokens"] != expected[item.get("reference_case", item["case"])] for item in actual)):
        raise AssertionError(f"greedy mismatch: baseline={expected}, candidate={actual}")


def main():
    import os
    prepare_runtime_environment()
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
    freeze = os.environ.get("ORTHRUS_VALIDATE_FROZEN") == "1"
    ar_view = os.environ.get("ORTHRUS_VALIDATE_AR_VIEW") == "1"
    proposal_fixture = os.environ.get("ORTHRUS_VALIDATE_PROPOSAL_FIXTURE")
    head_digest = args.online or bool(args.head_dir)
    repeats = int(os.environ.get("ORTHRUS_VALIDATE_REPEATS", "1"))
    if not 1 <= repeats <= 100:
        raise ValueError("validation repeats must be in 1..100")
    calibration = int(os.environ.get("ORTHRUS_VALIDATE_SAMPLING_TRIALS", "0"))
    if audit or trace or freeze or ar_view or head_digest or proposal_fixture or calibration or repeats > 1:
        options["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    start = time.perf_counter()
    llm = LLM(**options)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    if trace:
        llm.collective_rpc("orthrus_enable_logits_audit",
                          args=(int(os.environ.get("ORTHRUS_VALIDATE_LOGITS_MIN_LEN", "3000")),))
    if ar_view:
        if target_only:
            raise ValueError("AR view audit requires Orthrus")
        llm.collective_rpc("orthrus_enable_ar_view_audit")
    if proposal_fixture:
        if (target_only or args.online or os.environ.get("ORTHRUS_VALIDATE_CASES") != "long-decode"
                or not os.environ.get("ORTHRUS_VALIDATE_BASELINE")):
            raise ValueError("proposal fixture requires fixed-head long-decode and its matching baseline")
        reference = json.loads(Path(os.environ["ORTHRUS_VALIDATE_BASELINE"]).read_text())
        tokens = next(row["tokens"] for row in reference["outputs"] if row["case"] == "long-decode")
        llm.collective_rpc("orthrus_set_proposal_fixture", args=(tokens, proposal_fixture))
    if audit:
        if os.environ.get("ORTHRUS_VALIDATE_NUMERIC", "1") == "1":
            llm.collective_rpc("orthrus_enable_numeric_audit")
        if not target_only and not ngram_control:
            llm.collective_rpc("orthrus_enable_cache_audit")
    frozen_before = llm.collective_rpc("orthrus_target_fingerprint") if freeze else None
    head_before = llm.collective_rpc("orthrus_head_fingerprint") if head_digest else None
    soak = os.environ.get("ORTHRUS_VALIDATE_SOAK") == "1"
    cases = validation_cases(extended=os.environ.get("ORTHRUS_VALIDATE_EXTENDED") == "1",
                             soak=soak, comprehensive=os.environ.get("ORTHRUS_VALIDATE_COMPREHENSIVE") == "1",
                             max_model_len=args.max_model_len)
    selected = os.environ.get("ORTHRUS_VALIDATE_CASES")
    if selected:
        names = set(selected.split(","))
        if names - {case[0] for case in cases}:
            raise ValueError("unknown validation cases")
        cases = [case for case in cases if case[0] in names]
    repeat_names = set(filter(None, os.environ.get("ORTHRUS_VALIDATE_REPEAT_CASES", "").split(",")))
    if repeat_names - {case[0] for case in cases}:
        raise ValueError("unknown repeated validation cases")
    cases = [(name if index == 0 else f"{name}-repeat{index}", name, prompt, temperature, length)
             for index in range(repeats) for name, prompt, temperature, length in cases
             if index == 0 or not repeat_names or name in repeat_names]
    outputs = []
    memory = []
    for name, reference, prompt, temperature, length in cases:
        parameters = SamplingParams(temperature=temperature, seed=7, max_tokens=length,
                                    ignore_eos=True, top_p=1., top_k=-1)
        if name.startswith("sampling-topk"):
            parameters = SamplingParams(temperature=temperature, seed=7, max_tokens=length,
                                        ignore_eos=True, top_p=.9, top_k=20)
        result = llm.generate([prompt], parameters)[0].outputs[0]
        if len(result.token_ids) != length:
            raise AssertionError(f"{name}: expected {length} generated tokens, got {len(result.token_ids)}")
        outputs.append({"case": name, "reference_case": reference, "temperature": temperature,
                        "tokens": list(result.token_ids), "text": result.text})
        if repeats > 1:
            memory.append(dict(case=name, workers=llm.collective_rpc("orthrus_memory")))
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
    if args.online and args.checkpoint_dir and os.environ.get("ORTHRUS_VALIDATE_REPLAY_STEPS"):
        checkpoint = llm.collective_rpc("orthrus_checkpoint")
        if os.environ.get("ORTHRUS_VALIDATE_OUTPUT"):
            Path(os.environ["ORTHRUS_VALIDATE_OUTPUT"]).write_text(json.dumps(dict(
                completed=False, stage="before_fixed_replay", outputs=outputs,
                workers=llm.collective_rpc("orthrus_stats"), checkpoint=checkpoint),
                ensure_ascii=False, indent=2) + "\n")
    if os.environ.get("ORTHRUS_VALIDATE_REPLAY_STEPS"):
        if target_only or not args.online or os.environ.get("ORTHRUS_VALIDATE_AUDIT") != "1":
            raise ValueError("fixed replay validation requires the online validation worker")
        replay = llm.collective_rpc("orthrus_validate_online_replay",
                                    args=(int(os.environ["ORTHRUS_VALIDATE_REPLAY_STEPS"]),))
    stats = llm.collective_rpc("orthrus_stats")
    evidence = {"model": args.model, "target_only": target_only, "tp": args.tp,
                "block_size": args.block_size, "gdn_prefill_backend": args.gdn_prefill_backend,
                "max_model_len": args.max_model_len, "online": args.online,
                "last_layers": args.last_layers, "update_stride": args.update_stride,
                "native_ngram_control": ngram_control,
                "proposal_fixture": proposal_fixture,
                "outputs": outputs, "workers": stats,
                "seconds": time.perf_counter() - start}
    destination = os.environ.get("ORTHRUS_VALIDATE_OUTPUT")
    if destination:
        evidence["completed"] = False
        Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    if calibration:
        evidence["sampling_calibration"] = sampling_calibration(llm, calibration, check_repeat=not args.online)
        stats = evidence["workers"] = llm.collective_rpc("orthrus_stats")
    if frozen_before is not None:
        frozen_after = llm.collective_rpc("orthrus_target_fingerprint")
        if frozen_after != frozen_before:
            raise AssertionError("native target weights changed during inference or online updates")
        evidence["target_weights"] = dict(before=frozen_before, after=frozen_after, unchanged=True)
    if head_before is not None:
        evidence["head_before"] = head_before
        evidence["head_after"] = llm.collective_rpc("orthrus_head_fingerprint")
    if memory:
        evidence["memory"] = memory
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
        if ar_view:
            assert all(item["ar_view_checks"] and all(check["passed"] for check in item["ar_view_checks"])
                       for item in stats), stats
        if proposal_fixture == "oracle":
            assert all(args.block_size in item["fixture_accepted_reads"] for item in stats), stats
        elif proposal_fixture == "reject":
            assert all(item["accepted"] == 0 and item["fixture_accepted_reads"]
                       and set(item["fixture_accepted_reads"]) == {1} for item in stats), stats
        if not selected:
            assert all(item["max_verified_tokens"] == args.block_size - 1 for item in stats), stats
        if len({(item["proposals"], item["verified"], item["accepted"], item["updates"]) for item in stats}) != 1:
            raise AssertionError(f"TP worker counters diverged: {stats}")
        if args.online and any(not math.isclose(item["loss"], stats[0]["loss"], rel_tol=1e-5, abs_tol=1e-6)
                               for item in stats):
            raise AssertionError(f"TP worker online losses diverged: {stats}")
        if args.online and args.checkpoint_dir:
            evidence["checkpoint"] = llm.collective_rpc("orthrus_checkpoint")
    if replay is not None and not all(item["passed"] for item in replay):
        if destination:
            Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        raise AssertionError(f"fixed feedback did not meet its acceptance gate: {replay}")
    if destination:
        evidence["completed"] = True
        evidence["seconds"] = time.perf_counter() - start
        Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
