#!/usr/bin/env python3
"""Measure real draft cost and a controlled all-accepted greedy upper bound."""

import argparse
import json
from pathlib import Path
import statistics
import time

from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def counter_delta(before, after):
    if not before or len(before) != len(after):
        raise ValueError("matching worker counters required")
    values = [{name: new.get(name, 0) - old.get(name, 0)
               for name in ("proposals", "verified", "accepted", "full_blocks", "updates")}
              for old, new in zip(before, after, strict=True)]
    if any(row != values[0] for row in values[1:]):
        raise AssertionError("TP ranks disagree on measured draft counters")
    return values[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("target", "orthrus"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", default="Count the integers from 1 to 100:")
    args = arguments(parser)
    if (args.output.exists() or args.online or args.all_draft_layers or args.block_size != 32
            or args.repeats < 2 or not 64 <= args.tokens <= 512):
        parser.error("require a new output, fixed K32 head, 2+ repeats and 64..512 output tokens")
    reference = None
    if args.stage == "orthrus":
        if args.reference is None:
            parser.error("Orthrus requires the matching target reference")
        reference = json.loads(args.reference.read_text())
        expected = dict(model=args.model, tp=args.tp, prompt=args.prompt, tokens=args.tokens,
                        max_model_len=args.max_model_len,
                        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                        gdn_prefill_backend=args.gdn_prefill_backend)
        if not reference.get("passed") or reference.get("settings") != expected:
            parser.error("target reference must match model, TP, prompt, budget and runtime options")
    prepare_runtime_environment()
    from vllm import LLM, SamplingParams

    options = native_options(args)
    if args.stage == "orthrus":
        options.update(speculative_options(args))
    options["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    report = dict(passed=False, scope="warmed_single_request_e2e",
                  settings=dict(model=args.model, tp=args.tp, prompt=args.prompt, tokens=args.tokens,
                                max_model_len=args.max_model_len,
                                kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                                gdn_prefill_backend=args.gdn_prefill_backend),
                  head_dir=args.head_dir, stage=args.stage, cases=[])
    llm = None
    try:
        start = time.perf_counter()
        llm = LLM(**options)
        if args.gdn_prefill_backend == "cann":
            llm.collective_rpc("orthrus_use_cann_gdn_prefill")
        report["startup_seconds"] = time.perf_counter() - start
        params = SamplingParams(temperature=0, seed=1234, max_tokens=args.tokens, ignore_eos=True)
        modes = ("target",) if args.stage == "target" else ("real_head", "oracle_keep_real_draft")
        for mode in modes:
            if mode == "oracle_keep_real_draft":
                # 先运行真实 draft，再替换候选；目标模型仍正常验证，不伪造接受结果。
                llm.collective_rpc("orthrus_set_proposal_fixture",
                                   args=(reference["reference_tokens"], "oracle"))
            warm = time.perf_counter()
            llm.generate([args.prompt], params, use_tqdm=False)
            entry = dict(mode=mode, warmup_seconds=time.perf_counter() - warm, samples=[])
            report["cases"].append(entry)
            for repeat in range(args.repeats):
                before = llm.collective_rpc("orthrus_stats")
                start = time.perf_counter()
                request = llm.generate([args.prompt], params, use_tqdm=False)[0]
                seconds = time.perf_counter() - start
                after = llm.collective_rpc("orthrus_stats")
                tokens = list(request.outputs[0].token_ids)
                if len(tokens) != args.tokens:
                    raise AssertionError("measurement must reach the requested token budget")
                row = dict(repeat=repeat, seconds=seconds, output_tokens=len(tokens),
                           tokens_per_second=len(tokens) / seconds)
                if args.stage == "target":
                    if "reference_tokens" not in report:
                        report["reference_tokens"] = tokens
                    if tokens != report["reference_tokens"]:
                        raise AssertionError("target greedy output changed between repeats")
                else:
                    delta = counter_delta(before, after)
                    row["counters"] = delta
                    row["greedy_matches_target"] = tokens == reference["reference_tokens"]
                    if delta["updates"] or delta["verified"] < 1:
                        raise AssertionError("need neural speculation without online updates")
                    row["accepted_per_verified_round"] = delta["accepted"] / delta["verified"]
                    # 所有验证轮都提交满 31 个候选时，才能从现有计数精确恢复分母。
                    if delta["full_blocks"] == delta["verified"]:
                        row["candidate_acceptance"] = delta["accepted"] / (31 * delta["verified"])
                    if not row["greedy_matches_target"]:
                        raise AssertionError("greedy output differs from matching target run")
                    if mode == "oracle_keep_real_draft" and row.get("candidate_acceptance") != 1:
                        raise AssertionError("oracle did not achieve full candidate acceptance")
                entry["samples"].append(row)
                print(json.dumps(dict(mode=mode, **row)), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            entry["median_tokens_per_second"] = statistics.median(
                sample["tokens_per_second"] for sample in entry["samples"])
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        # 引擎在独立进程/容器中运行，由有超时的外层任务释放 worker 与设备。
        if llm is not None:
            del llm
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
