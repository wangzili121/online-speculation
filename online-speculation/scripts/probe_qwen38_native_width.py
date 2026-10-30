#!/usr/bin/env python3
"""Diagnose native decode/verification differences on the same loaded target."""

import json
import os
from pathlib import Path

from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def main():
    prepare_runtime_environment()
    from vllm import LLM, SamplingParams

    args = arguments()
    if args.online or args.block_size != 32:
        raise ValueError("width diagnostics require a fixed K32 head")
    options = native_options(args) | speculative_options(args)
    options["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    llm = LLM(**options)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    llm.collective_rpc("orthrus_enable_logits_audit", args=(0,))
    llm.collective_rpc("orthrus_enable_projection_audit")
    llm.collective_rpc("orthrus_enable_cache_audit")
    rows = []
    report = dict(diagnostic_only=True, model=args.model, tp=args.tp, outputs=rows)
    destination = Path(os.environ["ORTHRUS_WIDTH_OUTPUT"])
    for limit in (0, 1, 7, 15, 16, 31):
        llm.collective_rpc("orthrus_set_proposal_limit", args=(limit,))
        if limit == 31:
            llm.collective_rpc("orthrus_enable_ar_view_audit")
        result = llm.generate(['Return JSON with keys "name" and "age":'],
            SamplingParams(temperature=0, seed=7, max_tokens=16, ignore_eos=True), use_tqdm=False)[0].outputs[0]
        traces = llm.collective_rpc("orthrus_take_logits_audit")
        row = dict(draft_limit=limit, tokens=list(result.token_ids), text=result.text,
                   matches_no_draft=list(result.token_ids) == (rows[0]["tokens"] if rows else list(result.token_ids)),
                   logits=traces, workers=llm.collective_rpc("orthrus_stats"))
        rows.append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("draft_limit", "tokens", "matches_no_draft")}), flush=True)
    report["projection_audit"] = llm.collective_rpc("orthrus_projection_audit")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["projection_audit"]), flush=True)


if __name__ == "__main__":
    main()
