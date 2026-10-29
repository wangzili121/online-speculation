#!/usr/bin/env python3
"""Real native-engine smoke: run target-only and Orthrus in separate processes."""

import json
from pathlib import Path
import time

from run_qwen38_vllm import arguments, native_options, speculative_options


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
    outputs = []
    for temperature in (0., 1.):
        parameters = SamplingParams(temperature=temperature, seed=7, max_tokens=8,
                                    ignore_eos=True, top_p=1., top_k=-1)
        result = llm.generate(["Hello"], parameters)[0].outputs[0]
        outputs.append({"temperature": temperature, "tokens": list(result.token_ids), "text": result.text})
    stats = llm.collective_rpc("orthrus_stats")
    evidence = {"model": args.model, "target_only": target_only, "tp": args.tp,
                "block_size": args.block_size, "outputs": outputs, "workers": stats,
                "seconds": time.perf_counter() - start}
    if not target_only:
        assert all(item["orthrus"] and item["verified"] > 0 for item in stats), stats
        if args.online:
            assert all(item["updates"] > 0 for item in stats), stats
    destination = os.environ.get("ORTHRUS_VALIDATE_OUTPUT")
    if destination:
        Path(destination).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
