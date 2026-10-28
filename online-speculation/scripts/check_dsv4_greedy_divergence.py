#!/usr/bin/env python3
"""Probe the first divergent greedy prefix using fresh target-model prefills.

Echoed prompt logprobs expose the same causal query in two prefill shapes.
No draft is used to predict that prompt position. This is a diagnostic, not
a substitute for cache rollback or full-model correctness tests.
"""

import argparse
import json
from pathlib import Path

from validate_dsv4_service import _completion, _json_request
from blockspec.vllm_ascend.artifacts import write_private_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-original", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    url = args.base_url.rstrip("/")
    reports = []
    for index, (left, right) in enumerate(zip(baseline["greedy"], candidate["greedy"], strict=True)):
        a, b = left["token_ids"], right["token_ids"]
        offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if offset is None:
            continue
        case = left["case"]
        assert case == right["case"], "snapshots use different prompts or parameters"
        prompt = case["prompt"] * int(case.get("prompt_repeat", 1))
        tokenized = _json_request(url + "/tokenize", {"model": args.model, "prompt": prompt})
        prompt_ids = tokenized["tokens"]
        assert len(prompt_ids) == left["usage"]["prompt_tokens"], "tokenizer/prompt mismatch"
        prefix = prompt_ids + a[:offset]
        base = {"model": args.model, "max_tokens": 1, "temperature": 0,
                "return_token_ids": True, "logprobs": 5}
        fresh = _json_request(url + "/v1/completions", {**base, "prompt": prefix})
        row = {"case_index": index, "divergence_index": offset,
               "baseline_token": a[offset], "candidate_token": b[offset],
               "fresh_prefix": fresh["choices"][0], "echoed_prefills": []}
        if args.replay_original:
            row["original_replay"] = _completion(url, args.model, {**case, "logprobs": 5})
        for extra in (0, 31):
            # The scored token is fixed; the extra suffix must not be visible
            # to its causal query, but changes the matrix/kernel batch shape.
            tokens = prefix + [a[offset]] * (extra + 1)
            response = _json_request(url + "/v1/completions",
                                     {**base, "prompt": tokens, "echo": True})
            assert response["usage"]["prompt_tokens"] == len(tokens)
            logprobs = response["choices"][0]["logprobs"]
            row["echoed_prefills"].append({
                "extra_future_tokens": extra,
                "scored_token": logprobs["tokens"][len(prefix)],
                "token_logprob": logprobs["token_logprobs"][len(prefix)],
                "top_logprobs": logprobs["top_logprobs"][len(prefix)],
            })
        reports.append(row)
        write_private_json(args.output, {"scope": "fresh target prefill diagnostic", "cases": reports})
        print(json.dumps({key: value for key, value in row.items()
                          if key != "original_replay"}), flush=True)
    if not reports:
        write_private_json(args.output, {"scope": "fresh target prefill diagnostic", "cases": []})


if __name__ == "__main__":
    main()
