#!/usr/bin/env python3
"""Diagnose batching and probability lifetime in a real DSV4 engine.

The worker observes request IDs and probability buffers without replacing
tokens, logits, sampling, scheduling or cache operations. Generated outputs
are diagnostics; random outputs need not match across different batches.
"""

import argparse
from collections import Counter
import functools
import json
from pathlib import Path


class BatchStateWorker:
    def batch_probe_install(self, directory):
        self._batch_probe = {"phase": "initial", "events": [], "wrapped": []}
        self._batch_probe_path = Path(directory) / f"events-rank{self.rank}.jsonl"
        runner = self.model_runner
        for name, probe in (("_get_spec_decode_draft_probs", self._batch_probe_get),
                            ("propose_draft_token_ids", self._batch_probe_propose)):
            original = getattr(runner, name)

            def wrap(method, callback):
                @functools.wraps(method)
                def wrapped(*args, **kwargs):
                    return callback(method, *args, **kwargs)
                return wrapped

            self._batch_probe["wrapped"].append((runner, name, original))
            setattr(runner, name, wrap(original, probe))
        return {"rank": self.rank, "async_scheduling": runner.use_async_scheduling}

    def batch_probe_phase(self, phase):
        self._batch_probe["phase"] = phase

    def _batch_probe_record(self, event):
        event.update(phase=self._batch_probe["phase"], rank=self.rank)
        self._batch_probe["events"].append(event)
        with self._batch_probe_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def _batch_probe_get(self, original, metadata):
        runner = self.model_runner
        ids = list(runner.input_batch.req_ids)
        cached = list(runner._draft_prob_req_ids or [])
        counts = list(metadata.num_draft_tokens)
        missing = [req for req, count in zip(ids, counts, strict=True)
                   if count and req not in cached]
        result = original(metadata)
        random_requests = [req for req, count in zip(ids, counts, strict=True)
                           if count and runner.requests[req].sampling_params.temperature > 0]
        self._batch_probe_record({
            "kind": "get", "requests": ids, "cached_requests": cached,
            "counts": counts, "missing": missing,
            "random_requests": random_requests,
            "all_greedy": bool(runner.input_batch.sampling_metadata.all_greedy),
            "has_probs": result is not None,
            "rows": result.shape[0] if result is not None else 0,
        })
        return result

    def _batch_probe_propose(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        runner = self.model_runner
        self._batch_probe_record({
            "kind": "propose", "requests": list(runner.input_batch.req_ids),
            "cached_requests": list(runner._draft_prob_req_ids or []),
            "all_greedy": bool(runner.input_batch.sampling_metadata.all_greedy),
            "has_probs": runner._draft_probs is not None,
        })
        return result

    def batch_probe_report(self):
        events = self._batch_probe["events"]
        gets = [event for event in events if event["kind"] == "get" and sum(event["counts"])]
        missing = [event for event in gets if event["random_requests"] and not event["has_probs"]]
        stale = [event for event in gets if event["all_greedy"] and event["missing"]
                 and event["cached_requests"]]
        return {"rank": self.rank, "verification_steps": len(gets),
                "missing_random_probabilities": len(missing),
                "stale_greedy_probability_steps": len(stale),
                "missing_by_phase": dict(Counter(event["phase"] for event in missing)),
                "stale_by_phase": dict(Counter(event["phase"] for event in stale))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kv_cache_mib <= 0 or args.tokens <= 0:
        parser.error("KV cache size and output length must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams

    options = {}
    if args.head:
        options["speculative_config"] = {
            "method": "orthrus", "model": args.head, "num_speculative_tokens": 31,
            "parallel_drafting": True, "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard", "quantization": "ascend", "enforce_eager": True,
        }
    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp, enable_expert_parallel=True,
        quantization="ascend", tokenizer_mode="deepseek_v4", enforce_eager=True,
        async_scheduling=False, compilation_config={"cudagraph_mode": "NONE"},
        additional_config={"multistream_dsv4_dsa_overlap": False,
                           "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": False},
                           "enable_cpu_binding": True, "multistream_overlap_shared_expert": True},
        max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=8192,
        gpu_memory_utilization=0.5, kv_cache_memory_bytes=args.kv_cache_mib * 1024 ** 2,
        enable_prefix_caching=False, block_size=128,
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 128},
        worker_extension_cls="check_dsv4_batch_state.BatchStateWorker", **options,
    )
    report = {"scope": "batch diagnostics, not full-model acceptance", "options": vars(args).copy()}
    report["options"]["output"] = str(args.output)
    report["workers_installed"] = llm.collective_rpc("batch_probe_install", args=(str(args.output),))
    records = {}

    def generate(phase, temperatures, serial=False):
        llm.collective_rpc("batch_probe_phase", args=(phase,))
        params = [SamplingParams(temperature=t, seed=i, top_p=0.95, max_tokens=args.tokens,
                                 logprobs=5, ignore_eos=True) for i, t in enumerate(temperatures)]
        prompts = ["A random color is"] * len(params)
        if serial:
            outputs = [llm.generate([prompt], [param], use_tqdm=False)[0]
                       for prompt, param in zip(prompts, params, strict=True)]
        else:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        records[phase] = [list(output.outputs[0].token_ids) for output in outputs]

    generate("warmup", [0.0])
    generate("greedy_serial", [0.0] * 4, serial=True)
    generate("greedy_batch", [0.0] * 4)
    generate("greedy_after_batch", [0.0] * 4, serial=True)
    generate("random_serial", [0.8] * 4, serial=True)
    generate("random_batch", [0.8] * 4)
    generate("greedy_after_random", [0.0] * 4)
    generate("mixed_batch", [0.0, 0.8, 0.0, 0.8])
    report["records"] = records
    report["greedy_batch_matches"] = [a == b for a, b in zip(records["greedy_serial"], records["greedy_batch"], strict=True)]
    report["greedy_post_batch_matches"] = [a == b for a, b in zip(records["greedy_serial"], records["greedy_after_batch"], strict=True)]
    report["workers"] = llm.collective_rpc("batch_probe_report")
    report["passed"] = all(report["greedy_batch_matches"]) and all(report["greedy_post_batch_matches"]) and not any(
        worker["missing_random_probabilities"] or worker["stale_greedy_probability_steps"] for worker in report["workers"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
