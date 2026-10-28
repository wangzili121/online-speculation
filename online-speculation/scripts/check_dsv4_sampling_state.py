#!/usr/bin/env python3
"""Real-engine probabilistic/state tests with explicit synthetic p/q fixtures.

Target and draft forwards still execute. Test-only hooks replace their logits,
not the sampler, scheduler, cache operations or probability transport. This
checks integration, not natural model accuracy, acceptance or performance.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

from check_dsv4_engine_state import EngineStateWorker, check_scheduled_candidates


def fixture(token, slot=None):
    """First-token probabilities; distinct pairs reveal request/slot swaps."""
    target, draft = {200: (0.5, 0.5), 210: (0.35, 0.82),
                     220: (0.5, 0.58), 230: (0.7, 0.25)}[token]
    return target if slot is None else draft + (0 if token == 200 else 0.001 * slot)


def check_rejection(candidates, sampled, token):
    if not sampled or len(sampled) > len(candidates) + 1:
        raise AssertionError("invalid sampled length")
    accepted = len(sampled) - 1
    if sampled[:accepted] != candidates[:accepted]:
        raise AssertionError("accepted prefix differs from proposed tokens")
    if any(value not in (token, token + 1) for value in sampled):
        raise AssertionError("sampled token is outside target support")
    if accepted < len(candidates):
        p, q = fixture(token), fixture(token, accepted)
        if p == q:
            raise AssertionError("equal distributions must accept every candidate")
        expected = token if p > q else token + 1
        if sampled[-1] != expected:
            raise AssertionError("recovery token is outside positive (p-q) support")
    return accepted


def frequency_check(values, token, alpha=1e-5):
    if len(values) < 512 or any(value not in (token, token + 1) for value in values):
        raise AssertionError("insufficient samples or output outside target support")
    observed = values.count(token) / len(values)
    tolerance = math.sqrt(math.log(2 / alpha) / (2 * len(values)))
    if abs(observed - fixture(token)) > tolerance:
        raise AssertionError(f"target distribution mismatch: {observed}, expected {fixture(token)}")
    return dict(samples=len(values), observed=observed, expected=fixture(token),
                tolerance=tolerance, alpha=alpha)


class SamplingStateWorker(EngineStateWorker):
    def sampling_probe_install(self, directory):
        result = self.state_probe_install(directory)
        self._prob_pending = {}
        self._state_wrap(self.model_runner.drafter, "_sample_from_logits", self._probe_draft_sample)
        self._state_wrap(self.model_runner, "_get_spec_decode_draft_probs", self._probe_probabilities)
        self._state_wrap(self.model_runner, "execute_model", self._probe_execute)
        return result

    def _probe_execute(self, original, scheduler_output, *args, **kwargs):
        cached = scheduler_output.scheduled_cached_reqs
        allocated = {req.req_id: req.block_ids for req in scheduler_output.scheduled_new_reqs}
        allocated.update({req: blocks for req, blocks in zip(cached.req_ids, cached.new_block_ids, strict=True)
                          if blocks is not None})
        self._record_state(dict(kind="schedule", resumed=sorted(cached.resumed_req_ids),
                                preempted=sorted(scheduler_output.preempted_req_ids or []),
                                finished=sorted(scheduler_output.finished_req_ids), allocated=allocated))
        return original(scheduler_output, *args, **kwargs)

    def _token(self, req_id):
        return self.model_runner.requests[req_id].sampling_params.allowed_token_ids[0]

    def _draft_probability(self, token, slot):
        return fixture(token) if self._state_probe["phase"].startswith("pressure-") else fixture(token, slot)

    def _probe_draft_sample(self, original, logits, sampling_metadata):
        import torch

        runner = self.model_runner
        ids = list(runner.input_batch.req_ids)
        if logits.shape[0] != len(ids) * 31:
            raise AssertionError("draft logits are not request-major K31")
        controlled = torch.full_like(logits, -float("inf"), dtype=torch.float32)
        for row, req_id in enumerate(ids):
            token = self._token(req_id)
            q = torch.tensor([self._draft_probability(token, i) for i in range(31)], device=logits.device)
            temperature = runner.requests[req_id].sampling_params.temperature
            controlled[row * 31:(row + 1) * 31, token] = q.log() * temperature
            controlled[row * 31:(row + 1) * 31, token + 1] = (1 - q).log() * temperature
        result, probs = original(controlled, sampling_metadata)
        if probs is None or probs.shape != logits.shape:
            raise AssertionError("probabilistic draft did not return full-vocabulary q")
        return result, probs

    def _probe_propose(self, original, **kwargs):
        result = super()._probe_propose(original, **kwargs)
        runner = self.model_runner
        probs = runner.drafter._last_draft_probs
        ids = list(runner.input_batch.req_ids)
        if probs is None or probs.shape[:2] != (len(ids), 31):
            raise AssertionError("draft probabilities were not preserved")
        for row, req_id in enumerate(ids):
            token = self._token(req_id)
            pair = probs[row, :, token:token + 2].cpu().tolist()
            if int(probs[row].count_nonzero()) != 62:
                raise AssertionError("q has mass outside its two-token support")
            for slot, (a, b) in enumerate(pair):
                if abs(a - self._draft_probability(token, slot)) > 1e-6 or abs(a + b - 1) > 1e-6:
                    raise AssertionError("draft q differs from its request/position fixture")
            self._prob_pending[req_id] = pair
        return result

    def _probe_probabilities(self, original, metadata):
        result = original(metadata)
        counts = metadata.num_draft_tokens
        if sum(counts) == 0:
            if result is not None:
                raise AssertionError("unexpected q for a non-speculative step")
            return result
        if result is None or result.shape[0] != sum(counts):
            raise AssertionError("missing or truncated verifier draft probabilities")
        offset = 0
        for req_id, count in zip(self.model_runner.input_batch.req_ids, counts, strict=True):
            if count:
                token = self._token(req_id)
                pair = result[offset:offset + count, token:token + 2].cpu().tolist()
                if pair != self._prob_pending[req_id][:count]:
                    raise AssertionError("verifier q belongs to another request/slot")
                if int(result[offset:offset + count].count_nonzero()) != 2 * count:
                    raise AssertionError("verifier q has unexpected support")
            offset += count
        self._record_state(dict(kind="probabilities", batch=list(self.model_runner.input_batch.req_ids),
                                counts=list(counts), checked=True))
        return result

    def _probe_sample(self, original, logits, metadata):
        import torch

        runner, state = self.model_runner, self._state_probe
        ids = list(runner.input_batch.req_ids)
        counts = metadata.num_draft_tokens if metadata is not None else [0] * len(ids)
        flat = metadata.draft_token_ids.cpu().tolist() if metadata is not None else []
        controlled = torch.full_like(logits, -float("inf"), dtype=torch.float32)
        offset = 0
        for req_id, count in zip(ids, counts, strict=True):
            token, p = self._token(req_id), fixture(self._token(req_id))
            temperature = runner.requests[req_id].sampling_params.temperature
            controlled[offset:offset + count + 1, token] = math.log(p) * temperature
            controlled[offset:offset + count + 1, token + 1] = math.log(1 - p) * temperature
            offset += count + 1
        if offset != logits.shape[0]:
            raise AssertionError("target logits do not match scheduled query lengths")
        result = original(controlled, metadata)
        sampled = result.sampled_token_ids.cpu().tolist()
        discarded = set(map(int, runner.discard_request_indices.np[:runner.num_discarded_requests]))
        offset = 0
        for row, (req_id, count) in enumerate(zip(ids, counts, strict=True)):
            candidates = flat[offset:offset + count]
            offset += count
            actual = [int(value) for value in sampled[row] if value >= 0]
            accepted = None
            if row not in discarded:
                if count:
                    check_scheduled_candidates(candidates, state["pending"][req_id])
                accepted = check_rejection(candidates, actual, self._token(req_id))
                if state["phase"].startswith("pressure-") and accepted != count:
                    raise AssertionError("equal-distribution pressure fixture rejected a candidate")
            self._record_state(dict(kind="sample", request_id=req_id, token=self._token(req_id),
                                    batch=ids, candidates=candidates, sampled=actual, accepted=accepted,
                                    discarded=row in discarded))
        return result

    def sampling_probe_report(self):
        events = self._state_probe["events"]
        samples = [e for e in events if e["kind"] == "sample" and not e["discarded"]]
        drafts = [e for e in events if e["kind"] == "draft"]
        resumed = {req for e in events if e["kind"] == "schedule" for req in e["resumed"]}
        after_resume = set()
        seen = set()
        owners, reused = {}, set()
        for event in events:
            if event["kind"] == "schedule":
                seen.update(event["resumed"])
                # These are allocator-issued pages, not stale entries remaining
                # in worker tables after a request is shortened or preempted.
                for req_id, groups in event["allocated"].items():
                    for group, blocks in enumerate(groups):
                        for block in blocks:
                            if block == 0:
                                continue
                            key = (group, block)
                            if key in owners and owners[key] != req_id:
                                reused.add(key)
                            owners[key] = req_id
            elif event["kind"] == "sample" and not event["discarded"] and event["candidates"]:
                if event["request_id"] in seen:
                    after_resume.add(event["request_id"])
        signature = [{k: v for k, v in e.items() if k != "rank"} for e in events]
        return dict(rank=self.rank, sample_events=len(samples), draft_events=len(drafts),
                    probability_checks=sum(e["kind"] == "probabilities" for e in events),
                    accepted=dict(Counter(e["accepted"] for e in samples if e["candidates"])),
                    full_accepts=sum(len(e["candidates"]) == 31 and e["accepted"] == 31 for e in samples),
                    rejections=sum(e["accepted"] < len(e["candidates"]) for e in samples),
                    batches=sorted({len(e["batch"]) for e in drafts}),
                    reused_pages=len(reused), reused_groups=sorted({g for g, _ in reused}),
                    resumed_requests=sorted(resumed), verified_after_resume=sorted(after_resume),
                    state_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest())


def run_checks(llm, args):
    from vllm import SamplingParams

    llm.collective_rpc("sampling_probe_install", args=(str(args.output),))
    frequencies, outputs_by_request = {}, {}
    tokens = [200, 210, 220, 230]
    phases = [f"random-wave-{i}" for i in range(args.waves)] + [f"pressure-{i}" for i in range(2)]
    for wave, phase in enumerate(phases):
        llm.collective_rpc("state_probe_phase", args=(phase,))
        order = tokens if wave % 2 == 0 else list(reversed(tokens))
        lengths = [31, 63, 95, 127] if phase.startswith("pressure-") else [127, 128, 511, 513]
        budget = 1536 if phase.startswith("pressure-") else args.tokens
        prompts = [{"prompt_token_ids": [token + 10] * length}
                   for token, length in zip(order, lengths, strict=True)]
        params = [SamplingParams(temperature=temperature, seed=900 + wave * 4 + i,
                                 allowed_token_ids=[token, token + 1], max_tokens=budget,
                                 ignore_eos=True)
                  for i, (token, temperature) in enumerate(zip(order, [0.7, 1., 1.3, 0.9], strict=True))]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for token, output in zip(order, outputs, strict=True):
            values = list(output.outputs[0].token_ids)
            if len(values) != budget:
                raise AssertionError("request ended early")
            outputs_by_request[output.request_id] = values
            mode = "equal-pressure" if phase.startswith("pressure-") else "random"
            frequencies.setdefault(mode, {}).setdefault(token, []).extend(values)
        print(json.dumps(dict(phase=phase, completed=True)), flush=True)
    workers = llm.collective_rpc("sampling_probe_report")
    errors = []
    for worker in workers:
        if not worker["probability_checks"] or not worker["full_accepts"] or not worker["rejections"]:
            errors.append("missing probabilistic acceptance/rejection coverage")
        if not worker["reused_pages"]:
            errors.append("no physical cache page reused by a different request")
        if args.require_preemption and not worker["verified_after_resume"]:
            errors.append("no preempted request resumed speculative verification")
    if len({w["state_sha256"] for w in workers}) != 1:
        errors.append("TP ranks disagree on sampling/state")
    injection_path = args.output / "allocation-events.jsonl"
    allocation_events = [json.loads(line) for line in injection_path.read_text().splitlines()] if injection_path.exists() else []
    if args.inject_allocation_failure and not any(e["kind"] == "injected_allocation_failure" for e in allocation_events):
        errors.append("allocation failure injection did not execute")
    sources = [Path(__file__), Path(__file__).with_name("check_dsv4_engine_state.py"),
               Path(__file__).with_name("dsv4_test_scheduler.py")]
    return dict(passed=not errors, errors=errors, scope="synthetic-distribution integration, not model accuracy or speed",
                model=args.model, head=args.head, tp=args.tp, num_gpu_blocks_override=args.num_blocks,
                require_preemption=args.require_preemption, workers=workers,
                preemption_trigger="injected allocate_slots failure" if args.inject_allocation_failure else "natural capacity pressure",
                allocation_events=allocation_events,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                frequencies={mode: {token: frequency_check(values, token) for token, values in rows.items()}
                             for mode, rows in frequencies.items()},
                outputs=outputs_by_request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--require-preemption", action="store_true")
    parser.add_argument("--inject-allocation-failure", action="store_true",
                        help="test-only one-shot failures; native scheduler still performs preemption/recovery")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.waves < 2 or args.tokens < 256 or args.tokens + 513 > 2048 or args.num_blocks < 1:
        parser.error("need >=2 waves, >=256 tokens, <=2048 total context and positive blocks")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["ORTHRUS_VERIFY_CACHE_CHECKSUM"] = "full"
    options = {}
    if args.inject_allocation_failure:
        os.environ["ORTHRUS_TEST_ALLOCATION_LOG"] = str(args.output / "allocation-events.jsonl")
        options["scheduler_cls"] = "dsv4_test_scheduler.AllocationFailureScheduler"
    from vllm import LLM

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enable_expert_parallel=True,
              quantization="ascend", tokenizer_mode="deepseek_v4", enforce_eager=True,
              async_scheduling=False, compilation_config={"cudagraph_mode": "NONE"},
              additional_config={"multistream_dsv4_dsa_overlap": False,
                                 "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": False},
                                 "enable_cpu_binding": True, "multistream_overlap_shared_expert": True},
              max_model_len=2048, max_num_seqs=4, max_num_batched_tokens=1024,
              gpu_memory_utilization=0.5, kv_cache_memory_bytes=1024**3,
              num_gpu_blocks_override=args.num_blocks,
              enable_prefix_caching=False, block_size=128,
              model_loader_extra_config={"enable_multithread_load": True, "num_threads": 128},
              speculative_config={"method": "orthrus", "model": args.head, "num_speculative_tokens": 31,
                                  "parallel_drafting": True, "draft_sample_method": "probabilistic",
                                  "rejection_sample_method": "standard", "quantization": "ascend", "enforce_eager": True},
              worker_extension_cls="check_dsv4_sampling_state.SamplingStateWorker", **options)
    try:
        report = run_checks(llm, args)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"passed": False, "error": repr(error)}, indent=2) + "\n")
        raise
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
