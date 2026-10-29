#!/usr/bin/env python3
"""Exercise real Orthrus scheduling with controlled proposals, never fake logits.

The worker extension exists only in this test engine. The real head still runs;
greedy candidates are replaced afterwards. Singleton allowed-token constraints
make acceptance lengths predictable independently of model numerical variation.
This tests integration/state transitions, not head quality or unconstrained logits.
"""

import argparse
from collections import Counter
import functools
import hashlib
import json
import os
from pathlib import Path


def expected_greedy_tokens(candidates, token):
    accepted = next((i for i, value in enumerate(candidates) if value != token), len(candidates))
    return [token] * (accepted + 1)


def make_candidates(token, reject_at, width=31):
    if not 0 <= reject_at <= width:
        raise ValueError("rejection position must be within the candidate block")
    result = [token] * width
    if reject_at < width:
        result[reject_at] = token + 1
    return result


def check_scheduled_candidates(scheduled, proposed):
    if scheduled != proposed[:len(scheduled)]:
        raise AssertionError("scheduled candidates do not match this request's proposal")


def logit_errors(reference, actual):
    import torch

    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("logit diagnostics require matching nonempty tensors")
    delta = (reference.float() - actual.float()).abs()
    changed = reference.argmax(-1) != actual.argmax(-1)
    return {"max_abs": delta.max().item(), "first_row_max_abs": delta[0].max().item(),
            "changed_argmax_rows": int(changed.sum()), "rows": reference.shape[0],
            "first_row_argmax_changed": bool(changed[0]),
            "reference_first_token": int(reference[0].argmax()),
            "actual_first_token": int(actual[0].argmax())}


def compare_shadow_logits(events, names=("shadow-real", "shadow-no-head")):
    phases = {}
    for phase in names:
        phases[phase] = [{key: event[key] for key in
                          ("computed", "committed_before", "candidates", "raw_logits_sha256", "sampled")}
                         for event in events if event["phase"] == phase and event["kind"] == "verify"]
    left, right = phases.values()
    if not left or len(left) != len(right):
        raise AssertionError("shadow comparison requires two complete identical schedules")
    differences = []
    for step, (a, b) in enumerate(zip(left, right, strict=True)):
        if any(a[key] != b[key] for key in a if key != "raw_logits_sha256"):
            raise AssertionError(f"shadow state/schedule differs at step {step}")
        if a["raw_logits_sha256"] != b["raw_logits_sha256"]:
            differences.append(step)
    return {"steps": len(left), "all_target_logits_bitwise_equal": not differences,
            "different_steps": differences}


def assess_shadow_results(workers):
    if not workers:
        raise ValueError("shadow assessment requires worker results")
    # The first token-only run is fixed as the reference before any comparison.
    # Baseline non-repeatability never excuses a mismatch against that reference.
    adaptation = all(worker[key]["all_target_logits_bitwise_equal"] for worker in workers
                     for key in ("shadow", "token_only_vs_real"))
    repeatable = all(worker["token_only_repeat"]["all_target_logits_bitwise_equal"] for worker in workers)
    return {"adaptation_reference_passed": adaptation, "reference_repeat_exact": repeatable,
            "all_shadow_logits_exact": adaptation and repeatable}


class EngineStateWorker:
    """Explicit RPC installation; never imported by normal deployment scripts."""

    def state_probe_install(self, directory, compare_logits=False):
        runner = self.model_runner
        if runner.use_async_scheduling:
            raise ValueError("this probe currently requires synchronous scheduling")
        self._state_probe = dict(directory=directory, phase="initial", controlled=False,
                                 events=[], pending={}, rounds={}, plans={}, skip_head=False,
                                 compare_logits=compare_logits, logit_references={})
        self._state_wrapped = []
        self._state_wrap(runner, "_sample", self._probe_sample)
        if runner.drafter is not None:
            self._state_wrap(runner.drafter, "_propose", self._probe_propose)
            self._state_wrap(runner.drafter.model.model, "forward", self._probe_head)
        return {"rank": self.rank, "drafter": type(runner.drafter).__name__}

    def _state_wrap(self, obj, name, probe):
        original = getattr(obj, name)
        self._state_wrapped.append((obj, name, original))

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return probe(original, *args, **kwargs)

        setattr(obj, name, wrapped)

    def state_probe_phase(self, name, plans=None, skip_head=False):
        if skip_head and plans is None:
            raise ValueError("head bypass is allowed only with controlled test proposals")
        state = self._state_probe
        state.update(phase=name, controlled=plans is not None, plans=plans or {},
                     pending={}, rounds={}, skip_head=skip_head)

    def _probe_head(self, original, input_ids, positions):
        if not self._state_probe["skip_head"]:
            return original(input_ids=input_ids, positions=positions)
        import torch

        model = self.model_runner.drafter.model
        return torch.zeros(input_ids.numel(), model.config.hidden_size, device=input_ids.device,
                           dtype=self.model_runner.model.model.norm.weight.dtype)

    def _record_state(self, event):
        state = self._state_probe
        event.update(phase=state["phase"], rank=self.rank, controlled=state["controlled"])
        state["events"].append(event)
        path = Path(state["directory"]) / f"events-rank{self.rank}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def _probe_sample(self, original, logits, metadata):
        runner, state = self.model_runner, self._state_probe
        ids = list(runner.input_batch.req_ids)
        counts = metadata.num_draft_tokens if metadata is not None else [0] * len(ids)
        flat = metadata.draft_token_ids.cpu().tolist() if metadata is not None else []
        # Record raw target top choices before allowed-token processing. They
        # remain diagnostics: a constrained output alone cannot prove KV values.
        top_values, top_ids = logits.detach().float().topk(2, dim=-1)
        top_values, top_ids = top_values.cpu().tolist(), top_ids.cpu().tolist()
        raw_logits = logits.detach().cpu().contiguous() if state["phase"].startswith("shadow-") else None
        result = original(logits, metadata)
        sampled = result.sampled_token_ids.cpu().tolist()
        discarded = set(map(int, runner.discard_request_indices.np[:runner.num_discarded_requests]))
        offset = logit_offset = 0
        for row, (req_id, count) in enumerate(zip(ids, counts, strict=True)):
            candidates = flat[offset:offset + count]
            offset += count
            actual = [int(value) for value in sampled[row] if value >= 0]
            req = runner.requests[req_id]
            event = dict(kind="verify", request_id=req_id, batch=ids, row=row,
                         computed=int(runner.input_batch.num_computed_tokens_cpu[row]),
                         committed_before=int(runner.input_batch.num_tokens_no_spec[row]),
                         prompt_length=req.num_prompt_tokens, candidates=candidates,
                         sampled=actual, discarded=row in discarded,
                         raw_top_ids=top_ids[logit_offset:logit_offset + count + 1],
                         raw_top_logits=top_values[logit_offset:logit_offset + count + 1])
            if raw_logits is not None:
                import torch

                values = raw_logits[logit_offset:logit_offset + count + 1]
                raw = values.view(torch.uint8).numpy().tobytes()
                event["raw_logits_sha256"] = hashlib.sha256(raw).hexdigest()
                if state.get("compare_logits"):
                    key = (event["prompt_length"], event["computed"], tuple(candidates), event["discarded"])
                    references = state["logit_references"]
                    if state["phase"] == "shadow-token-only-1":
                        references[("target", key)] = values.clone()
                    else:
                        event["reference_logit_errors"] = logit_errors(references[("target", key)], values)
                    if state["phase"] == "shadow-real":
                        references[("head", key)] = values.clone()
                    elif state["phase"] == "shadow-no-head":
                        event["head_logit_errors"] = logit_errors(references[("head", key)], values)
            logit_offset += count + 1
            self._record_state(event)
            if row in discarded:
                continue
            if count:
                check_scheduled_candidates(candidates, state["pending"][req_id])
            if state["controlled"]:
                allowed = req.sampling_params.allowed_token_ids
                if len(allowed) != 1 or req.sampling_params.temperature != 0:
                    raise AssertionError("controlled probe requires greedy singleton allowed tokens")
                expected = expected_greedy_tokens(candidates, allowed[0])
                if actual != expected:
                    raise AssertionError(f"{req_id}: expected {expected}, received {actual}")
        return result

    def _probe_propose(self, original, **kwargs):
        import torch

        runner, state = self.model_runner, self._state_probe
        ids = list(runner.input_batch.req_ids)
        next_ids = kwargs["next_token_ids"].cpu().tolist()
        # Capture pre-draft metadata because the proposer expands it in place.
        original_lengths = kwargs["common_attn_metadata"].seq_lens.cpu().tolist()
        rejected = kwargs.get("num_rejected_tokens_gpu")
        rejected = rejected.cpu().tolist() if rejected is not None else [0] * len(ids)
        discarded = set(map(int, runner.discard_request_indices.np[:runner.num_discarded_requests]))
        if state["phase"].startswith("shadow-token-only"):
            # No head, block expansion, metadata builder, or cache read. Only
            # the native runner's sampling/scheduling and supplied tokens remain.
            proposed = []
            for row, req_id in enumerate(ids):
                token = runner.requests[req_id].sampling_params.allowed_token_ids[0]
                plan = state["plans"][str(token)]
                index = state["rounds"].get(req_id, 0)
                proposed.append(make_candidates(token, plan[index % len(plan)]))
                if row not in discarded:
                    state["rounds"][req_id] = index + 1
                state["pending"][req_id] = proposed[-1]
            return torch.tensor(proposed, dtype=torch.int64, device=kwargs["next_token_ids"].device)
        result = original(**kwargs)
        if not isinstance(result, torch.Tensor) or result.shape != (len(ids), 31):
            raise AssertionError(f"unexpected proposal shape: {getattr(result, 'shape', None)}")
        positions = runner.drafter.positions[:len(ids) * 32].view(len(ids), 32).cpu().tolist()
        inputs = runner.drafter.input_ids[:len(ids) * 32].view(len(ids), 32).cpu().tolist()
        proposed = result.cpu().tolist()
        for row, req_id in enumerate(ids):
            committed = int(runner.input_batch.num_tokens_no_spec[row])
            history = int(original_lengths[row] - rejected[row])
            event = dict(kind="draft", request_id=req_id, batch=ids, row=row,
                         committed=committed, anchor=positions[row][0], history=history,
                         rejected=int(rejected[row]), input_anchor=int(inputs[row][0]),
                         next_token=int(next_ids[row]), discarded=row in discarded,
                         blocks=[list(group) for group in runner.requests[req_id].block_ids])
            self._record_state(event)
            if positions[row] != list(range(history, history + 32)):
                raise AssertionError(f"{req_id}: draft positions disagree with effective history")
            if row not in discarded:
                if history != committed - 1 or inputs[row][0] != next_ids[row]:
                    raise AssertionError(f"{req_id}: anchor/committed-token mismatch: {event}")
                if any(value != runner.drafter.parallel_drafting_token_id for value in inputs[row][1:]):
                    raise AssertionError(f"{req_id}: draft block is not anchor plus MASK")
            # Compare each cache group's copied table to the actual runner row;
            # a request's position in a continuous batch is not its identity.
            for group, table in runner.drafter._per_group_block_table_buffers.items():
                expected_table = runner.input_batch.block_table[group].get_device_tensor(len(ids))
                allocated = len(runner.requests[req_id].block_ids[group])
                actual_table = table[row, :allocated]
                if not torch.equal(actual_table, expected_table[row, :actual_table.numel()]):
                    raise AssertionError(f"{req_id}: cache group {group} points to another row")
            if state["controlled"]:
                token = runner.requests[req_id].sampling_params.allowed_token_ids[0]
                plan = state["plans"][str(token)]
                round_index = state["rounds"].get(req_id, 0)
                proposed[row] = make_candidates(token, plan[round_index % len(plan)])
                if row not in discarded:
                    state["rounds"][req_id] = round_index + 1
            state["pending"][req_id] = proposed[row]
        if state["controlled"]:
            result.copy_(torch.tensor(proposed, dtype=result.dtype, device=result.device))
        return result

    def state_probe_report(self):
        events = self._state_probe["events"]
        verifies = [event for event in events if event["kind"] == "verify"
                    and not event["discarded"] and event["controlled"]]
        drafts = [event for event in events if event["kind"] == "draft"]
        signature = [{key: value for key, value in event.items()
                      if key not in {"rank", "raw_top_logits", "raw_top_ids", "raw_logits_sha256",
                                     "reference_logit_errors", "head_logit_errors"}}
                     for event in events]
        return dict(rank=self.rank, verify_steps=len(verifies), draft_steps=len(drafts),
                    observed_accepted=dict(Counter(len(event["sampled"]) - 1 for event in verifies
                                                   if event["candidates"])),
                    batches=sorted({len(event["batch"]) for event in drafts}),
                    state_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                    shadow=compare_shadow_logits(events),
                    token_only_repeat=compare_shadow_logits(events, ("shadow-token-only-1", "shadow-token-only-2")),
                    token_only_vs_real=compare_shadow_logits(events, ("shadow-token-only-1", "shadow-real")),
                    logit_diagnostics=[{key: event[key] for key in
                                       ("phase", "prompt_length", "computed", "discarded",
                                        "reference_logit_errors", "head_logit_errors") if key in event}
                                       for event in events if any(event.get(key, {}).get("max_abs", 0) > 0
                                           for key in ("reference_logit_errors", "head_logit_errors"))],
                    kernel_block_sizes=self.model_runner.drafter._per_group_kernel_block_sizes,
                    history_beyond_page=any(event["history"] >= 128 for event in drafts),
                    history_beyond_c128_group=any(event["history"] >= 256 for event in drafts))


def generate_shadow_batch(llm, prompts, params):
    # Multiprocess engines may schedule while enqueue() is still sending requests.
    # Level zero pauses scheduling only; weights and KV stay resident.
    llm.sleep(level=0, mode="keep")
    try:
        llm.enqueue(prompts, params, use_tqdm=False)
    finally:
        llm.wake_up(tags=["scheduling"])
    return llm.wait_for_completion(use_tqdm=False)


def run_checks(llm, args):
    from vllm import SamplingParams

    report = dict(scope="engine state checks, not full-model/head-quality acceptance", state_checks_passed=False,
                  model=args.model, head=args.head, tp=args.tp, kv_cache_mib=args.kv_cache_mib,
                  cache_guard=os.environ.get("ORTHRUS_VERIFY_CACHE_CHECKSUM"),
                  target_layout="canonical_c4_diagnostic" if args.trace_canonical_c4 else "native")
    llm.collective_rpc("state_probe_install", args=(str(args.output), args.shadow_logit_errors))
    if args.trace_block_start is not None:
        llm.collective_rpc("batch_trace_configure", args=(str(args.output / "layer-trace"),
                                                         args.trace_block_start, args.trace_prompt_length,
                                                         args.trace_repeat_attention, args.trace_c4_inputs,
                                                         args.trace_canonical_c4))
    plans = {"19": list(range(32)), "23": [31], "29": [0, 0, 31, 7, 0, 31], "31": [15, 30, 1, 31]}
    lengths, budgets = [127, 128, 511, 513], [600, 193, 160, 65]
    prompts = [{"prompt_token_ids": [100 + i] * length} for i, length in enumerate(lengths)]
    params = [SamplingParams(temperature=0, max_tokens=budget, ignore_eos=True, allowed_token_ids=[int(token)])
              for token, budget in zip(plans, budgets, strict=True)]
    for phase, skip in (("shadow-token-only-1", True), ("shadow-token-only-2", True),
                        ("shadow-real", False), ("shadow-no-head", True)):
        llm.collective_rpc("state_probe_phase", args=(phase, plans, skip))
        count = args.shadow_batch
        outputs = generate_shadow_batch(llm, prompts[:count], params[:count])
        for output, param in zip(outputs, params[:count], strict=True):
            if list(output.outputs[0].token_ids) != [param.allowed_token_ids[0]] * param.max_tokens:
                raise AssertionError(f"{phase}: incorrect controlled output")
        print(json.dumps({"phase": phase, "completed": True, "requests": count}), flush=True)
    for phase, order in (("controlled", [0, 1, 2, 3]), ("controlled-reversed", [3, 2, 1, 0])):
        llm.collective_rpc("state_probe_phase", args=(phase, plans))
        outputs = llm.generate([prompts[i] for i in order], [params[i] for i in order], use_tqdm=False)
        for i, output in zip(order, outputs, strict=True):
            if list(output.outputs[0].token_ids) != [params[i].allowed_token_ids[0]] * budgets[i]:
                raise AssertionError(f"{phase}: output leaked rejected tokens or ended early")
        report[phase] = {"passed": True, "requests": len(outputs)}
        print(json.dumps({"phase": phase, "passed": True}), flush=True)

    llm.collective_rpc("state_probe_phase", args=("cancel-and-next-request", plans))
    engine = llm.llm_engine
    engine.add_request("state-cancel", prompts[0], params[0])
    for _ in range(3):
        engine.step()
    engine.abort_request(["state-cancel"])
    while engine.has_unfinished_requests():
        engine.step()
    output = llm.generate([prompts[1]], [params[1]], use_tqdm=False)[0]
    if list(output.outputs[0].token_ids) != [23] * budgets[1]:
        raise AssertionError("post-cancellation request was contaminated")
    report["cancel-and-next-request"] = {"passed": True}

    if args.natural_baseline:
        cases = json.loads(args.natural_baseline.read_text())["greedy"][::3]
        natural_prompts = [row["case"]["prompt"] * int(row["case"].get("prompt_repeat", 1)) for row in cases]
        natural_params = [SamplingParams(temperature=0, max_tokens=row["case"]["max_tokens"], ignore_eos=True)
                          for row in cases]
        natural = {}
        for phase in ("natural-serial", "natural-batch"):
            llm.collective_rpc("state_probe_phase", args=(phase,))
            outputs = (sum([llm.generate([p], [s], use_tqdm=False)
                            for p, s in zip(natural_prompts, natural_params, strict=True)], [])
                       if phase == "natural-serial" else llm.generate(natural_prompts, natural_params, use_tqdm=False))
            natural[phase] = [{"request_id": out.request_id, "prompt_token_ids": out.prompt_token_ids,
                               "token_ids": list(out.outputs[0].token_ids)} for out in outputs]
        report["natural"] = natural
        report["natural_exact"] = [a["token_ids"] == b["token_ids"] for a, b in
                                   zip(natural["natural-serial"], natural["natural-batch"], strict=True)]
    if args.trace_block_start is not None:
        report["layer_traces"] = llm.collective_rpc("batch_trace_report")
    report["workers"] = llm.collective_rpc("state_probe_report")
    for worker in report["workers"]:
        if set(map(int, worker["observed_accepted"])) != set(range(32)):
            raise AssertionError(f"missing acceptance positions: {worker['observed_accepted']}")
    if len({worker["state_sha256"] for worker in report["workers"]}) != 1:
        raise AssertionError("TP ranks disagree on request/state transitions")
    report["state_checks_passed"] = True
    report.update(assess_shadow_results(report["workers"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-cache-mib", type=int, default=1024,
                        help="bounded per-rank KV pool; full debug snapshots need additional memory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-baseline", type=Path)
    parser.add_argument("--shadow-batch", type=int, choices=(1, 4), default=1,
                        help="requests in fixed-candidate target-logit comparisons")
    parser.add_argument("--shadow-logit-errors", action="store_true",
                        help="retain CPU reference logits to quantify differences; no threshold changes")
    parser.add_argument("--trace-block-start", type=int)
    parser.add_argument("--trace-prompt-length", type=int)
    parser.add_argument("--trace-repeat-attention", action="store_true",
                        help="diagnostic only: replay selected native Attention with identical live inputs")
    parser.add_argument("--trace-c4-inputs", action="store_true",
                        help="save complete selected C4 call inputs, including full physical KV pools")
    parser.add_argument("--trace-canonical-c4", action="store_true",
                        help="diagnostic only: canonicalize target compressed page layout in shadow phases")
    args = parser.parse_args()
    if args.kv_cache_mib <= 0:
        parser.error("kv-cache-mib must be positive")
    if (args.trace_block_start is None) != (args.trace_prompt_length is None):
        parser.error("layer tracing requires both block start and prompt length")
    if args.trace_repeat_attention and args.trace_block_start is None:
        parser.error("Attention replay requires layer tracing")
    if args.trace_c4_inputs and args.trace_block_start is None:
        parser.error("C4 snapshots require layer tracing")
    if args.trace_canonical_c4 and args.trace_block_start is None:
        parser.error("canonical C4 diagnostics require layer tracing")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Full cache snapshots once per layer/batch size, sampled guards afterwards.
    os.environ["ORTHRUS_VERIFY_CACHE_CHECKSUM"] = "full"
    from vllm import LLM

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enable_expert_parallel=True,
              quantization="ascend", tokenizer_mode="deepseek_v4", enforce_eager=True,
              async_scheduling=False, compilation_config={"cudagraph_mode": "NONE"},
              additional_config={"multistream_dsv4_dsa_overlap": False,
                                 "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": False},
                                 "enable_cpu_binding": True, "multistream_overlap_shared_expert": True},
              max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=1024,
              gpu_memory_utilization=0.5, kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
              enable_prefix_caching=False, block_size=128,
              model_loader_extra_config={"enable_multithread_load": True, "num_threads": 128},
              speculative_config={"method": "orthrus", "model": args.head, "num_speculative_tokens": 31,
                                  "parallel_drafting": True, "draft_sample_method": "greedy",
                                  "rejection_sample_method": "standard", "quantization": "ascend", "enforce_eager": True},
              worker_extension_cls=("trace_dsv4_batch_layers.BatchLayerWorker" if args.trace_block_start is not None
                                    else "check_dsv4_engine_state.EngineStateWorker"))
    try:
        report = run_checks(llm, args)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"passed": False, "error": repr(error)}, indent=2) + "\n")
        raise
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "natural"}), flush=True)
    return 0 if report["state_checks_passed"] and report["adaptation_reference_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
