#!/usr/bin/env python3
"""Run 1122 W8A8 AR and optionally compare a resident Orthrus head on both EP16 hosts."""

import argparse
from collections import Counter
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace


@dataclass
class OnlineProgress:
    completed_waves: int = 0
    next_capture_wave: int = 0
    updated_rounds: int = 0
    updates: list = field(default_factory=list)


class GreedyDivergence(ValueError):
    def __init__(self, request, position, phase, expected, observed):
        self.witness = dict(request=request, position=position, phase=phase, expected=expected, observed=observed)
        super().__init__(f"{request}: {phase} greedy token {position} expected {expected}, got {observed}")


def workload_batches(cases, slots_per_dp):
    """Pack consecutive requests into fixed waves within the per-DP slot budget."""
    batch, occupancy = [], Counter()
    for case in cases:
        dp = case["dp"]
        if occupancy[dp] == slots_per_dp:
            yield batch
            batch, occupancy = [], Counter()
        batch.append(case)
        occupancy[dp] += 1
    if batch:
        yield batch


def _generate_batch(session, cases, prefill_rows, dp_size, *, width, teacher_rows, reverse, optimizer, online, progress,
                    expected):
    """Pack prompt chunks by DP owner and continue from device-greedy IDs."""
    handles = [session.admit(case["id"], case["dp"], len(case["input_ids"])) for case in cases]
    output = {handle: [] for handle in handles}
    offsets = [0] * len(cases)
    requests = list(zip(cases, handles, strict=True))

    def append(handle, tokens, phase):
        if expected is not None:
            reference = expected[handle.request_id]
            for position, token in enumerate(tokens, len(output[handle])):
                wanted = reference[position] if position < len(reference) else None
                if wanted != token:
                    raise GreedyDivergence(handle.request_id, position, phase, wanted, token)
        output[handle].extend(tokens)

    start = perf_counter()
    while any(offset < len(case["input_ids"]) for offset, case in zip(offsets, cases, strict=True)):
        remaining = [prefill_rows] * dp_size
        batch = []
        for index, (case, handle) in enumerate(requests):
            offset, ids = offsets[index], case["input_ids"]
            take = min(remaining[handle.dp], len(ids) - offset)
            if take:
                batch.append((handle, ids[offset:offset + take]))
                offsets[index] += take
                remaining[handle.dp] -= take
        for handle, token in session.prefill_greedy_batch(batch).items():
            append(handle, (token,), "prefill")

    prefill_seconds = perf_counter() - start
    start = perf_counter()
    steps = 0
    speculation = Counter(draft_rounds=0, draft_requests=0, proposed=0, accepted=0)
    replay = None
    captured_at = 0
    while pending := [(handle, case["new_tokens"] - len(output[handle])) for case, handle in requests
                      if len(output[handle]) < case["new_tokens"]]:
        if width:
            # The fixed draft block serves one resident request per DP in a wave.
            owners = {}
            for handle, remaining in pending:
                owners.setdefault(handle.dp, (handle, min(width, remaining)))
            limits = dict(owners.values())
            capture = None
            capture_due = progress is None or (len(progress.updates) < online["max_updates"]
                                               and progress.completed_waves >= progress.next_capture_wave)
            if teacher_rows and capture_due and replay is None:
                order = requests
                if optimizer is not None:
                    # Alternate the preferred owner host when both have eligible queries.
                    preferred_dp = (len(progress.updates) % 2) * (dp_size // 2)
                    order = sorted(requests, key=lambda item: (item[1].dp - preferred_dp) % dp_size)
                for case, handle in order:
                    position = len(case["input_ids"]) + len(output[handle]) - 1
                    if limits.get(handle, 0) > 1 and position + session.draft.block <= session.context:
                        captured_at = len(output[handle])
                        capture = replay = (handle, min(teacher_rows, case["new_tokens"] - captured_at))
                        break
            wave = session.speculate_batch(
                [(handle, output[handle][-1]) for handle in limits],
                width=width, output_limits=limits, student_capture=capture,
            )
            draft_requests = sum(bool(proposal) for proposal in wave.proposals.values())
            speculation["draft_rounds"] += bool(draft_requests)
            speculation["draft_requests"] += draft_requests
            if optimizer is not None and optimizer.adam.version:
                progress.updated_rounds += bool(draft_requests)
            speculation["proposed"] += sum(map(len, wave.proposals.values()))
            for handle, emitted in wave.verified.emitted.items():
                append(handle, emitted, "verify/accept")
                speculation["accepted"] += len(emitted) - 1
            if optimizer is not None and replay is not None:
                handle, expected_rows = replay
                if len(output[handle]) - captured_at >= expected_rows:
                    outcome = optimizer.update(
                        f"online-update/{len(progress.updates) + 1}", owner=handle.dp * session.draft.tp,
                        feedback=session.feedback.get(handle), finished=False,
                        rate=online["rate"], max_norm=online["max_norm"],
                    )
                    if outcome is None or outcome["teacher_rows"] != expected_rows:
                        raise ValueError("Committed output and the online teacher block have different lengths")
                    receipt = dict(request=handle.request_id, dp=handle.dp,
                                   round=progress.completed_waves + 1, **outcome)
                    progress.updates.append(receipt)
                    print(json.dumps(dict(stage="online_update", **receipt)), flush=True)
                    session.release_student(handle)
                    replay = None
                    progress.next_capture_wave = progress.completed_waves + online["every"]
        else:
            batch = [(handle, [output[handle][-1]]) for handle, _ in pending]
            for handle, token in session.decode_greedy_batch(batch).items():
                output[handle].append(token)
        steps += 1
        if progress is not None:
            progress.completed_waves += 1
    decode_seconds = perf_counter() - start
    capture_receipt = None
    if replay is not None:
        handle, expected_rows = replay
        feedback = session.feedback.get(handle)
        capture_receipt = dict(request=handle.request_id, dp=handle.dp, expected_rows=expected_rows)
        if feedback is not None:
            capture_receipt.update(
                position=feedback.student.position, teacher_rows=expected_rows - feedback.remaining,
                attention_and_ffn_tapes=sorted(next(iter(feedback.student.ffn_tape.values()))),
            )
    backward = None
    if reverse is not None and replay is not None:
        backward = reverse.run("check-backward", owner=handle.dp * session.draft.tp, feedback=feedback, finished=True,
                               consume=lambda *_: None)
    for handle in handles:
        session.release(handle)
    result = {handle.request_id: output[handle] for handle in handles}
    tok = sum(map(len, result.values())) - len(cases)
    return dict(generated=result, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                decode_steps=steps, decode_tokens=tok, speculation=speculation, capture=capture_receipt, backward=backward)


def generate(session, cases, prefill_rows, dp_size, *, width=0, teacher_rows=0, reverse=None,
             optimizer=None, online=None, slots_per_dp=16, expected=None, mode=None):
    """Reuse the resident session across bounded batches and keep a continuous update schedule."""
    progress = OnlineProgress() if optimizer is not None else None
    report = dict(generated={}, prefill_seconds=0., decode_seconds=0., decode_steps=0,
                  decode_tokens=0, speculation=Counter(draft_rounds=0, draft_requests=0, proposed=0, accepted=0),
                  capture=None, backward=None, request_batches=0, max_live_requests=0)
    for batch in workload_batches(cases, slots_per_dp):
        try:
            batch_report = _generate_batch(
                session, batch, prefill_rows, dp_size, width=width, optimizer=optimizer, online=online, progress=progress,
                teacher_rows=teacher_rows if report["capture"] is None else 0,
                reverse=reverse if report["capture"] is None else None, expected=expected,
            )
        except GreedyDivergence as error:
            if mode is not None:
                error.witness["mode"] = mode
            raise
        report["request_batches"] += 1
        report["max_live_requests"] = max(report["max_live_requests"], len(batch))
        report["generated"].update(batch_report["generated"])
        report["speculation"].update(batch_report["speculation"])
        for key in ("prefill_seconds", "decode_seconds", "decode_steps", "decode_tokens"):
            report[key] += batch_report[key]
        if batch_report["capture"] is not None:
            report.update(capture=batch_report["capture"], backward=batch_report["backward"])
        if mode is not None:
            print(json.dumps(dict(stage="batch_complete", mode=mode, batch=report["request_batches"],
                                  requests_finished=len(report["generated"]), requests_total=len(cases),
                                  prefill_seconds=batch_report["prefill_seconds"],
                                  decode_seconds=batch_report["decode_seconds"])), flush=True)
    tokens, elapsed = report["decode_tokens"], report["decode_seconds"]
    report["decode_tok_per_s"] = tokens / elapsed if tokens else 0.
    counts = report["speculation"]
    report["acceptance"] = dict(candidate_fraction=counts["accepted"] / counts["proposed"] if counts["proposed"] else 0.,
                                mean_accepted_per_request=counts["accepted"] / counts["draft_requests"]
                                if counts["draft_requests"] else 0.)
    if progress is not None:
        updates = progress.updates
        update_seconds = sum(item["update_seconds"] for item in updates)
        inference_seconds = elapsed - update_seconds
        report["online"] = dict(updates=updates, version=optimizer.adam.version,
                                rounds_after_publication=progress.updated_rounds, update_seconds=update_seconds,
                                gradient_scratch_bytes_per_rank=2 * optimizer.packet_capacity * 4 if updates else 0,
                                decode_tok_per_s_excluding_updates=tokens / inference_seconds if tokens else 0.0)
    return report


def first_difference(expected, actual):
    """Identify the first differing or missing token across identical workloads."""
    from itertools import zip_longest

    for request, tokens in expected.items():
        for position, (reference, observed) in enumerate(zip_longest(tokens, actual[request])):
            if reference != observed:
                return dict(request=request, position=position, expected=reference, observed=observed)
    return None


def compare_generation(reference, actual):
    difference = first_difference(reference["generated"], actual["generated"])
    return dict(passed=difference is None and actual["speculation"]["draft_rounds"] > 0,
                compared_tokens=sum(map(len, reference["generated"].values())), first_difference=difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rank0-ip", required=True)
    parser.add_argument("--port", type=int, default=29642)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefill-graph", type=Path, required=True)
    parser.add_argument("--decode-graph", type=Path, required=True)
    parser.add_argument("--prefill-capacity", type=int, required=True)
    parser.add_argument("--prefill-moe-tokens", type=int, required=True)
    parser.add_argument("--decode-capacity", type=int, choices=(16, 80), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--slots-per-dp", type=int, default=1)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--golden", type=Path, help="JSON mapping from request ID to expected greedy IDs")
    parser.add_argument("--draft-head", type=Path, help="Orthrus head in the DSV4.1 exchange format")
    parser.add_argument("--draft-graph", type=Path, help="Compiled eight-rank draft bundle")
    parser.add_argument("--draft-block", type=int, choices=(16, 32), default=16)
    parser.add_argument("--verify-width", type=int, choices=range(2, 17), default=8)
    parser.add_argument("--capture-last-layers", type=int, choices=range(27), default=0,
                        metavar="N", help="Retain N suffix layers and verify one teacher replay")
    parser.add_argument("--teacher-rows", type=int, choices=range(1, 16), default=7)
    parser.add_argument("--check-backward", action="store_true", help="Run the captured suffix backward before release")
    parser.add_argument("--adam-graph", type=Path, help="Enable online Adam using the eight-rank optimizer bundle")
    parser.add_argument("--update-every", type=int, default=16, help="Minimum waves between online updates")
    parser.add_argument("--max-updates", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-gradient-norm", type=float, default=1.)
    parser.add_argument("--warmup-workload", type=Path, help="Warm AR and draft kernels before measured requests")
    parser.add_argument("--eval-workload", type=Path, help="Compare the initial and frozen heads on separate requests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.draft_head) != bool(args.draft_graph):
        parser.error("Provide both --draft-head and --draft-graph to compare Orthrus")
    if args.capture_last_layers and not args.draft_head:
        parser.error("Suffix replay requires a draft head and a matching compiled suffix")
    if args.check_backward and not args.capture_last_layers:
        parser.error("--check-backward requires --capture-last-layers")
    if args.adam_graph and (not args.capture_last_layers or args.check_backward):
        parser.error("--adam-graph requires --capture-last-layers and runs the backward during generation")
    if args.eval_workload and not args.adam_graph:
        parser.error("--eval-workload requires an online optimizer")
    if args.adam_graph and (min(args.update_every, args.max_updates) < 1
                            or not all(isfinite(v) and v > 0 for v in (args.learning_rate, args.max_gradient_norm))):
        parser.error("Online updates require positive frequencies, limits, learning rate and clip norm")

    import torch
    from blockspec.dsv41_online_profile import profile_online
    from blockspec.dsv41_packed_proposer import StagedPackedDraftProposer, load_staged_draft_graphs, read_packed_head
    from blockspec.dsv41_staged_adam import StagedPackedAdam, load_staged_adam_programs
    from blockspec.dsv41_staged_target import load_staged_target_programs, open_staged_target
    from blockspec.dsv41_staged_reverse import StagedDraftReverse, load_staged_reverse_programs
    from blockspec.pypto_dsv41_cluster import ControlLink, PeerDispatchError, RankPlan
    from blockspec.vllm_ascend.head_contract import find_weight_index
    from models.deepseek_v4_1_flash_a3.config import DP_SIZE, FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from tokenizers import Tokenizer

    torch.set_num_threads(4)

    def load_workload(path):
        payload = path.read_bytes()
        rows = json.loads(payload)
        if not rows or len({case["id"] for case in rows}) != len(rows):
            parser.error(f"{path}: provide at least one request with a unique ID")
        for case in rows:
            if (not case["input_ids"] or not 0 <= case["dp"] < DP_SIZE
                    or not 1 <= case["new_tokens"] <= args.context - len(case["input_ids"])
                    or any(type(token) is not int or not 0 <= token < FLASH.vocab_size for token in case["input_ids"])):
                parser.error(f"Invalid workload request {case['id']!r}")
        if args.draft_head and not any(
            case["new_tokens"] >= 3 and len(case["input_ids"]) + args.draft_block <= args.context for case in rows
        ):
            parser.error(f"{path}: leave room for a draft block and at least one candidate")
        return rows, sha256(payload).hexdigest()

    cases, workload = load_workload(args.workload)
    evaluation, eval_workload = None, None
    warmup, warmup_workload = None, None
    if args.eval_workload:
        evaluation, eval_workload = load_workload(args.eval_workload)
    if args.warmup_workload:
        warmup, warmup_workload = load_workload(args.warmup_workload)
    seen_prompts = set()
    for partition in (cases, evaluation or (), warmup or ()):
        prompts = {tuple(case["input_ids"]) for case in partition}
        if prompts & seen_prompts:
            parser.error("Warmup, online and evaluation workloads must use separate prompts")
        seen_prompts |= prompts

    capacities = args.prefill_capacity, args.decode_capacity
    tokens = args.prefill_moe_tokens, 16 if args.decode_capacity == 16 else 64
    devices = list(range(8))
    plan = RankPlan((8, 8), TP_SIZE, args.node)
    config = RunConfig(
        backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ring_dep_pool=131072, ring_heap=(4 << 30, 4 << 30, 2 << 30, 1 << 30),
    )
    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    engram = HostEngram(checkpoint, Tokenizer.from_file(str(checkpoint.root / "tokenizer.json")))
    programs = load_staged_target_programs(
        args.prefill_graph, args.decode_graph, capacities, tokens,
        context=args.context, slots=args.slots_per_dp, local_ranks=len(devices),
    )
    key = args.key_file.read_bytes().strip()
    signature = dict(workload=workload, evaluation=eval_workload, warmup=warmup_workload, checkpoint=checkpoint.identity,
                     capacities=list(capacities), context=args.context, slots=args.slots_per_dp, tokens=list(tokens))
    draft_programs = ()
    reverse_programs = {}
    adam_programs = {}
    online = None
    if args.draft_head:
        artifact = read_packed_head(checkpoint, args.draft_head, args.draft_block)
        draft_programs = load_staged_draft_graphs(
            args.draft_graph, context=args.context, block=args.draft_block,
            local_ranks=len(devices), request_slots=args.slots_per_dp, last_layers=args.capture_last_layers,
        )
        if args.check_backward or args.adam_graph:
            reverse_programs = load_staged_reverse_programs(args.draft_graph, args.capture_last_layers)
        if args.adam_graph:
            adam_programs = load_staged_adam_programs(args.adam_graph, FLASH, TP_SIZE)
            online = dict(every=args.update_every, max_updates=args.max_updates, rate=args.learning_rate,
                          max_norm=args.max_gradient_norm)
        signature["draft"] = dict(
            head=sha256((args.draft_head / "manifest.json").read_bytes()
                        + find_weight_index(args.draft_head).read_bytes()).hexdigest(),
            graphs=sha256((args.draft_graph / "orthrus_draft_manifest.json").read_bytes()).hexdigest(),
            width=args.verify_width,
            teacher_rows=args.teacher_rows if args.capture_last_layers else 0,
            check_backward=args.check_backward,
            online=online,
        )

    def measured(session, mode, width=0, teacher_rows=0, reverse=None, optimizer=None, *,
                 requests=cases, reference=None):
        before = dict(vars(session.draft.stats)) if width else None
        metrics = dict(stage_seconds=Counter(), counts=Counter())
        start = perf_counter()
        with profile_online(SimpleNamespace(current=metrics, lock=Lock())):
            arm = generate(session, requests, capacities[0], DP_SIZE,
                           width=width, teacher_rows=teacher_rows, reverse=reverse,
                           optimizer=optimizer, online=online, slots_per_dp=min(args.slots_per_dp, 16),
                           expected=None if reference is None else reference["generated"], mode=mode) | metrics
        elapsed = perf_counter() - start
        arm["wall_seconds"] = elapsed
        arm["wall_tok_per_s"] = sum(map(len, arm["generated"].values())) / elapsed
        if width:
            arm["draft_profile"] = {key: value - before[key] for key, value in vars(session.draft.stats).items()}
        return arm

    with ControlLink.connect(plan.node, (args.rank0_ip, args.port), key, args.timeout) as link, ExitStack() as workers:
        if link.exchange("staged-target/start", signature) != signature:
            raise ValueError("Both hosts must use the same checkpoint, workload and artifact geometry")
        start = perf_counter()
        warmup_seconds = {}
        try:
            target = workers.enter_context(open_staged_target(
                config, plan, link, weights, engram, programs, capacities=capacities,
                context=args.context, slots=args.slots_per_dp, tokens=tokens,
                draft_programs=(*draft_programs, *reverse_programs.values(),
                                *(program for entry in adam_programs.values() for program in entry)),
            ))
            initialization_seconds = perf_counter() - start
            session = target.session()
            warm_ar = None
            if warmup:
                warm_ar = measured(session, "warmup_ar", requests=warmup)
                warmup_seconds["ar"] = warm_ar["wall_seconds"]
            if draft_programs:
                start = perf_counter()
                session.draft = StagedPackedDraftProposer(
                    target, draft_programs, draft_head=args.draft_head, head_artifact=artifact, block=args.draft_block,
                    last_layers=args.capture_last_layers,
                )
                reverse = StagedDraftReverse(session.draft, config, reverse_programs) if reverse_programs else None
                optimizer = StagedPackedAdam(session.draft, reverse, FLASH, adam_programs) if adam_programs else None
                draft_init = perf_counter() - start
                if warmup:
                    warmup_seconds["orthrus"] = measured(
                        session, "warmup_orthrus", args.verify_width, requests=warmup, reference=warm_ar,
                    )["wall_seconds"]
            print(json.dumps(dict(stage="generate", mode="ar", requests=len(cases))), flush=True)
            result = measured(session, "ar")
            arms = {"ar": result}
            if draft_programs:
                if evaluation:
                    arms["eval_ar"] = measured(session, "eval_ar", requests=evaluation)
                    arms["eval_initial"] = measured(session, "eval_initial", args.verify_width, requests=evaluation,
                                                   reference=arms["eval_ar"])
                print(json.dumps(dict(stage="generate", mode="online" if optimizer else "orthrus",
                                      requests=len(cases))), flush=True)
                arms["orthrus"] = measured(session, "online" if optimizer else "orthrus", args.verify_width,
                                           args.teacher_rows if args.capture_last_layers else 0,
                                           reverse if args.check_backward else None, optimizer, reference=result)
                arms["orthrus"].update(
                    initialization_seconds=draft_init, resident_head_bytes=session.draft.resident_head_bytes,
                    width=args.verify_width, block=args.draft_block, head_role=session.draft.head_role,
                )
                if evaluation:
                    arms["eval_frozen"] = measured(session, "eval_frozen", args.verify_width, requests=evaluation,
                                                  reference=arms["eval_ar"])
                    arms["eval_frozen"]["head_version"] = optimizer.adam.version
        except Exception as exc:
            if isinstance(exc, GreedyDivergence):
                args.output.write_text(json.dumps(dict(passed=False, node=args.node, divergence=exc.witness),
                                                  indent=2) + "\n")
            if not isinstance(exc, PeerDispatchError):
                with suppress(OSError):
                    link.abort("staged-target", exc)
            raise
        generated = {mode: arm["generated"] for mode, arm in arms.items()}
        generated_identity = sha256(json.dumps(generated, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = arms.get("orthrus", {}).get("capture")
        peer = link.exchange("staged-target/finish", dict(generated=generated_identity, capture=receipt))
        if peer["generated"] != generated_identity:
            raise AssertionError("Hosts observed different generated IDs")

    golden = json.loads(args.golden.read_text()) if args.golden else None
    replay_passed = True
    if args.capture_last_layers and not args.adam_graph:
        owners = [item for item in (receipt, peer["capture"]) if item and "teacher_rows" in item]
        replay_passed = (
            len(owners) == 1 and receipt is not None and peer["capture"] is not None
            and receipt["request"] == peer["capture"]["request"] and receipt["dp"] == peer["capture"]["dp"]
            and owners[0]["teacher_rows"] == owners[0]["expected_rows"]
        )
    parity = None
    if "orthrus" in arms:
        parity = compare_generation(arms["ar"], arms["orthrus"])
    report = dict(passed=(golden is None or golden == result["generated"]) and (parity is None or parity["passed"])
                  and replay_passed,
                  node=args.node,
                  global_ranks=list(plan.local_ranks), initialization_seconds=initialization_seconds,
                  golden_compared=golden is not None, warmup_seconds=warmup_seconds, configuration=signature, **result)
    if parity is not None:
        report.update(arms=arms, parity=parity)
    if args.capture_last_layers and not args.adam_graph:
        report["teacher_replay"] = dict(passed=replay_passed, layers=args.capture_last_layers,
                                        local=receipt, peer=peer["capture"])
    if args.check_backward:
        outcome = arms["orthrus"]["backward"]
        report["backward_passed"] = bool(
            outcome is not None and outcome["teacher_rows"] == owners[0]["expected_rows"]
            and len(outcome["layers"]) == args.capture_last_layers and isfinite(outcome["loss"])
        ) if replay_passed else False
        report["passed"] &= report["backward_passed"]
    if args.adam_graph:
        result = arms["orthrus"]["online"]
        report["online_passed"] = bool(result["version"] and result["rounds_after_publication"])
        report["passed"] &= report["online_passed"]
    if evaluation:
        report["frozen_eval_parity"] = {
            name: compare_generation(arms["eval_ar"], arms[name]) for name in ("eval_initial", "eval_frozen")
        }
        report["passed"] &= all(item["passed"] for item in report["frozen_eval_parity"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], node=args.node, results=str(args.output), parity=parity,
                          frozen_eval_parity=report.get("frozen_eval_parity"), online_passed=report.get("online_passed"),
                          arms={name: {key: arm[key] for key in ("wall_tok_per_s", "decode_tok_per_s",
                                                               "request_batches", "max_live_requests", "speculation",
                                                               "acceptance")}
                                for name, arm in arms.items()})), flush=True)
    if not report["passed"]:
        raise SystemExit("Staged acceptance failed. See the parity and online update results in the report.")


if __name__ == "__main__":
    main()
