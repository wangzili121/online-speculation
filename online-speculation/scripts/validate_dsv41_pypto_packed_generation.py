#!/usr/bin/env python3
"""Compare native AR and resident Orthrus on the same packed A3 workload."""

import argparse
from collections import Counter
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter
from types import SimpleNamespace


@dataclass
class OnlineSchedule:
    every: int
    teacher_rows: int
    learning_rate: float
    warmup_updates: int
    rounds: int = 0
    updates: int = 0
    next_capture: int = 1
    next_dp: int = 0

    def capture(self, eligible):
        self.rounds += 1
        if self.rounds < self.next_capture or not eligible:
            return None
        handle = next((item for item in eligible if item.dp >= self.next_dp), eligible[0])
        self.next_capture = self.rounds + self.every
        self.next_dp = handle.dp + 1
        return handle, self.teacher_rows

    @property
    def rate(self):
        return self.learning_rate * min(1., (self.updates + 1) / max(self.warmup_updates, 1))


def append_emitted(outputs, emitted, expected, phase):
    matched = 0
    divergence = None
    for handle, rows in emitted.items():
        output = outputs[handle]
        start = len(output)
        output.extend(rows)
        if expected is None or divergence is not None:
            continue
        target = expected[handle.request_id][start:len(output)]
        offset = next((i for i, (a, b) in enumerate(zip(target, rows)) if a != b), min(len(target), len(rows)))
        matched += offset
        if offset < len(rows):
            divergence = dict(request_id=handle.request_id, dp=handle.dp, phase=phase,
                              position=start + offset, target=target[offset:offset + 1],
                              orthrus=list(rows[offset:offset + 1]))
    return matched, divergence


def run_wave(session, requests, *, chunk, tokens, width, speculative, online=None, expected=None):
    """Run one wave, with one request per DP and a fixed output budget."""
    begin = perf_counter()
    staged = {
        session.admit(request_id, dp, len(prompt)): prompt
        for dp, (request_id, prompt) in enumerate(requests)
    }
    outputs = {handle: [] for handle in staged}
    matched = 0
    divergence = None
    for start in range(0, max(map(len, staged.values())), chunk):
        first_tokens = session.prefill_greedy_batch(
            (handle, prompt[start:start + chunk]) for handle, prompt in staged.items() if start < len(prompt)
        )
        checked, divergence = append_emitted(outputs, {handle: (token,) for handle, token in first_tokens.items()},
                                             expected, "prefill")
        matched += checked
        if divergence:
            break
    if divergence is None and any(len(rows) != 1 for rows in outputs.values()):
        raise RuntimeError("Packed Prefill must produce exactly one token per request")
    prefill = perf_counter()
    steps = proposed = accepted = draft_requests = 0
    pending = None
    updates = []
    draft_before = asdict(session.draft.stats) if speculative else {}
    while divergence is None and (active := {handle: rows[-1] for handle, rows in outputs.items() if len(rows) < tokens}):
        steps += 1
        if speculative:
            capture = None
            if online:
                eligible = [
                    handle for handle in active
                    if pending is None and tokens - len(outputs[handle]) > 1
                    and len(staged[handle]) + len(outputs[handle]) - 1 + session.draft.block <= session.context
                ]
                capture = online.capture(eligible)
                if capture:
                    pending, rows = capture
                    capture = pending, min(rows, tokens - len(outputs[pending]))
            result = session.speculate_batch(
                active.items(), width=width,
                output_limits={handle: min(width, tokens - len(outputs[handle])) for handle in active},
                student_capture=capture,
            )
            proposed += sum(map(len, result.proposals.values()))
            draft_requests += sum(bool(rows) for rows in result.proposals.values())
            accepted += sum(len(rows) - 1 for rows in result.verified.emitted.values())
            emitted = result.verified.emitted
        else:
            result = session.decode_greedy_batch((handle, (anchor,)) for handle, anchor in active.items())
            emitted = {handle: (token,) for handle, token in result.items()}
        if emitted.keys() != active.keys() or any(not 1 <= len(rows) <= tokens - len(outputs[handle])
                                                  for handle, rows in emitted.items()):
            raise RuntimeError("Packed Decode must advance each live request within its output budget")
        checked, divergence = append_emitted(outputs, emitted, expected, "verify/accept" if speculative else "decode")
        matched += checked
        if divergence:
            divergence["round"] = steps
            if speculative:
                failed = next(handle for handle in emitted if handle.dp == divergence["dp"])
                divergence["draft_candidates"] = list(result.proposals[failed])
                divergence["emitted"] = list(emitted[failed])
            break
        if pending is not None:
            begin_update = perf_counter()
            update = session.update_draft(pending, rate=online.rate, finished=len(outputs[pending]) == tokens)
            if update is not None:
                updates.append(dict(request=pending.request_id, dp=pending.dp, learning_rate=online.rate,
                                    wall_seconds=perf_counter() - begin_update, **asdict(update)))
                online.updates += int(update.applied)
                pending = None
    decode = perf_counter()
    for handle in staged:
        session.release(handle)
    finish = perf_counter()
    draft_stats = {key: value - draft_before[key] for key, value in asdict(session.draft.stats).items()} if speculative else {}
    draft_seconds = sum(draft_stats[key] for key in (
        "prepare_seconds", "upload_seconds", "graph_seconds", "readback_seconds",
    )) if speculative else 0.0
    update_seconds = sum(update["wall_seconds"] for update in updates)
    return {
        "requests": {handle.request_id: rows for handle, rows in outputs.items()},
        "greedy_tokens_matched": matched,
        "divergence": divergence,
        "rounds": steps,
        "proposed_candidates": proposed,
        "accepted_candidates": accepted,
        "draft_requests": draft_requests,
        "prefill_seconds": prefill - begin,
        "decode_seconds": decode - prefill,
        "draft_seconds": draft_seconds,
        "draft_stats": draft_stats,
        "verify_accept_and_dispatch_seconds": decode - prefill - draft_seconds - update_seconds if speculative else None,
        "updates": updates,
        "update_seconds": update_seconds,
        "release_seconds": finish - decode,
        "total_seconds": finish - begin,
    }


def benchmark(session, requests, *, chunk, tokens, width, concurrency, warmup, online=None, on_wave=None):
    from blockspec.dsv41_online_profile import profile_online

    begin = perf_counter()
    waves = [requests[start:start + concurrency] for start in range(0, len(requests), concurrency)]

    def run(wave, mode, schedule=None, expected=None):
        profile = dict(stage_seconds=Counter(), counts=Counter())
        with profile_online(SimpleNamespace(current=profile, lock=Lock())):
            result = run_wave(session, wave, chunk=chunk, tokens=tokens, width=width,
                              speculative=mode, online=schedule, expected=expected)
        return dict(result, profile=profile)

    divergence = failed_wave = None
    for _ in range(warmup):
        ar = run(waves[0], False)
        orthrus = run(waves[0], True, expected=ar["requests"])
        if orthrus["divergence"]:
            divergence = dict(orthrus["divergence"], warmup=True)
            failed_wave = dict(index=0, ar=ar, orthrus=orthrus)
            if on_wave is not None:
                on_wave(0, len(waves), ar, orthrus, divergence)
            break
    warmed = perf_counter()
    arms = {name: [] for name in ("ar", "orthrus")}
    matched = 0
    for index, wave in enumerate(waves if divergence is None else (), 1):
        ar = run(wave, False)
        orthrus = run(wave, True, online, expected=ar["requests"])
        matched += orthrus["greedy_tokens_matched"]
        divergence = orthrus["divergence"]
        if on_wave is not None:
            on_wave(index, len(waves), ar, orthrus, divergence)
        if divergence:
            failed_wave = dict(index=index, ar=ar, orthrus=orthrus)
            break
        arms["ar"].append(ar)
        arms["orthrus"].append(orthrus)
    for name, samples in arms.items():
        emitted = sum(len(rows) for wave in samples for rows in wave["requests"].values())
        seconds = sum(wave["total_seconds"] for wave in samples)
        decode_seconds = sum(wave["decode_seconds"] for wave in samples)
        updated = sum(wave["update_seconds"] for wave in samples)
        updates = [update for wave in samples for update in wave["updates"]]
        proposed = sum(wave["proposed_candidates"] for wave in samples)
        accepted = sum(wave["accepted_candidates"] for wave in samples)
        draft_requests = sum(wave["draft_requests"] for wave in samples)
        decode_tokens = emitted - sum(len(wave["requests"]) for wave in samples)
        arms[name] = dict(
            waves=samples, emitted_tokens=emitted, tokens_per_second=emitted / seconds if samples else None,
            decode_tokens_per_second=decode_tokens / decode_seconds if samples else None,
            update_seconds=updated, tokens_per_second_excluding_update=emitted / (seconds - updated) if samples else None,
            decode_tokens_per_second_excluding_update=decode_tokens / (decode_seconds - updated) if samples else None,
            updates_applied=sum(update["applied"] for update in updates),
            first_update_seconds=updates[0]["wall_seconds"] if updates else None,
            later_updates_median_seconds=median(update["wall_seconds"] for update in updates[1:])
            if len(updates) > 1 else None,
            proposed_candidates=proposed, accepted_candidates=accepted,
            candidate_acceptance=accepted / proposed if proposed else None,
            mean_accepted_per_draft=accepted / draft_requests if draft_requests else None,
        )
    return dict(passed=divergence is None and (online is None or online.updates > 0),
                online_schedule=asdict(online) if online else None,
                greedy_tokens_matched=matched, divergence=divergence, failed_wave=failed_wave,
                concurrency=concurrency, output_tokens_per_request=tokens, verify_width=width,
                warmup_seconds=warmed - begin, arms=arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "graph-dir", "draft-graph", "head", "workload", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--devices", default=",".join(map(str, range(16))))
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--width", type=int, choices=(8, 16), default=8)
    parser.add_argument("--draft-block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--online-last-layers", type=int, default=0)
    parser.add_argument("--online-loss", choices=("tv", "forward_kl"), default="tv")
    parser.add_argument("--update-every", type=int, default=16)
    parser.add_argument("--teacher-rows", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--heldout-workload", type=Path)
    parser.add_argument("--resident-teacher", action="store_true",
                        help="Profile device snapshots against the default committed-row host transfer")
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    if len(devices) != 16 or len(set(devices)) != 16:
        parser.error("The current runtime requires 16 distinct visible devices")
    if args.tokens < 2 or args.warmup < 0:
        parser.error("Use at least two output tokens and a nonnegative warmup count")
    if args.online_last_layers and (args.update_every < 1 or not 1 <= args.teacher_rows < args.width
                                    or not isfinite(args.learning_rate) or args.learning_rate <= 0
                                    or args.warmup_updates < 0):
        parser.error("Use positive update cadence and learning rate, and teacher rows within the draft width")
    if args.resident_teacher and not args.online_last_layers:
        parser.error("--resident-teacher requires an online suffix update")
    if args.heldout_workload and not args.online_last_layers:
        parser.error("A held-out workload compares the frozen head before and after online updates")

    from tokenizers import Tokenizer
    from blockspec.dsv41_online import read_workload
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.driver import load_manifest
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.generation import HbmSampler
    from models.deepseek_v4_1_flash_a3.packed import open_packed_session
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    prefill_dir, decode_dir = (args.graph_dir / phase for phase in ("prefill", "decode"))
    if any(not (path / "distributed_meta.json").is_file() for path in (prefill_dir, decode_dir)):
        parser.error("--graph-dir must contain the compiled 1122 native Prefill and Decode whole graphs")
    prefill, decode = (load_manifest(path, mode=phase) for phase, path in (
        ("prefill", prefill_dir), ("decode", decode_dir),
    ))
    context = prefill["history_capacity"]
    max_suffix = FLASH.num_hidden_layers - max(FLASH.engram_layer_ids)
    if not 0 <= args.online_last_layers <= max_suffix:
        parser.error(f"The differentiable suffix length must be within 1..{max_suffix}, or 0 for frozen inference")
    workload = [(row["id"], tokenizer.encode(row["prompt"]).ids) for row in read_workload(args.workload)]
    heldout = []
    if args.heldout_workload:
        heldout = [(row["id"], tokenizer.encode(row["prompt"]).ids) for row in read_workload(args.heldout_workload)]
        if {tuple(prompt) for _, prompt in workload}.intersection(tuple(prompt) for _, prompt in heldout):
            parser.error("The online and held-out workloads share an identical prompt")
    if any(not prompt or len(prompt) + args.tokens > context for _, prompt in (*workload, *heldout)):
        parser.error("Each prompt plus its generation budget must fit the compiled context")
    options = SimpleNamespace(runtime_dir=prefill_dir, decode_runtime_dir=decode_dir,
                              context_capacity=context, capacity=prefill["capacity"],
                              decode_capacity=decode["capacity"], max_new_tokens=args.tokens,
                              profile_all_steps=None, orthrus_draft_head=args.head,
                              orthrus_draft_runtime_dir=args.draft_graph, orthrus_draft_block=args.draft_block,
                              orthrus_online_last_layers=args.online_last_layers,
                              orthrus_resident_teacher=args.resident_teacher,
                              orthrus_online_loss=args.online_loss if args.online_last_layers else None)
    schedule = (OnlineSchedule(args.update_every, args.teacher_rows, args.learning_rate, args.warmup_updates)
                if args.online_last_layers else None)
    report = dict(
        checkpoint_identity=checkpoint.identity, target_execution="native_l3_full_graph",
        graph_capacities={phase: manifest["capacity"] for phase, manifest in (("prefill", prefill), ("decode", decode))},
        context_capacity=context, topology=prefill["topology"],
        workload_sha256=sha256(args.workload.read_bytes()).hexdigest(),
        online_last_layers=args.online_last_layers, online_loss=options.orthrus_online_loss,
        teacher_capture="device_snapshot" if args.resident_teacher else "committed_host_rows",
    )
    memory = HbmSampler(devices)
    begin = perf_counter()
    with open_packed_session(options, devices, ResidentWeights(checkpoint), HostEngram(checkpoint, tokenizer),
                             memory, report, request_slots=1) as session:
        report["startup_seconds"] = perf_counter() - begin
        sizing = dict(chunk=prefill["capacity"], tokens=args.tokens, width=args.width, concurrency=args.concurrency)
        if heldout:
            report["heldout_workload_sha256"] = sha256(args.heldout_workload.read_bytes()).hexdigest()
            report["heldout_initial"] = benchmark(session, heldout, **sizing, warmup=args.warmup)
        report.update(benchmark(session, workload, **sizing, warmup=args.warmup, online=schedule))
        if heldout:
            report["heldout_published"] = benchmark(session, heldout, **sizing, warmup=args.warmup)
            report["passed"] &= report["heldout_initial"]["passed"] and report["heldout_published"]["passed"]
        report["draft_head_role"] = session.draft.head_role
        report["draft_block"] = session.draft.block
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "greedy_tokens_matched", "divergence")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
