#!/usr/bin/env python3
"""Compare native AR and resident Orthrus on the same packed A3 workload."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


def run_wave(session, requests, *, chunk, tokens, width, speculative):
    """Run one wave, with one request per DP and a fixed output budget."""
    begin = perf_counter()
    staged = {
        session.admit(request_id, dp, len(prompt)): prompt
        for dp, (request_id, prompt) in enumerate(requests)
    }
    outputs = {handle: [] for handle in staged}
    for start in range(0, max(map(len, staged.values())), chunk):
        first_tokens = session.prefill_greedy_batch(
            (handle, prompt[start:start + chunk]) for handle, prompt in staged.items() if start < len(prompt)
        )
        for handle, token in first_tokens.items():
            outputs[handle].append(token)
    if any(len(rows) != 1 for rows in outputs.values()):
        raise RuntimeError("Packed Prefill must produce exactly one token per request")
    prefill = perf_counter()
    steps = proposed = accepted = 0
    draft_before = asdict(session.draft.stats) if speculative else {}
    while active := {handle: rows[-1] for handle, rows in outputs.items() if len(rows) < tokens}:
        steps += 1
        if speculative:
            result = session.speculate_batch(
                active.items(), width=width,
                output_limits={handle: min(width, tokens - len(outputs[handle])) for handle in active},
            )
            proposed += sum(map(len, result.proposals.values()))
            accepted += sum(len(rows) - 1 for rows in result.verified.emitted.values())
            emitted = result.verified.emitted
        else:
            result = session.decode_greedy_batch((handle, (anchor,)) for handle, anchor in active.items())
            emitted = {handle: (token,) for handle, token in result.items()}
        if emitted.keys() != active.keys() or any(not 1 <= len(rows) <= tokens - len(outputs[handle])
                                                  for handle, rows in emitted.items()):
            raise RuntimeError("Packed Decode must advance each live request within its output budget")
        for handle, rows in emitted.items():
            outputs[handle].extend(rows)
    decode = perf_counter()
    for handle in staged:
        session.release(handle)
    finish = perf_counter()
    draft_stats = {key: value - draft_before[key] for key, value in asdict(session.draft.stats).items()} if speculative else {}
    draft_seconds = sum(draft_stats[key] for key in (
        "prepare_seconds", "upload_seconds", "graph_seconds", "readback_seconds",
    )) if speculative else 0.0
    return {
        "requests": {handle.request_id: rows for handle, rows in outputs.items()},
        "rounds": steps,
        "proposed_candidates": proposed,
        "accepted_candidates": accepted,
        "prefill_seconds": prefill - begin,
        "decode_seconds": decode - prefill,
        "draft_seconds": draft_seconds,
        "draft_stats": draft_stats,
        "verify_accept_and_dispatch_seconds": decode - prefill - draft_seconds if speculative else None,
        "release_seconds": finish - decode,
        "total_seconds": finish - begin,
    }


def benchmark(session, requests, *, chunk, tokens, width, concurrency, warmup):
    begin = perf_counter()
    waves = [requests[start:start + concurrency] for start in range(0, len(requests), concurrency)]

    def run(wave, mode):
        return run_wave(session, wave, chunk=chunk, tokens=tokens, width=width, speculative=mode)

    for _ in range(warmup):
        run(waves[0], False)
        run(waves[0], True)
    warmed = perf_counter()
    arms = {name: [] for name in ("ar", "orthrus")}
    matched = 0
    divergence = None
    for wave in waves:
        ar, orthrus = run(wave, False), run(wave, True)
        arms["ar"].append(ar)
        arms["orthrus"].append(orthrus)
        for request_id, _ in wave:
            target, observed = (arm["requests"][request_id] for arm in (ar, orthrus))
            offset = next((i for i, (left, right) in enumerate(zip(target, observed)) if left != right),
                          min(len(target), len(observed)))
            matched += offset
            if offset != tokens:
                divergence = dict(request_id=request_id, position=offset,
                                  target=target[offset:offset + 1], orthrus=observed[offset:offset + 1])
                break
        if divergence:
            break
    for name, samples in arms.items():
        emitted = sum(len(rows) for wave in samples for rows in wave["requests"].values())
        seconds = sum(wave["total_seconds"] for wave in samples)
        decode_seconds = sum(wave["decode_seconds"] for wave in samples)
        arms[name] = dict(
            waves=samples, emitted_tokens=emitted, tokens_per_second=emitted / seconds,
            decode_tokens_per_second=(emitted - sum(len(wave["requests"]) for wave in samples)) / decode_seconds,
        )
    return dict(passed=divergence is None, greedy_tokens_matched=matched, divergence=divergence,
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
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    if len(devices) != 16 or len(set(devices)) != 16:
        parser.error("The current runtime requires 16 distinct visible devices")
    if args.tokens < 2 or args.warmup < 0:
        parser.error("Use at least two output tokens and a nonnegative warmup count")

    from tokenizers import Tokenizer
    from blockspec.dsv41_online import read_workload
    from models.deepseek_v4_1_flash_a3.driver import load_manifest
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.generation import HbmSampler
    from models.deepseek_v4_1_flash_a3.packed import open_packed_session
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    prefill_dir, decode_dir = (args.graph_dir / phase for phase in ("prefill", "decode"))
    prefill, decode = (load_manifest(path, mode=phase) for phase, path in (
        ("prefill", prefill_dir), ("decode", decode_dir),
    ))
    context = prefill["history_capacity"]
    workload = [(row["id"], tokenizer.encode(row["prompt"]).ids) for row in read_workload(args.workload)]
    if any(not prompt or len(prompt) + args.tokens > context for _, prompt in workload):
        parser.error("Each prompt plus its generation budget must fit the compiled context")
    options = SimpleNamespace(runtime_dir=prefill_dir, decode_runtime_dir=decode_dir,
                              context_capacity=context, capacity=prefill["capacity"],
                              decode_capacity=decode["capacity"], max_new_tokens=args.tokens,
                              profile_all_steps=None, orthrus_draft_head=args.head,
                              orthrus_draft_runtime_dir=args.draft_graph, orthrus_draft_block=args.draft_block)
    report = dict(
        checkpoint_identity=checkpoint.identity,
        graph_capacities={phase: manifest["capacity"] for phase, manifest in (("prefill", prefill), ("decode", decode))},
        context_capacity=context, topology=prefill["topology"],
        workload_sha256=sha256(args.workload.read_bytes()).hexdigest(),
    )
    memory = HbmSampler(devices)
    begin = perf_counter()
    with open_packed_session(options, devices, ResidentWeights(checkpoint), HostEngram(checkpoint, tokenizer),
                             memory, report, request_slots=1) as session:
        report["startup_seconds"] = perf_counter() - begin
        report.update(benchmark(session, workload, chunk=prefill["capacity"], tokens=args.tokens,
                                width=args.width, concurrency=args.concurrency, warmup=args.warmup))
        report["draft_head_role"] = session.draft.head_role
        report["draft_block"] = session.draft.block
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "greedy_tokens_matched", "divergence")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
