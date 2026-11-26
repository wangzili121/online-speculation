#!/usr/bin/env python3
"""Run the same packed 1122 W8A8 greedy target workload on both EP16 hosts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace


def generate(session, cases, prefill_rows, dp_size):
    """Pack prompt chunks by DP owner and continue from device-greedy IDs."""
    handles = [session.admit(case["id"], case["dp"], len(case["input_ids"])) for case in cases]
    output = {handle: [] for handle in handles}
    offsets = [0] * len(cases)
    requests = list(zip(cases, handles, strict=True))
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
            output[handle].append(token)

    prefill_seconds = perf_counter() - start
    start = perf_counter()
    steps = 0
    while batch := [(handle, [output[handle][-1]]) for case, handle in requests
                    if len(output[handle]) < case["new_tokens"]]:
        for handle, token in session.decode_greedy_batch(batch).items():
            output[handle].append(token)
        steps += 1
    decode_seconds = perf_counter() - start
    for handle in handles:
        session.release(handle)
    result = {handle.request_id: output[handle] for handle in handles}
    tok = sum(map(len, result.values())) - len(cases)
    return dict(generated=result, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                decode_steps=steps, decode_tokens=tok, decode_tok_per_s=tok / decode_seconds if steps else 0.0)


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from blockspec.dsv41_online_profile import profile_online
    from blockspec.dsv41_staged_target import load_staged_target_programs, open_staged_target
    from blockspec.pypto_dsv41_cluster import ControlLink, RankPlan
    from models.deepseek_v4_1_flash_a3.config import DP_SIZE, FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    workload = args.workload.read_bytes()
    cases = json.loads(workload)
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("Provide at least one request with a unique ID")
    for case in cases:
        if (not case["input_ids"] or not 0 <= case["dp"] < DP_SIZE
                or not 1 <= case["new_tokens"] <= args.context - len(case["input_ids"])
                or any(type(token) is not int or not 0 <= token < FLASH.vocab_size for token in case["input_ids"])):
            parser.error(f"Invalid workload request {case['id']!r}")
    if any(sum(case["dp"] == dp for case in cases) > min(args.slots_per_dp, 16) for dp in range(DP_SIZE)):
        parser.error("The workload must fit the configured resident request slots and vocabulary output rows")

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
    signature = dict(workload=sha256(workload).hexdigest(), checkpoint=checkpoint.identity,
                     capacities=list(capacities), context=args.context, slots=args.slots_per_dp, tokens=list(tokens))
    with ControlLink.connect(plan.node, (args.rank0_ip, args.port), key, args.timeout) as link:
        if link.exchange("staged-target/start", signature) != signature:
            raise ValueError("Both hosts must use the same checkpoint, workload and artifact geometry")
        start = perf_counter()
        with open_staged_target(config, plan, link, weights, engram, programs, capacities=capacities,
                                context=args.context, slots=args.slots_per_dp, tokens=tokens) as target:
            initialization_seconds = perf_counter() - start
            stage_metrics = dict(stage_seconds=Counter(), counts=Counter())
            with profile_online(SimpleNamespace(current=stage_metrics, lock=Lock())):
                result = generate(target.session(), cases, capacities[0], DP_SIZE)
        if link.exchange("staged-target/finish", result["generated"]) != result["generated"]:
            raise AssertionError("Hosts observed different generated IDs")

    golden = json.loads(args.golden.read_text()) if args.golden else None
    report = dict(passed=golden is None or golden == result["generated"], node=args.node,
                  global_ranks=list(plan.local_ranks), initialization_seconds=initialization_seconds,
                  golden_compared=golden is not None, **result, **stage_metrics)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Staged target greedy IDs differ from the supplied golden")


if __name__ == "__main__":
    main()
