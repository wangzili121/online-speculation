#!/usr/bin/env python3
"""Run 1122 W8A8 AR and optionally compare a resident Orthrus head on both EP16 hosts."""

import argparse
from collections import Counter
from contextlib import ExitStack, suppress
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace


def generate(session, cases, prefill_rows, dp_size, *, width=0):
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
    speculation = Counter(draft_rounds=0, proposed=0, accepted=0)
    while pending := [(handle, case["new_tokens"] - len(output[handle])) for case, handle in requests
                      if len(output[handle]) < case["new_tokens"]]:
        if width:
            # The fixed draft block serves one resident request per DP in a wave.
            owners = {}
            for handle, remaining in pending:
                owners.setdefault(handle.dp, (handle, min(width, remaining)))
            limits = dict(owners.values())
            wave = session.speculate_batch(
                [(handle, output[handle][-1]) for handle in limits],
                width=width, output_limits=limits,
            )
            speculation["draft_rounds"] += any(wave.proposals.values())
            speculation["proposed"] += sum(map(len, wave.proposals.values()))
            for handle, emitted in wave.verified.emitted.items():
                output[handle].extend(emitted)
                speculation["accepted"] += len(emitted) - 1
        else:
            batch = [(handle, [output[handle][-1]]) for handle, _ in pending]
            for handle, token in session.decode_greedy_batch(batch).items():
                output[handle].append(token)
        steps += 1
    decode_seconds = perf_counter() - start
    for handle in handles:
        session.release(handle)
    result = {handle.request_id: output[handle] for handle in handles}
    tok = sum(map(len, result.values())) - len(cases)
    return dict(generated=result, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                decode_steps=steps, decode_tokens=tok, decode_tok_per_s=tok / decode_seconds if steps else 0.0,
                speculation=speculation)


def first_difference(expected, actual):
    """Identify the first differing or missing token across identical workloads."""
    from itertools import zip_longest

    for request, tokens in expected.items():
        for position, (reference, observed) in enumerate(zip_longest(tokens, actual[request])):
            if reference != observed:
                return dict(request=request, position=position, expected=reference, observed=observed)
    return None


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.draft_head) != bool(args.draft_graph):
        parser.error("Provide both --draft-head and --draft-graph to compare Orthrus")

    import torch
    from blockspec.dsv41_online_profile import profile_online
    from blockspec.dsv41_packed_proposer import StagedPackedDraftProposer, load_staged_draft_graphs, read_packed_head
    from blockspec.dsv41_staged_target import load_staged_target_programs, open_staged_target
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
    workload = args.workload.read_bytes()
    cases = json.loads(workload)
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("Provide at least one request with a unique ID")
    for case in cases:
        if (not case["input_ids"] or not 0 <= case["dp"] < DP_SIZE
                or not 1 <= case["new_tokens"] <= args.context - len(case["input_ids"])
                or any(type(token) is not int or not 0 <= token < FLASH.vocab_size for token in case["input_ids"])):
            parser.error(f"Invalid workload request {case['id']!r}")
    if args.draft_head and not any(
        case["new_tokens"] >= 3 and len(case["input_ids"]) + args.draft_block <= args.context for case in cases
    ):
        parser.error("The workload must leave room for a draft block and at least one candidate")
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
    draft_programs = ()
    if args.draft_head:
        artifact = read_packed_head(checkpoint, args.draft_head, args.draft_block)
        draft_programs = load_staged_draft_graphs(
            args.draft_graph, context=args.context, block=args.draft_block,
            local_ranks=len(devices), request_slots=args.slots_per_dp,
        )
        signature["draft"] = dict(
            head=sha256((args.draft_head / "manifest.json").read_bytes()
                        + find_weight_index(args.draft_head).read_bytes()).hexdigest(),
            graphs=sha256((args.draft_graph / "orthrus_draft_manifest.json").read_bytes()).hexdigest(),
            width=args.verify_width,
        )

    def measured(session, width=0):
        metrics = dict(stage_seconds=Counter(), counts=Counter())
        with profile_online(SimpleNamespace(current=metrics, lock=Lock())):
            return generate(session, cases, capacities[0], DP_SIZE, width=width) | metrics

    with ControlLink.connect(plan.node, (args.rank0_ip, args.port), key, args.timeout) as link, ExitStack() as workers:
        if link.exchange("staged-target/start", signature) != signature:
            raise ValueError("Both hosts must use the same checkpoint, workload and artifact geometry")
        start = perf_counter()
        try:
            target = workers.enter_context(open_staged_target(
                config, plan, link, weights, engram, programs, capacities=capacities,
                context=args.context, slots=args.slots_per_dp, tokens=tokens, draft_programs=draft_programs,
            ))
            initialization_seconds = perf_counter() - start
            session = target.session()
            result = measured(session)
            arms = {"ar": result}
            if draft_programs:
                start = perf_counter()
                session.draft = StagedPackedDraftProposer(
                    target, draft_programs, draft_head=args.draft_head, head_artifact=artifact, block=args.draft_block,
                )
                draft_init = perf_counter() - start
                arms["orthrus"] = measured(session, args.verify_width)
                arms["orthrus"].update(
                    initialization_seconds=draft_init, resident_head_bytes=session.draft.resident_head_bytes,
                    draft_profile=vars(session.draft.stats), width=args.verify_width,
                    block=args.draft_block, head_role=session.draft.head_role,
                )
        except Exception as exc:
            if not isinstance(exc, PeerDispatchError):
                with suppress(OSError):
                    link.abort("staged-target", exc)
            raise
        generated = {mode: arm["generated"] for mode, arm in arms.items()}
        if link.exchange("staged-target/finish", generated) != generated:
            raise AssertionError("Hosts observed different generated IDs")

    golden = json.loads(args.golden.read_text()) if args.golden else None
    parity = None
    if "orthrus" in arms:
        difference = first_difference(generated["ar"], generated["orthrus"])
        parity = dict(passed=difference is None and arms["orthrus"]["speculation"]["draft_rounds"] > 0,
                      compared_tokens=sum(map(len, generated["ar"].values())), first_difference=difference)
    report = dict(passed=(golden is None or golden == result["generated"]) and (parity is None or parity["passed"]),
                  node=args.node,
                  global_ranks=list(plan.local_ranks), initialization_seconds=initialization_seconds,
                  golden_compared=golden is not None, **result)
    if parity is not None:
        report.update(arms=arms, parity=parity)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Staged greedy acceptance failed. See the parity and generated IDs in the report.")


if __name__ == "__main__":
    main()
