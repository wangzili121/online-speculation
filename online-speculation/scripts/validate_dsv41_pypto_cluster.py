#!/usr/bin/env python3
"""Run the two-host dispatch and HCCL path with node-local Ascend workers."""

import argparse
import json
import os
from pathlib import Path
import sys

from blockspec.pypto_dsv41_cluster import ClusterDispatcher, ControlLink, RankPlan
from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives


def run(args, plan):
    address, key = (args.master_address, args.port), args.key_file.read_bytes().strip()
    with ControlLink.connect(plan.node, address, key, args.timeout) as link:
        dispatch = ClusterDispatcher(plan, link, build_id=args.build_id)
        if args.control_only:
            return check_dispatch(dispatch, plan)
        return check_device(args, plan, link, dispatch)


def check_device(args, plan, link, dispatch):
    import torch

    sys.path.insert(0, str(args.lib.resolve()))
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(120, args.timeout)))
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", str(args.timeout))
    torch.set_num_threads(2)
    local_devices = len(plan.local_ranks)
    topology = dict(ep=local_devices, tp=plan.tp, dp=local_devices // plan.tp)
    with ResidentNetworkRuntime(list(range(local_devices)), topology) as runtime, ClusterCollectives(runtime, plan, link) as comm:
        return check_dispatch(dispatch, plan, runtime, comm)


def check_dispatch(dispatch, plan, runtime=None, comm=None):
    operations = (
        ("prefill", "tp", "all_gather"), ("draft", "ep", "all_to_all"),
        ("verify", "dp", "all_gather"), ("accept", "ep", "reduce_scatter"),
        ("decode", "ep", "all_reduce"), ("release", "tp", "all_gather"),
    )
    world = sum(plan.counts)
    topology = dict(dp=world // plan.tp, tp=plan.tp, ep=world)
    active_dps = range(0, topology["dp"], 2 if topology["dp"] > 2 else 1)
    for tag, (phase, group, method) in enumerate(operations, 1):
        engines = [
            dict(dp_rank=dp, active=int(dp in active_dps),
                 requests=[dict(request_slot=0, generation=1, token_ids=[tag])] if dp in active_dps else [])
            for dp in range(topology["dp"])
        ]
        contract = dict(topology=topology, DP_engines=engines)

        def execute(wave):
            local = plan.local_ranks
            local_engines = engines[local.start // plan.tp:local.stop // plan.tp]
            if wave["DP_engines"] != local_engines:
                raise ValueError(f"Incorrect request routing for {phase}")
            if comm is not None:
                check_exchange(plan, runtime, comm, phase, group, method, tag)
            return [[tag + engine["dp_rank"]] for engine in local_engines for _ in engine["requests"]]

        results = dispatch.execute_wave(phase, execute, contract=contract if plan.node == 0 else None,
                                        collect_tokens=phase != "release")
        if phase != "release" and results != [[tag + dp] for dp in active_dps]:
            raise AssertionError("Cluster token rows were not restored to Serving wave order")
    return dict(passed=True, node=plan.node, global_ranks=list(plan.local_ranks), epoch=dispatch.epoch,
                phases=[phase for phase, _, _ in operations], topology=topology,
                device_collectives=[f"{group}/{method}" for _, group, method in operations] if comm else [])


def check_exchange(plan, runtime, comm, phase, group, method, tag):
    import torch

    def value(rank):
        dtype = torch.uint8 if method == "all_to_all" else torch.bfloat16
        return (torch.arange(sum(plan.counts) * 8).reshape(-1, 1) % 13 + rank + tag).to(dtype)

    packets = [{"phase": {"value": value(rank)}} for rank in plan.local_ranks]
    try:
        outputs, receipts = comm.exchange(phase, packets, method, group)
    except RuntimeError as error:
        raise RuntimeError(f"{phase} {group}/{method}: {error}") from error
    for rank, result, receipt in zip(plan.local_ranks, outputs, receipts, strict=True):
        members, = (axis for axis in plan.groups(group) if rank in axis)
        if receipt["group_ranks"] != list(members) or receipt["activation_staging"]:
            raise AssertionError("Wrong global owners or unexpected activation staging")
        peers = [value(peer) for peer in members]
        index = members.index(rank)
        if method == "all_gather":
            expected = torch.cat(peers)
        elif method == "all_to_all":
            expected = torch.cat([peer.chunk(len(members))[index] for peer in peers])
        else:
            expected = torch.stack(peers).sum(0, dtype=torch.float32).bfloat16()
            if method == "reduce_scatter":
                expected = expected.chunk(len(members))[index]
        physical = rank - plan.local_ranks.start
        actual = runtime.read_output(result["phase"]["value"], physical)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    runtime.release_values()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--counts", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--port", type=int, default=29642)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()
    if not args.control_only and args.lib is None:
        parser.error("Specify --lib for the device-collective check")
    plan = RankPlan(tuple(args.counts), args.tp, args.node)
    try:
        report = run(args, plan)
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        print(json.dumps(dict(passed=False, node=args.node, error=str(error))), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
