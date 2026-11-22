#!/usr/bin/env python3
"""Run the two-host dispatch and HCCL path with node-local Ascend workers."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

from blockspec.pypto_dsv41_cluster import ClusterDispatcher, ControlLink, RankPlan
from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives


def run(args, plan):
    address, key = (args.master_address, args.port), args.key_file.read_bytes().strip()
    mode = "control" if args.control_only else "device"
    identity = f"{mode}/{args.build_id or build_identity(args.lib if not args.control_only else None)}"
    print(f"Node {plan.node} connecting to {address[0]}:{address[1]}, {identity}", file=sys.stderr, flush=True)
    with ControlLink.connect(plan.node, address, key, args.timeout) as link:
        dispatch = ClusterDispatcher(plan, link, build_id=identity)
        result = check_dispatch(dispatch, plan) if args.control_only else check_device(args, plan, link, dispatch)
    return dict(result, build_id=identity)


def build_identity(lib):
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / "src/blockspec/pypto_dsv41_cluster.py",
             root / "src/blockspec/pypto_dsv41_cluster_transport.py"]
    if lib is not None:
        source = lib / "models/deepseek_v4_1_flash_a3"
        files.extend((source / "prefill_resident.py", source / "moe_transport.py"))
    fingerprint = hashlib.sha256()
    for file in files:
        fingerprint.update(hashlib.sha256(file.read_bytes()).digest())
    return fingerprint.hexdigest()[:16]


def check_device(args, plan, link, dispatch):
    import torch

    sys.path.insert(0, str(args.lib.resolve()))
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(120, args.timeout)))
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", str(args.timeout))
    torch.set_num_threads(2)
    local_devices = len(plan.local_ranks)
    physical_cards = os.environ.get("ORTHRUS_CLUSTER_PHYSICAL_CARDS")
    card_map = None
    if physical_cards:
        cards = [int(card) for card in physical_cards.split(",")]
        if len(cards) != local_devices or len(set(cards)) != local_devices:
            raise ValueError(f"The rank plan requires {local_devices} distinct local NPU devices")
        card_map = check_visible_devices(cards)
        print(f"Node {plan.node} physical-to-runtime NPU mapping: {card_map}", file=sys.stderr, flush=True)
    topology = dict(ep=local_devices, tp=plan.tp, dp=local_devices // plan.tp)
    with ResidentNetworkRuntime(list(range(local_devices)), topology) as runtime, ClusterCollectives(runtime, plan, link) as comm:
        result = check_dispatch(dispatch, plan, runtime, comm)
    return dict(result, physical_to_runtime=card_map) if card_map is not None else result


def check_visible_devices(cards, *, native=None):
    if native is None:
        from models.deepseek_v4_1_flash_a3 import moe_transport as native

    acl = native.CANNMemory().library
    count = ctypes.c_uint32()
    native._check(acl.aclrtGetDeviceCount(ctypes.byref(count)), "aclrtGetDeviceCount")
    if count.value != len(cards):
        raise ValueError(f"CANN reports {count.value} visible devices for the {len(cards)} mounted cards")
    mapping = {}
    # The launcher exposes all mounted container logical IDs in order.
    for runtime_id in range(count.value):
        physical = ctypes.c_int32()
        native._check(acl.aclrtGetPhyDevIdByLogicDevId(runtime_id, ctypes.byref(physical)),
                      f"aclrtGetPhyDevIdByLogicDevId({runtime_id})")
        mapping[physical.value] = runtime_id
    if list(mapping) != cards:
        raise ValueError(f"Unexpected CANN device order {mapping}; order NPU_DEVICES by its runtime IDs")
    return mapping


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
        print(f"Node {plan.node} completed {phase}" + (f" ({group}/{method})" if comm else ""),
              file=sys.stderr, flush=True)
    return dict(passed=True, node=plan.node, global_ranks=list(plan.local_ranks), epoch=dispatch.epoch,
                phases=[phase for phase, _, _ in operations], topology=topology,
                device_collectives=[f"{group}/{method}" for _, group, method in operations] if comm else [])


def check_exchange(plan, runtime, comm, phase, group, method, tag):
    import torch

    def value(rank):
        rows = torch.arange(sum(plan.counts) * 8).reshape(-1, 1)
        if method == "all_to_all":
            return ((rows * 3 + rank * 17 + tag) % 251).to(torch.uint8)
        # Each EP16 sum is at most 192 and is exact for BF16 accumulation.
        return ((rows + rank + tag) % 13).to(torch.bfloat16)

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
    parser.add_argument("--lib", type=Path, default=os.environ.get("PYPTO_LIB_ROOT"))
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--build-id", help="Override the automatically computed source fingerprint")
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
