"""Check the 40-layer draft sequence with one or more real TP4 head shards."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW
from dsv41_pypto_indexer_fixture import fixture


def histories(runtime, attention, anchor, ranks, slot, generation):
    pinned = []

    def bind(kind, ratio, value, stride, offset, table):
        runtime.pin(value, attention.rank_ids.values(), readonly=False, name=f"raw/{slot}/{generation}/{kind}/{ratio}")
        pinned.append(value)
        return {rank: dict(handle=runtime.handle(value, attention.rank_ids[rank], readonly=True), pages=table,
                     shape=(value.shape[0] // stride, WINDOW // ratio, 1, value.shape[1]),
                     stride_rows=stride, offset_rows=offset) for rank in ranks}

    swa = torch.randn((4 * WINDOW, HEAD_DIM)).bfloat16()
    local = bind("swa", 1, swa, WINDOW, 0, (1, 2, 3))
    planes = {0: {rank: {"swa": (local[rank],)} for rank in ranks}}
    for ratio in (1, 2):
        geometry, buffers, _, kv, _, gold = fixture(
            attention.config, anchor, ratio, max_context=attention.max_context,
        )
        pages = tuple(gold.table.flatten().tolist())
        long = bind("long", ratio, kv, geometry.kv_stride, geometry.kv_offset, pages)
        keys = bind("key", ratio, buffers["index_codes"], geometry.code_stride, geometry.code_offset, pages)
        scales = bind("scale", ratio, buffers["index_scales"], geometry.scale_stride, geometry.scale_offset, pages)
        planes[ratio] = {rank: dict(swa=(local[rank],), long=(long[rank],), index=(keys[rank], scales[rank]))
                         for rank in ranks}
    owner = ranks[0] // attention.topology["tp"], slot, generation
    table = {}
    for layer in range(40):
        role = native("cache").resolve_layer_role(layer)
        for rank in ranks:
            table[rank, layer, owner] = dict(role=role, committed_tokens=anchor, planes=planes[role["ratio"]][rank])
    return table, pinned


def check_rank_reduction(runtime, values, outputs, method, layer):
    packets = [{"draft": {"contribution": value}} for value in values]
    received, receipts = runtime.exchange(f"draft/layer{layer}", packets, method, "tp")
    fp32 = torch.stack(outputs).float()
    reference, magnitude = fp32.sum(0), fp32.abs().sum(0)
    eps = torch.finfo(torch.bfloat16).eps
    largest = 0.0
    for rank, packet in enumerate(received):
        actual = runtime.read_output(packet["draft"]["contribution"], rank).float()
        expected = reference.chunk(len(values))[rank] if method == "reduce_scatter" else reference
        limit = magnitude.chunk(len(values))[rank] if method == "reduce_scatter" else magnitude
        error = (actual - expected).abs()
        assert (error <= 2 * eps * limit).all(), f"Reduction mismatch at layer {layer}, rank {rank}"
        assert not receipts[rank]["activation_staging"]
        largest = max(largest, float(error.max()))
    runtime.release_tensors([(rank, packet["draft"]["contribution"]) for rank, packet in enumerate(received)])
    return largest


def prepare_round(runtime, attention, ranks, anchors, generation):
    capacity = 128
    table, pinned, items = {}, [], []
    positions = torch.zeros(capacity)
    for slot, anchor in enumerate(anchors):
        history, caches = histories(runtime, attention, anchor, ranks, slot, generation)
        table.update(history)
        pinned.extend(caches)
        begin, end = (slot + 1) * BLOCK, (slot + 2) * BLOCK
        positions[begin:end] = torch.arange(anchor, anchor + BLOCK)
        items.append(dict(request_slot=slot, token_begin=begin, token_end=end, initial_context_length=anchor))
    active = items[-1]["token_end"]
    record = dict(request_token_map=items)
    width = attention.config["hidden_size"]
    hidden = torch.randn((capacity, width)).bfloat16()
    phase = positions[:, None] * torch.linspace(0.001, 0.11, 32)[None]
    cos, sin = phase.cos(), phase.sin()
    return table, pinned, active, record, width, hidden, cos, sin


def run_round(runtime, attention, args, ranks, anchors, generation, prepared):
    table, pinned, active, record, width, hidden, cos, sin = prepared
    capacity = hidden.shape[0]
    draft = attention.start_round(table)
    replay = args.replay
    roles = []
    collective_error = 0.0
    fingerprint = hashlib.sha256() if replay else None
    for layer in range(40):
        role = native("cache").resolve_layer_role(layer)
        roles.append(role["mode"])

        def project(local):
            rank = ranks[local]
            contribution = runtime.workspace(hidden.shape, hidden.dtype, local)
            draft.project_layer(rank, layer, record, hidden, cos, sin, contribution)
            output = runtime.read_output(contribution, local)
            for block in output[BLOCK:active].split(BLOCK):
                assert torch.isfinite(block).all() and block.abs().max() > 0, f"Invalid layer {layer}, rank {rank} output"
            assert not output[:BLOCK].any() and not output[active:].any()
            if role["index_source"] is not None and layer in (2, 3, 20, 21, 24, 25):
                for slot, anchor in enumerate(anchors):
                    owner = rank // attention.topology["tp"], slot, generation
                    rows = draft.selection[rank, owner, role["index_source"]][1]
                    assert (runtime.read_control(rows["lengths"], local) == anchor // role["ratio"]).all()
            if args.cards == 4 or (args.cards == 1 and layer == 39):
                physical = active // 4
                packed = runtime.workspace((4 * capacity, width), hidden.dtype, local)
                draft.pack(rank, contribution, packed, physical=physical, total=active)
                packed_output = runtime.read_output(packed, local).view(4, capacity, width)
                assert torch.equal(packed_output[:, :physical].reshape(active, width), output[:active])
                assert not packed_output[:, physical:].any()
                return [contribution, packed], packed_output.flatten(0, 1)
            return [contribution], output

        result = runtime.map_ranks(project)
        values, outputs = zip(*result, strict=True)
        if fingerprint is not None:
            for output in outputs:
                fingerprint.update(output.contiguous().view(torch.uint8).numpy())
        if args.cards > 1:
            method = "reduce_scatter" if args.cards == 4 else "all_reduce"
            error = check_rank_reduction(runtime, [row[-1] for row in values], outputs, method, layer)
            collective_error = max(collective_error, error)
        runtime.release_values(draft.finish_layer(layer))
    assert not draft.selection
    for rank in range(len(ranks)):
        for value in pinned:
            assert torch.equal(value.view(torch.uint8), runtime.read_output(value, rank).view(torch.uint8))
    return roles, collective_error, fingerprint.hexdigest() if fingerprint is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lib", "model", "head", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cards", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--requests", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rank-offset", type=int, choices=(0, 4, 8, 12), default=0,
                        help="Global rank at the start of the local TP4 partition")
    parser.add_argument("--replay", action="store_true", help="Interleave a second generation and replay the first")
    args = parser.parse_args()
    sys.path.insert(0, str(args.lib.resolve()))
    args.output.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    devices = list(range(args.cards))
    ranks = list(range(args.rank_offset, args.rank_offset + args.cards))
    topology = dict(ep=args.cards, dp=1, tp=args.cards)
    with native("prefill_resident").ResidentNetworkRuntime(devices, topology) as runtime:
        attention = ResidentDraftAttention(
            args.model, args.head, runtime, dict(ep=16 if args.rank_offset else args.cards, tp=4),
            args.output / "kernels", local_ranks=ranks,
        )
        anchors = (257, 129)[:args.requests]
        primary = prepare_round(runtime, attention, ranks, anchors, 1)
        before = runtime.stats["device_weight_upload_calls"]
        roles, collective_error, fingerprint = run_round(runtime, attention, args, ranks, anchors, 1, primary)
        replay_passed = None
        if args.replay:
            other_anchors = (255, 127)[:args.requests]
            other = prepare_round(runtime, attention, ranks, other_anchors, 2)
            _, middle_error, other_fingerprint = run_round(runtime, attention, args, ranks, other_anchors, 2, other)
            _, replay_error, replay_fingerprint = run_round(runtime, attention, args, ranks, anchors, 1, primary)
            assert fingerprint != other_fingerprint, "Distinct generations produced identical draft contributions"
            assert fingerprint == replay_fingerprint, "Interleaved generation changed repeated draft outputs"
            collective_error = max(collective_error, middle_error, replay_error)
            replay_passed = True
        assert before == runtime.stats["device_weight_upload_calls"]
        result = dict(physical_devices=args.cards, global_ranks=ranks, tp4_lanes=devices, full_tp4=args.cards == 4,
                      anchors=list(anchors), layers=list(range(40)),
                      roles={role: roles.count(role) for role in sorted(set(roles))},
                      block=BLOCK, cache_unchanged=True, weights_reuploaded=False,
                      dispatches=runtime.stats["dispatches"], weight_bytes=runtime.stats["weight_bytes"],
                      collective_max_abs_error=collective_error, transport_calls=runtime.stats["transport_calls"])
        if replay_passed:
            result.update(rounds=3, replay_byte_exact=True, interleaved_anchors=list(other_anchors))
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
