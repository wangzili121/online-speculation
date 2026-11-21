"""Check the 40-layer draft sequence with one or more real TP4 head shards."""

import argparse
import json
import sys
from pathlib import Path

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW
from dsv41_pypto_indexer_fixture import fixture


def histories(runtime, attention, anchor, ranks, slot):
    pinned = []

    def bind(kind, ratio, value, stride, offset, table):
        runtime.pin(value, ranks, readonly=False, name=f"raw/{slot}/{kind}/{ratio}")
        pinned.append(value)
        return [dict(handle=runtime.handle(value, rank, readonly=True), pages=table,
                     shape=(value.shape[0] // stride, WINDOW // ratio, 1, value.shape[1]),
                     stride_rows=stride, offset_rows=offset) for rank in ranks]

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
    owner = 0, slot, 1
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lib", "model", "head", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cards", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--requests", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lib.resolve()))
    args.output.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    ranks = list(range(args.cards))
    topology = dict(ep=args.cards, dp=1, tp=args.cards)
    with native("prefill_resident").ResidentNetworkRuntime(ranks, topology) as runtime:
        attention = ResidentDraftAttention(
            args.model, args.head, runtime, dict(ep=args.cards, tp=4), args.output / "kernels",
        )
        anchors, capacity = (257, 129)[:args.requests], 128
        table, pinned, items = {}, [], []
        positions = torch.zeros(capacity)
        for slot, anchor in enumerate(anchors):
            history, caches = histories(runtime, attention, anchor, ranks, slot)
            table.update(history)
            pinned.extend(caches)
            begin, end = (slot + 1) * BLOCK, (slot + 2) * BLOCK
            positions[begin:end] = torch.arange(anchor, anchor + BLOCK)
            items.append(dict(request_slot=slot, token_begin=begin, token_end=end, initial_context_length=anchor))
        draft = attention.start_round(table)
        active = items[-1]["token_end"]
        record = dict(request_token_map=items)
        width = attention.config["hidden_size"]
        hidden = torch.randn((capacity, width)).bfloat16()
        phase = positions[:, None] * torch.linspace(0.001, 0.11, 32)[None]
        cos, sin = phase.cos(), phase.sin()
        before = runtime.stats["device_weight_upload_calls"]
        roles = []
        collective_error = 0.0
        for layer in range(40):
            role = native("cache").resolve_layer_role(layer)
            roles.append(role["mode"])

            def project(rank):
                contribution = runtime.workspace(hidden.shape, hidden.dtype, rank)
                draft.project_layer(rank, layer, record, hidden, cos, sin, contribution)
                output = runtime.read_output(contribution, rank)
                for block in output[BLOCK:active].split(BLOCK):
                    assert torch.isfinite(block).all() and block.abs().max() > 0, f"Invalid layer {layer}, rank {rank} output"
                assert not output[:BLOCK].any() and not output[active:].any()
                if role["index_source"] is not None and layer in (2, 3, 20, 21, 24, 25):
                    for slot, anchor in enumerate(anchors):
                        rows = draft.selection[rank, (0, slot, 1), role["index_source"]][1]
                        assert (runtime.read_control(rows["lengths"], rank) == anchor // role["ratio"]).all()
                if args.cards == 4 or (args.cards == 1 and layer == 39):
                    physical = active // 4
                    packed = runtime.workspace((4 * capacity, width), hidden.dtype, rank)
                    draft.pack(rank, contribution, packed, physical=physical, total=active)
                    packed_output = runtime.read_output(packed, rank).view(4, capacity, width)
                    assert torch.equal(packed_output[:, :physical].reshape(active, width), output[:active])
                    assert not packed_output[:, physical:].any()
                    return [contribution, packed], packed_output.flatten(0, 1)
                return [contribution], output

            result = runtime.map_ranks(project)
            if args.cards > 1:
                values, outputs = zip(*result, strict=True)
                method = "reduce_scatter" if args.cards == 4 else "all_reduce"
                error = check_rank_reduction(runtime, [row[-1] for row in values], outputs, method, layer)
                collective_error = max(collective_error, error)
            runtime.release_values(draft.finish_layer(layer))
        assert not draft.selection
        for rank in ranks:
            for value in pinned:
                assert torch.equal(value.view(torch.uint8), runtime.read_output(value, rank).view(torch.uint8))
        assert before == runtime.stats["device_weight_upload_calls"]
        result = dict(physical_devices=args.cards, tp4_lanes=ranks, full_tp4=args.cards == 4,
                      anchors=list(anchors), layers=list(range(40)),
                      roles={role: roles.count(role) for role in sorted(set(roles))},
                      block=BLOCK, cache_unchanged=True, weights_reuploaded=False,
                      dispatches=runtime.stats["dispatches"], weight_bytes=runtime.stats["weight_bytes"],
                      collective_max_abs_error=collective_error, transport_calls=runtime.stats["transport_calls"])
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
