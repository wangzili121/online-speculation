"""Check the 40-layer draft sequence with one or more real TP4 head shards."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_sparse import TOPK
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
    pages = tuple(1 + page % 3 for page in range(max(3, (anchor + WINDOW - 1) // WINDOW)))
    local = bind("swa", 1, swa, WINDOW, 0, pages)
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
    block_size = attention.block_size
    prefix = block_size if len(anchors) < capacity // block_size else 0
    table, pinned, items = {}, [], []
    positions = torch.zeros(capacity)
    for slot, anchor in enumerate(anchors):
        history, caches = histories(runtime, attention, anchor, ranks, slot, generation)
        table.update(history)
        pinned.extend(caches)
        begin, end = prefix + slot * block_size, prefix + (slot + 1) * block_size
        positions[begin:end] = torch.arange(anchor, anchor + block_size)
        items.append(dict(request_slot=slot, token_begin=begin, token_end=end, initial_context_length=anchor))
    active = items[-1]["token_end"]
    record = dict(request_token_map=items)
    width = attention.config["hidden_size"]
    hidden = torch.randn((capacity, width)).bfloat16()
    phase = positions[:, None] * torch.linspace(0.001, 0.11, 32)[None]
    cos, sin = phase.cos(), phase.sin()
    return table, pinned, active, record, width, hidden, cos, sin


def run_round(runtime, attention, args, ranks, anchors, generation, prepared):
    block_size = attention.block_size
    table, pinned, active, record, width, hidden, cos, sin = prepared
    capacity = hidden.shape[0]
    draft = attention.start_round(table)
    roles = []
    collective_error = 0.0
    prefix = record["request_token_map"][0]["token_begin"]
    comparison = (args.compare_joined_attention or args.compare_scratch_fill
                  or args.compare_swa_controls or args.compare_index_scratch)
    fingerprint = hashlib.sha256() if args.replay or comparison else None
    for layer in range(40):
        role = native("cache").resolve_layer_role(layer)
        roles.append(role["mode"])

        def project(local):
            rank = ranks[local]
            contribution = runtime.workspace(hidden.shape, hidden.dtype, local)
            draft.project_layer(rank, layer, record, hidden, cos, sin, contribution)
            output = runtime.read_output(contribution, local)
            for block in output[prefix:active].split(block_size):
                assert torch.isfinite(block).all() and block.abs().max() > 0, f"Invalid layer {layer}, rank {rank} output"
            assert not output[:prefix].any() and not output[active:].any()
            if role["index_source"] is not None and layer in (2, 3, 20, 21, 24, 25):
                for slot, anchor in enumerate(anchors):
                    owner = rank // attention.topology["tp"], slot, generation
                    rows = draft.selection[rank, owner, role["index_source"]][1]
                    assert (runtime.read_control(rows["lengths"], local) == min(TOPK, anchor // role["ratio"])).all()
            if args.cards == 4 or (args.cards == 1 and layer == 39):
                physical = active // 4
                packed = runtime.workspace((4 * capacity, width), hidden.dtype, local)
                draft.pack(rank, contribution, packed, physical=physical, total=active)
                packed_output = runtime.read_output(packed, local).view(4, capacity, width)
                assert torch.equal(packed_output[:, :physical].reshape(active, width), output[:active])
                assert not packed_output[:, physical:].any()
                if layer == 39:
                    widened = draft.pack_fp32(rank, contribution, physical=physical, total=active)
                    assert torch.equal(runtime.read_output(widened, local), packed_output.flatten(0, 1).float())
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
    parser.add_argument("--block-size", type=int, choices=(8, 16, BLOCK), default=BLOCK)
    parser.add_argument("--requests", type=int, choices=(1, 2), default=1)
    parser.add_argument("--anchors", type=int, nargs="+", help="One to four context lengths in the same DP partition")
    parser.add_argument("--rank-offset", type=int, choices=(0, 4, 8, 12), default=0,
                        help="Global rank at the start of the local TP4 partition")
    parser.add_argument("--replay", action="store_true", help="Interleave a second generation and replay the first")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-joined-attention", action="store_true",
                            help="Compare the 40-layer draft with grouped SWA/compressed dispatch")
    comparison.add_argument("--compare-scratch-fill", action="store_true",
                            help="Compare reused output buffers against zero-filled buffers")
    comparison.add_argument("--compare-swa-controls", action="store_true",
                            help="Compare fresh vs resident SWA controls across all draft layers")
    comparison.add_argument("--compare-index-scratch", action="store_true",
                            help="Compare zero-filled vs overwritten full-block indexer scratch")
    args = parser.parse_args()
    if args.anchors is not None and (len(args.anchors) > 4 or min(args.anchors) < 0):
        parser.error("--anchors requires one to four nonnegative context lengths")
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
            args.output / "kernels", local_ranks=ranks, joined_attention=not args.compare_joined_attention, block_size=args.block_size,
        )
        anchors = tuple(args.anchors) if args.anchors is not None else (257, 129)[:args.requests]
        primary = prepare_round(runtime, attention, ranks, anchors, 1)
        upload_counts = ("device_weight_upload_calls", "device_draft_weight_upload_calls")
        before = [runtime.stats.get(name, 0) for name in upload_counts]
        start_dispatches = runtime.stats["dispatches"]
        roles, collective_error, fingerprint = run_round(runtime, attention, args, ranks, anchors, 1, primary)
        joined_dispatches = None
        if args.compare_joined_attention:
            baseline_dispatches = runtime.stats["dispatches"] - start_dispatches
            attention.joined_attention = True
            start_dispatches = runtime.stats["dispatches"]
            _, joined_error, joined_fingerprint = run_round(runtime, attention, args, ranks, anchors, 1, primary)
            if fingerprint != joined_fingerprint:
                raise ValueError("Grouped dispatch changed draft output bytes")
            collective_error = max(collective_error, joined_error)
            joined_dispatches = dict(baseline=baseline_dispatches,
                                     joined=runtime.stats["dispatches"] - start_dispatches,
                                     output_byte_exact=True)
            samples = {False: [], True: []}
            for _ in range(3):
                for joined in samples:
                    attention.joined_attention = joined
                    started = perf_counter()
                    _, _, sample = run_round(runtime, attention, args, ranks, anchors, 1, primary)
                    samples[joined].append((perf_counter() - started) * 1000)
                    if sample != fingerprint:
                        raise ValueError("A repeated draft round changed output bytes")
            joined_dispatches.update(steady_repeats=3, baseline_wall_ms=median(samples[False]),
                                     joined_wall_ms=median(samples[True]))
        scratch_comparison = None
        control_comparison = None
        if args.compare_swa_controls or args.compare_index_scratch:
            samples = {"previous": [], "optimized": []}
            traffic = {}
            counter = "h2d_bytes" if args.compare_swa_controls else "workspace_fill_bytes"
            for _ in range(5):
                for mode in samples:
                    attention.reuse_swa_controls = not args.compare_swa_controls or mode == "optimized"
                    attention.clear_index_scratch = args.compare_index_scratch and mode == "previous"
                    before_bytes = runtime.stats[counter]
                    started = perf_counter()
                    _, _, sample = run_round(runtime, attention, args, ranks, anchors, 1, primary)
                    samples[mode].append((perf_counter() - started) * 1000)
                    traffic[mode] = runtime.stats[counter] - before_bytes
                    if sample != fingerprint:
                        raise ValueError("Reusing draft buffers changed output bytes")
            control_comparison = {"wall_ms": {mode: median(times) for mode, times in samples.items()},
                                  counter: traffic, "output_byte_exact": True}
        if args.compare_scratch_fill:
            raw_workspace = runtime.workspace

            def zero_workspace(shape, dtype, rank, *, fill=0):
                return raw_workspace(shape, dtype, rank, fill=0 if fill is None else fill)

            samples = {"zero": [], "reuse": []}
            filled_bytes = {}
            try:
                for _ in range(3):
                    for mode in samples:
                        runtime.workspace = zero_workspace if mode == "zero" else raw_workspace
                        before_fill = runtime.stats["workspace_fill_bytes"]
                        started = perf_counter()
                        _, _, sample = run_round(runtime, attention, args, ranks, anchors, 1, primary)
                        samples[mode].append((perf_counter() - started) * 1000)
                        filled_bytes[mode] = runtime.stats["workspace_fill_bytes"] - before_fill
                        if sample != fingerprint:
                            raise ValueError("Reused scratch changed draft output bytes")
            finally:
                runtime.workspace = raw_workspace
            scratch_comparison = dict(wall_ms={mode: median(times) for mode, times in samples.items()},
                                      filled_bytes=filled_bytes, output_byte_exact=True)
        replay_passed = None
        if args.replay:
            other_anchors = tuple(max(0, anchor - 2) for anchor in anchors)
            other = prepare_round(runtime, attention, ranks, other_anchors, 2)
            _, middle_error, other_fingerprint = run_round(runtime, attention, args, ranks, other_anchors, 2, other)
            _, replay_error, replay_fingerprint = run_round(runtime, attention, args, ranks, anchors, 1, primary)
            assert fingerprint != other_fingerprint, "Distinct generations produced identical draft contributions"
            assert fingerprint == replay_fingerprint, "Interleaved generation changed repeated draft outputs"
            collective_error = max(collective_error, middle_error, replay_error)
            replay_passed = True
        assert before == [runtime.stats.get(name, 0) for name in upload_counts]
        draft_bytes = runtime.stats.get("draft_weight_bytes", runtime.stats["weight_bytes"])
        result = dict(physical_devices=args.cards, global_ranks=ranks, tp4_lanes=devices, full_tp4=args.cards == 4,
                      anchors=list(anchors), layers=list(range(40)),
                      roles={role: roles.count(role) for role in sorted(set(roles))},
                      block=args.block_size, cache_unchanged=True, weights_reuploaded=False,
                      dispatches=runtime.stats["dispatches"], weight_bytes=draft_bytes,
                      collective_max_abs_error=collective_error, transport_calls=runtime.stats["transport_calls"])
        if replay_passed:
            result.update(rounds=3, replay_byte_exact=True, interleaved_anchors=list(other_anchors))
        if joined_dispatches is not None:
            result["joined_attention"] = joined_dispatches
        if scratch_comparison is not None:
            result["scratch_fill"] = scratch_comparison
        if control_comparison is not None:
            result["swa_controls" if args.compare_swa_controls else "index_scratch"] = control_comparison
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
