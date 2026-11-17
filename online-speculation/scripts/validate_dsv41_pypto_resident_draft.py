"""Check the 40-layer attention sequence on one physical NPU with a real TP4 head shard."""

import argparse
import json
import sys
from pathlib import Path

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW
from dsv41_pypto_indexer_fixture import fixture


def histories(runtime, attention, anchor):
    pinned = []

    def bind(kind, ratio, value, stride, offset, table):
        runtime.pin(value, (0,), readonly=False, name=f"raw/{kind}/{ratio}")
        pinned.append(value)
        return dict(handle=runtime.handle(value, 0, readonly=True), pages=table,
                    shape=(value.shape[0] // stride, WINDOW // ratio, 1, value.shape[1]),
                    stride_rows=stride, offset_rows=offset)

    swa = torch.randn((4 * WINDOW, HEAD_DIM)).bfloat16()
    local = (bind("swa", 1, swa, WINDOW, 0, (1, 2, 3)),)
    planes = {0: {"swa": local}}
    for ratio in (1, 2):
        geometry, buffers, _, kv, _, gold = fixture(
            attention.config, anchor, ratio, max_context=attention.max_context,
        )
        pages = tuple(gold.table.flatten().tolist())
        planes[ratio] = dict(swa=local,
            long=(bind("long", ratio, kv, geometry.kv_stride, geometry.kv_offset, pages),),
            index=(
                bind("key", ratio, buffers["index_codes"], geometry.code_stride, geometry.code_offset, pages),
                bind("scale", ratio, buffers["index_scales"], geometry.scale_stride, geometry.scale_offset, pages),
            ),
        )
    owner = 0, 0, 1
    table = {}
    for layer in range(40):
        role = native("cache").resolve_layer_role(layer)
        table[0, layer, owner] = dict(role=role, committed_tokens=anchor, planes=planes[role["ratio"]])
    return table, pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lib", "model", "head", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lib.resolve()))
    args.output.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    with native("prefill_resident").ResidentNetworkRuntime([0], dict(ep=1, dp=1, tp=1)) as runtime:
        attention = ResidentDraftAttention(
            args.model, args.head, runtime, dict(ep=1, tp=4), args.output / "kernels",
        )
        anchor, capacity = 257, 128
        table, pinned = histories(runtime, attention, anchor)
        draft = attention.start_round(table)
        width = attention.config["hidden_size"]
        hidden = torch.randn((capacity, width)).bfloat16()
        phase = torch.arange(anchor - BLOCK, anchor - BLOCK + capacity)[:, None] * torch.linspace(0.001, 0.11, 32)[None]
        cos, sin = phase.cos(), phase.sin()
        record = dict(request_token_map=[dict(request_slot=0, token_begin=BLOCK, token_end=2 * BLOCK,
                                             initial_context_length=anchor)])
        before = runtime.stats["device_weight_upload_calls"]
        roles = []
        for layer in range(40):
            contribution = runtime.workspace(hidden.shape, hidden.dtype, 0)
            draft.project_layer(0, layer, record, hidden, cos, sin, contribution)
            output = runtime.read_output(contribution, 0)
            block = output[BLOCK:2 * BLOCK]
            assert torch.isfinite(block).all() and block.abs().max() > 0, f"Invalid layer {layer} output"
            assert not output[:BLOCK].any() and not output[2 * BLOCK:].any()
            role = native("cache").resolve_layer_role(layer)
            roles.append(role["mode"])
            if role["index_source"] is not None and layer in (2, 3, 20, 21, 24, 25):
                rows = draft.selection[0, (0, 0, 1), role["index_source"]][1]
                assert (runtime.read_control(rows["lengths"], 0) == anchor // role["ratio"]).all()
            if layer == 39:
                packed = runtime.workspace((4 * capacity, width), hidden.dtype, 0)
                draft.pack(0, contribution, packed, physical=16, total=2 * BLOCK)
                output = runtime.read_output(packed, 0).view(4, capacity, width)
                assert torch.equal(output[:, :16].reshape(2 * BLOCK, width), runtime.read_output(contribution, 0)[:2 * BLOCK])
                assert not output[:, 16:].any()
            runtime.release_values(draft.finish_layer(layer))
        assert not draft.selection
        for value in pinned:
            assert torch.equal(value.view(torch.uint8), runtime.read_output(value, 0).view(torch.uint8))
        assert before == runtime.stats["device_weight_upload_calls"]
        result = dict(physical_devices=1, rank_shard="TP4/lane0", anchor=anchor, layers=list(range(40)),
                      roles={role: roles.count(role) for role in sorted(set(roles))},
                      block=BLOCK, cache_unchanged=True, weights_reuploaded=False,
                      dispatches=runtime.stats["dispatches"], weight_bytes=runtime.stats["weight_bytes"][0])
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)


if __name__ == "__main__":
    main()
