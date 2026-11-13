#!/usr/bin/env python3
"""Single-card C2A Full/Reuse check using the existing Ascend FIA operator."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch_npu

from blockspec.vllm_ascend.dsv41_attention import (
    DraftSelection,
    PagedRows,
    attend_layer,
    select_history,
)
from blockspec.vllm_ascend.dsv41_contract import BLOCK_SIZE, V41Topology, text_config
from blockspec.vllm_ascend.dsv4_hybrid import (
    merge_attention_segments,
    tail_logical_indices,
)
from blockspec.vllm_ascend.head_contract import read_json
from blockspec.vllm_ascend.torch_npu_dsv4_layer import _native_fia_segment


def paged(rows, page_size, order, device):
    batch, width, dim = rows.shape
    return PagedRows(
        rows.reshape(-1, page_size, dim)[order].to(device),
        order.argsort().view(batch, width // page_size).to(device),
    )


def native_attention(query, temporary, swa, anchors, sink, window, long_kv, indices):
    batch, block, heads, dim = query.shape
    swa_ids = tail_logical_indices(anchors, window)
    # Put the always-present temporary block first, then front-packed SWA history.
    local = torch.cat((temporary, swa.gather(swa_ids)), dim=1).unsqueeze(2)
    local_lengths = (anchors.clamp_max(window) + block).tolist()
    local_out, local_lse = _native_fia_segment(query, local, [block] * batch, local_lengths)

    # Per-query sparse lists have different contents. Flatten into one-query batches.
    sparse = long_kv.gather(indices).reshape(batch * block, -1, 1, dim)
    sparse_lengths = (indices >= 0).sum(-1).flatten()
    sparse_out, sparse_lse = _native_fia_segment(
        query.reshape(batch * block, 1, heads, dim), sparse, [1] * (batch * block),
        sparse_lengths.clamp_min(1).tolist(),
    )
    sparse_lse = sparse_lse.reshape_as(local_lse)
    sparse_lse = sparse_lse.masked_fill(
        (sparse_lengths == 0).view(batch, block, 1, 1), -torch.inf
    )
    sink_lse = sink.reshape(1, 1, heads, 1).expand_as(local_lse)
    return merge_attention_segments(
        (local_out.float(), sparse_out.reshape_as(query).float(), torch.zeros_like(query, dtype=torch.float32)),
        (local_lse, sparse_lse, sink_lse),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp", type=int, default=8, help="Use the head count of one rank. No collectives run.")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    config = read_json(args.config)
    text = text_config(config)
    topology = V41Topology.from_config(config)
    full = next(role for role in topology.layers if role.ratio == 2 and role.mode == "full")
    reuse = next(role for role in topology.layers if role.mode == "reuse" and role.index_source == full.layer)
    assert text["num_attention_heads"] % args.tp == 0
    heads, dim = text["num_attention_heads"] // args.tp, text["head_dim"]
    batch, block, page_size = 3, BLOCK_SIZE, 16
    # Bounded sample with top-k pruning, an empty request and a C2 page boundary.
    topology = replace(topology, topk=16)
    cpu_anchors = torch.tensor([0, 33, 34])
    anchors = cpu_anchors.to(device)
    generator = torch.Generator().manual_seed(41017)

    def sample(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    swa_rows = {layer: sample(batch, 48, dim) for layer in (full.layer, reuse.layer)}
    long_rows = sample(batch, 32, dim)
    index_heads, index_dim = text["index_n_heads"], text["index_head_dim"]
    key_rows = sample(batch, 32, index_dim)
    q_index = sample(batch, block, index_heads, index_dim)
    weights = sample(batch, block, index_heads)
    cpu_sink = sample(heads).float()
    sink = cpu_sink.to(device)
    max_output, max_lse = 0., 0.
    previous = None
    for shuffled in (False, True):
        history_order = torch.randperm(batch * 2, generator=generator) if shuffled else torch.arange(batch * 2)
        cpu_keys = paged(key_rows, page_size, history_order, "cpu")
        cpu_long = paged(long_rows, page_size, history_order, "cpu")
        index = paged(key_rows, page_size, history_order, device)
        selection, state = select_history(
            topology, full.layer, anchors, DraftSelection(),
            query=q_index.to(device), weights=weights.to(device), index_keys={full.layer: index},
        )
        reference, _ = select_history(
            topology, full.layer, cpu_anchors, DraftSelection(),
            query=q_index, weights=weights, index_keys={full.layer: cpu_keys},
        )
        torch.testing.assert_close(selection.cpu(), reference, atol=0, rtol=0)
        if previous is not None:
            torch.testing.assert_close(selection.cpu(), previous, atol=0, rtol=0)
        previous = selection.cpu()
        long_kv = paged(long_rows, page_size, history_order, device)
        for layer in (full.layer, reuse.layer):
            indices = selection if layer == full.layer else select_history(topology, layer, anchors, state)[0]
            assert indices is selection
            order = torch.randperm(batch * 3, generator=generator) if shuffled else torch.arange(batch * 3)
            cpu_swa = paged(swa_rows[layer], page_size, order, "cpu")
            swa = paged(swa_rows[layer], page_size, order, device)
            query = sample(batch, block, heads, dim)
            temporary = sample(batch, block, dim)
            expected = attend_layer(
                topology, layer, query, temporary, cpu_swa, cpu_anchors, cpu_sink,
                global_pages={full.layer: cpu_long}, selection=reference,
            )
            output, lse = native_attention(
                query.to(device), temporary.to(device), swa, anchors, sink, topology.window, long_kv, indices,
            )
            output, lse = output.cpu(), lse.cpu()
            torch.testing.assert_close(output, expected[0], atol=0.015, rtol=0.015)
            torch.testing.assert_close(lse, expected[1], atol=0.01, rtol=0)
            max_output = max(max_output, (output - expected[0]).abs().max().item())
            max_lse = max(max_lse, (lse - expected[1]).abs().max().item())
            assert torch.equal(swa.values.cpu(), cpu_swa.values) and torch.equal(swa.table.cpu(), cpu_swa.table)
        for actual, before in ((long_kv, cpu_long), (index, cpu_keys)):
            assert torch.equal(actual.values.cpu(), before.values) and torch.equal(actual.table.cpu(), before.table)

    print(json.dumps({
        "device": torch.npu.get_device_name(device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "layers": [full.layer, reuse.layer],
        "block_size": block,
        "head_shape": [heads, dim],
        "anchors": cpu_anchors.tolist(),
        "max_output_abs_error": max_output,
        "max_lse_abs_error": max_lse,
        "target_pages_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
