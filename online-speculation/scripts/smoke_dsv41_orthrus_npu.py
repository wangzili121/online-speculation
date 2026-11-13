#!/usr/bin/env python3
"""Check V4.1 draft attention against the CPU reference on a single Ascend NPU."""

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
from blockspec.vllm_ascend.dsv41_precise_attention import precise_draft_attention
from blockspec.vllm_ascend.dsv41_native_attention import native_draft_attention
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


def segmented_fia(query, temporary, swa, anchors, sink, window, long_kv, indices):
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


def check_precision(actual, expected):
    output, lse = (part.cpu() for part in actual)
    torch.testing.assert_close(output, expected[0], atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(lse, expected[1], atol=1e-5, rtol=1.3e-6)
    return {
        "elements": output.numel(),
        "max_output_abs_error": (output - expected[0]).abs().max().item(),
        "max_lse_abs_error": (lse - expected[1]).abs().max().item(),
    }


def check_bf16_precision(actual, expected):
    output, lse = (part.cpu() for part in actual)
    torch.testing.assert_close(output, expected[0], atol=1e-2, rtol=2.01 / 128)
    torch.testing.assert_close(lse, expected[1], atol=1e-2, rtol=2.01 / 128)
    return {
        "elements": output.numel(),
        "max_output_abs_error": (output - expected[0]).abs().max().item(),
        "max_lse_abs_error": (lse - expected[1]).abs().max().item(),
    }


def check_long_history(topology, heads, dim, sample, device):
    """Exercise full SWA and Top-K 512 against several pages and partial batches."""
    cases = []
    anchors = torch.tensor([0, 4097, 8194])
    batch, page_size = anchors.numel(), 64
    # Use the same physical storage for logical C1A or C2A rows in this test.
    rows = sample(batch, 8256, dim)
    order = torch.arange(batch * (rows.shape[1] // page_size)).flip(0)
    host_pages = paged(rows, page_size, order, "cpu")
    device_pages = PagedRows(host_pages.values.to(device), host_pages.table.to(device))
    for ratio, block in ((0, 1), (0, BLOCK_SIZE), (1, 5), (1, BLOCK_SIZE), (2, BLOCK_SIZE)):
        role = next(role for role in topology.layers if role.ratio == ratio)
        query = sample(batch, block, heads, dim)
        temporary = sample(batch, block, dim)
        sink = torch.linspace(-8, 8, heads)
        host_global = device_global = selection = None
        if ratio:
            offsets = torch.arange(topology.topk)
            positions = torch.arange(block)[:, None] * 7 + offsets * 3
            lengths = (anchors // ratio)[:, None, None]
            selection = torch.where(positions < lengths, positions, -1)
            host_global, device_global = host_pages, device_pages
        expected = attend_layer(
            topology, role.layer, query, temporary, host_pages, anchors, sink,
            global_pages={role.kv_source: host_global} if ratio else {}, selection=selection,
        )
        actual = precise_draft_attention(
            query.to(device), temporary.to(device), device_pages, anchors.to(device), sink.to(device),
            window=topology.window, global_kv=device_global,
            global_indices=selection.to(device) if selection is not None else None,
        )
        cases.append({"ratio": ratio, "block_size": block, **check_precision(actual, expected)})
    torch.testing.assert_close(device_pages.values.cpu(), host_pages.values, atol=0, rtol=0)
    torch.testing.assert_close(device_pages.table.cpu(), host_pages.table, atol=0, rtol=0)
    return cases


def check_projected_history(
    topology, heads, text, sample, device, layer_weights, compare_native
):
    """Use real layer-2 projections and synthetic hidden states for the FP32 check."""
    weights = torch.load(layer_weights, map_location="cpu", weights_only=True)
    eps = text["rms_norm_eps"]
    dim = text["head_dim"]
    block = BLOCK_SIZE
    anchors = torch.tensor([1023, 1026])
    batch = anchors.numel()
    history, draft = sample(batch, 1088, text["hidden_size"]), sample(batch, block, text["hidden_size"])

    def rms(value, weight):
        return torch.nn.functional.rms_norm(value.float(), (value.shape[-1],), weights[weight].float(), eps)

    def project(value, weight):
        operand = weights[weight].float()
        if weights[weight].dtype == torch.int8:
            operand.mul_(weights[weight + "_scale"])
        return value.float() @ operand

    qr = rms(project(draft, "wq_a"), "q_norm")
    rank_width = heads * dim
    q = (qr @ (weights["wq_b"][:, :rank_width].float() * weights["wq_b_scale"][:rank_width]))
    q = q.view(batch, block, heads, dim).bfloat16()
    temp = rms(project(draft, "wkv"), "kv_norm").bfloat16()
    swa = rms(project(history, "wkv"), "kv_norm").bfloat16()
    # Each pair produces one latent. The selected rows below exclude the incomplete pair.
    latents = project(history, "compressor_wkv").view(batch, -1, 2, dim)
    gates = project(history, "compressor_wgate").view_as(latents).softmax(dim=2)
    global_rows = rms((latents * gates).sum(dim=2), "compressor_norm").bfloat16()

    def reverse_pages(rows):
        return paged(rows, 32, torch.arange(batch * (rows.shape[1] // 32)).flip(0), "cpu")

    def to_device(pages):
        return PagedRows(pages.values.to(device), pages.table.to(device))

    swa, compressed = reverse_pages(swa), reverse_pages(global_rows)
    positions = torch.arange(topology.topk)
    selected = torch.where(positions < (anchors // 2)[:, None, None], positions, -1)
    selected = selected.expand(-1, block, -1).contiguous()
    sink = weights["attn_sink"][:heads]
    expected = attend_layer(
        topology, 2, q, temp, swa, anchors, sink, global_pages={2: compressed}, selection=selected,
    )
    device_swa, device_compressed = to_device(swa), to_device(compressed)
    precise = precise_draft_attention(
        q.to(device), temp.to(device), device_swa, anchors.to(device), sink.to(device),
        window=topology.window, global_kv=device_compressed, global_indices=selected.to(device),
    )
    for host, resident in ((swa, device_swa), (compressed, device_compressed)):
        torch.testing.assert_close(resident.values.cpu(), host.values, atol=0, rtol=0)
        torch.testing.assert_close(resident.table.cpu(), host.table, atol=0, rtol=0)
    result = {"precise": check_precision(precise, expected)}
    if compare_native:
        native = native_draft_attention(
            q.to(device),
            temp.to(device),
            device_swa,
            anchors.to(device),
            sink.to(device),
            window=topology.window,
            global_kv=device_compressed,
            global_indices=selected.to(device),
            global_lengths=(anchors // 2).to(device),
            max_swa_history=int(anchors.max()),
            max_global_history=int((anchors // 2).max()),
        )
        result["native"] = check_bf16_precision(native, expected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp", type=int, default=8, help="Use the head count of one rank. No collectives run.")
    parser.add_argument("--compare-fia", action="store_true", help="Compare against the prior segmented BF16 path.")
    parser.add_argument("--compare-native", action="store_true", help="Run native sparse planes plus block FIA.")
    parser.add_argument("--layer2-weights", type=Path, help="Optional A2 W8A8 layer-2 projection sample.")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    if args.compare_native:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    config = read_json(args.config)
    text = text_config(config)
    topology = V41Topology.from_config(config)
    full = next(role for role in topology.layers if role.ratio == 2 and role.mode == "full")
    reuse = next(role for role in topology.layers if role.mode == "reuse" and role.index_source == full.layer)
    assert text["num_attention_heads"] % args.tp == 0
    heads, dim = text["num_attention_heads"] // args.tp, text["head_dim"]
    batch, block, page_size = 3, BLOCK_SIZE, 16
    # Bounded sample with top-k pruning, an empty request and a C2 page boundary.
    small = replace(topology, topk=16)
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
    fia_max_output, fia_excess = 0., 0
    previous = None
    for shuffled in (False, True):
        history_order = torch.randperm(batch * 2, generator=generator) if shuffled else torch.arange(batch * 2)
        cpu_keys = paged(key_rows, page_size, history_order, "cpu")
        cpu_long = paged(long_rows, page_size, history_order, "cpu")
        index = paged(key_rows, page_size, history_order, device)
        selection, state = select_history(
            small, full.layer, anchors, DraftSelection(),
            query=q_index.to(device), weights=weights.to(device), index_keys={full.layer: index},
        )
        reference, _ = select_history(
            small, full.layer, cpu_anchors, DraftSelection(),
            query=q_index, weights=weights, index_keys={full.layer: cpu_keys},
        )
        torch.testing.assert_close(selection.cpu(), reference, atol=0, rtol=0)
        if previous is not None:
            torch.testing.assert_close(selection.cpu(), previous, atol=0, rtol=0)
        previous = selection.cpu()
        long_kv = paged(long_rows, page_size, history_order, device)
        for layer in (full.layer, reuse.layer):
            indices = selection if layer == full.layer else select_history(small, layer, anchors, state)[0]
            assert indices is selection
            order = torch.randperm(batch * 3, generator=generator) if shuffled else torch.arange(batch * 3)
            cpu_swa = paged(swa_rows[layer], page_size, order, "cpu")
            swa = paged(swa_rows[layer], page_size, order, device)
            query = sample(batch, block, heads, dim)
            temporary = sample(batch, block, dim)
            expected = attend_layer(
                small, layer, query, temporary, cpu_swa, cpu_anchors, cpu_sink,
                global_pages={full.layer: cpu_long}, selection=reference,
            )
            device_q, device_temp = query.to(device), temporary.to(device)
            actual = precise_draft_attention(
                device_q, device_temp, swa, anchors, sink, window=small.window,
                global_kv=long_kv, global_indices=indices,
            )
            stats = check_precision(actual, expected)
            max_output = max(max_output, stats["max_output_abs_error"])
            max_lse = max(max_lse, stats["max_lse_abs_error"])
            if args.compare_fia:
                fia, _ = segmented_fia(device_q, device_temp, swa, anchors, sink, small.window, long_kv, indices)
                error = (fia.cpu() - expected[0]).abs()
                fia_max_output = max(fia_max_output, error.max().item())
                fia_excess += (error > (1e-3 + (2.01 / 128) * expected[0].abs())).sum().item()
            assert torch.equal(swa.values.cpu(), cpu_swa.values) and torch.equal(swa.table.cpu(), cpu_swa.table)
        for actual, before in ((long_kv, cpu_long), (index, cpu_keys)):
            assert torch.equal(actual.values.cpu(), before.values) and torch.equal(actual.table.cpu(), before.table)

    result = {
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
        "long_history": check_long_history(topology, heads, dim, sample, device),
    }
    if args.compare_fia:
        result["segmented_bf16_fia"] = {"max_output_abs_error": fia_max_output, "excess_elements": fia_excess}
    if args.layer2_weights is not None:
        result["layer2_projection_sample"] = check_projected_history(
            topology, heads, text, sample, device, args.layer2_weights, args.compare_native,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
