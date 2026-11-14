#!/usr/bin/env python3
"""Run all V4.1 Orthrus layer roles over synthetic read-only target pages."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

from blockspec.vllm_ascend.dsv41_attention import PagedRows
from blockspec.vllm_ascend.dsv41_contract import (
    BLOCK_SIZE,
    V41Topology,
    text_config,
)
from blockspec.vllm_ascend.dsv41_native_index import IndexPages
from blockspec.vllm_ascend.dsv41_runtime import (
    V41NativeDraftState,
    V41TargetHistory,
    run_native_layer,
)
from blockspec.vllm_ascend.head_contract import read_json


def _paged_rows(
    rows: torch.Tensor,
    page_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> PagedRows:
    batch, width, dim = rows.shape
    if width % page_size:
        raise ValueError("history width must be page aligned")
    pages_per_request = width // page_size
    logical = rows.reshape(-1, page_size, dim)
    order = torch.randperm(logical.shape[0], generator=generator)
    table = order.argsort().view(batch, pages_per_request).int()
    return PagedRows(logical[order].to(device), table.to(device))


def _index_pages(
    batch: int,
    width: int,
    page_size: int,
    dim: int,
    generator: torch.Generator,
    device: torch.device,
) -> IndexPages:
    pages_per_request = width // page_size
    key = torch.randint(
        -127,
        128,
        (batch, width, 1, dim),
        dtype=torch.int8,
        generator=generator,
    ).reshape(-1, page_size, 1, dim)
    scale = torch.full((key.shape[0], page_size, 1), 1 / 127, dtype=torch.float16)
    order = torch.randperm(key.shape[0], generator=generator)
    table = order.argsort().view(batch, pages_per_request).int()
    return IndexPages(key[order].to(device), scale[order].to(device), table.to(device))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators are unavailable")
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    init_device_properties_triton()

    config = read_json(args.config)
    text = text_config(config)
    topology = V41Topology.from_config(config)
    if text["num_attention_heads"] % args.tp:
        raise ValueError("query heads are not divisible by TP")
    heads = text["num_attention_heads"] // args.tp
    head_dim = text["head_dim"]
    index_heads = text["index_n_heads"]
    index_dim = text["index_head_dim"]
    batch, page_size, history = 2, 16, 544
    anchors = torch.tensor([0, history], dtype=torch.int32, device=device)
    generator = torch.Generator().manual_seed(41041)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.bfloat16).to(device)

    swa = _paged_rows(
        torch.randn(batch, history, head_dim, generator=generator, dtype=torch.bfloat16),
        page_size,
        generator,
        device,
    )
    global_rows = {
        ratio: _paged_rows(
            torch.randn(
                batch,
                history // ratio,
                head_dim,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            page_size,
            generator,
            device,
        )
        for ratio in (1, 2)
    }
    index_rows = {
        ratio: _index_pages(
            batch,
            history // ratio,
            page_size,
            index_dim,
            generator,
            device,
        )
        for ratio in (1, 2)
    }
    protected = [
        (swa.values, swa.values.clone()),
        (swa.table, swa.table.clone()),
    ]
    for pages in global_rows.values():
        protected.extend(((pages.values, pages.values.clone()), (pages.table, pages.table.clone())))
    for pages in index_rows.values():
        protected.extend(
            (
                (pages.key, pages.key.clone()),
                (pages.scale, pages.scale.clone()),
                (pages.block_table, pages.block_table.clone()),
            )
        )

    def run_round() -> tuple[V41NativeDraftState, list[float]]:
        state = V41NativeDraftState()
        layer_times = []
        for layer, role in enumerate(topology.layers):
            history_view = V41TargetHistory(
                swa=swa,
                anchors=anchors,
                long_kv=global_rows.get(role.ratio),
                index_k=index_rows.get(role.ratio),
                global_lengths=(anchors // role.ratio if role.ratio else None),
                max_swa_history=history,
                max_global_history=(history // role.ratio if role.ratio else None),
            )
            index_query = index_weights = None
            if role.layer == role.index_source:
                index_query = randn(batch, BLOCK_SIZE, index_heads, index_dim)
                index_weights = randn(batch, BLOCK_SIZE, index_heads)
            torch.npu.synchronize()
            start = time.perf_counter()
            output, lse, state = run_native_layer(
                topology,
                layer,
                randn(batch, BLOCK_SIZE, heads, head_dim),
                randn(batch, BLOCK_SIZE, head_dim),
                torch.zeros(heads, dtype=torch.float32, device=device),
                history_view,
                state,
                index_query=index_query,
                index_weights=index_weights,
            )
            torch.npu.synchronize()
            layer_times.append((time.perf_counter() - start) * 1000)
            if output.shape != (batch, BLOCK_SIZE, heads, head_dim):
                raise RuntimeError(f"layer {layer} returned an invalid output shape")
            if not torch.isfinite(output).all() or not torch.isfinite(lse).all():
                raise RuntimeError(f"layer {layer} produced a non-finite result")
        return state, layer_times

    cold_state, cold_times = run_round()
    state, steady_times = run_round()

    expected_index_sources = {
        role.index_source for role in topology.layers if role.index_source is not None
    }
    for round_state in (cold_state, state):
        if set(round_state.topk) != expected_index_sources or round_state.candidates is None:
            raise RuntimeError("the 40-layer QLI state handoff is incomplete")
    if any(not torch.equal(value, before) for value, before in protected):
        raise RuntimeError("the draft runtime modified target-owned pages")

    print(
        json.dumps(
            {
                "status": "ok",
                "device": torch_npu.npu.get_device_name(device),
                "layers": len(topology.layers),
                "layer_modes": dict(Counter(role.mode for role in topology.layers)),
                "index_sources": sorted(state.topk),
                "candidate_pool_shape": list(state.candidates.shape),
                "block_size": BLOCK_SIZE,
                "batch": batch,
                "anchors": anchors.cpu().tolist(),
                "target_pages_unchanged": True,
                "cold_elapsed_ms": sum(cold_times),
                "steady_elapsed_ms": sum(steady_times),
                "steady_maximum_layer_ms": max(steady_times),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
