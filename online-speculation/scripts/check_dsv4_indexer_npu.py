#!/usr/bin/env python3
"""Check the K32 history boundary using the installed Ascend QLI operator."""

import argparse
import json

import torch
import torch_npu  # noqa: F401

# Import the runtime adapter to register the installed vLLM Ascend operators.
from blockspec.vllm_ascend import dsv4  # noqa: F401
from vllm_ascend.utils import enable_custom_op


def run(mode, length):
    query = torch.ones((32, 64, 128), dtype=torch.int8, device="npu")
    blocks = max((length // 4 + 127) // 128, 1)
    key = torch.ones((blocks, 128, 1, 128), dtype=torch.int8, device="npu")
    qscale = torch.full((32, 64), 0.01, dtype=torch.float16, device="npu")
    kscale = torch.full((blocks, 128, 1), 0.01, dtype=torch.float16, device="npu")
    if length // 4 > 512:
        # Monotonic, exactly representable scales give an unambiguous Top-512.
        kscale = (torch.arange(1, blocks * 128 + 1, device="npu").float()
                  / 1024).half().view(blocks, 128, 1)
    weights = torch.ones_like(qscale)
    qlens = torch.tensor([32], dtype=torch.int32, device="npu")
    klens = torch.tensor([length], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    common = dict(
        actual_seq_lengths_query=qlens, actual_seq_lengths_key=klens,
        query_quant_mode=0, key_quant_mode=0, layout_query="TND",
        layout_key="PA_BSND", sparse_count=512, sparse_mode=mode,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1, cmp_ratio=4,
    )
    metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        **common, num_heads_q=64, num_heads_k=1, head_dim=128,
        batch_size=1, max_seqlen_q=32, max_seqlen_k=length, device="npu:0",
    )
    indices, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query, key=key, weights=weights, query_dequant_scale=qscale,
        key_dequant_scale=kscale, block_table=table, metadata=metadata,
        return_value=False, **common,
    )
    rows = indices.cpu().view(32, -1)
    counts = (rows >= 0).sum(-1).tolist()
    result = {"mode": mode, "history_length": length, "valid_counts": counts,
              "first_row": rows[0, :min(length // 4, 40)].tolist(),
              "last_row": rows[-1, :min(length // 4, 40)].tolist()}
    if mode == 0:
        assert counts == [min(length // 4, 512)] * 32, result
        expected = list(range(max(0, length // 4 - 512), length // 4))
        assert all(sorted(row[row >= 0].tolist()) == expected for row in rows), result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 4, 31, 32, 127, 128, 129, 2049, 4096, 4097])
    args = parser.parse_args()
    assert enable_custom_op(), "Ascend custom operator package is unavailable"
    torch.npu.set_device(0)
    for length in args.lengths:
        for mode in (3, 0):
            print(json.dumps(run(mode, length)), flush=True)


if __name__ == "__main__":
    main()
