#!/usr/bin/env python3
"""Compare sliced native FIA against dense attention across paged KV boundaries."""

import json
from types import SimpleNamespace

import torch
import torch_npu

from blockspec.vllm_ascend.qwen38_attention import verification_chunks


def main():
    torch.manual_seed(37)
    heads, kv_heads, dim, page_size = 12, 2, 256, 128
    causal_mask = torch.ones(2048, 2048, dtype=torch.bool, device="npu").triu(1)
    results = []
    for past, length in ((0, 32), (127, 17), (129, 32), (255, 32)):
        total = past + length
        query = torch.randn(length, heads, dim).bfloat16()
        key, value = [torch.randn(total, kv_heads, dim).bfloat16() for _ in range(2)]
        pages = (total + page_size - 1) // page_size
        table = torch.randperm(pages)
        cache = torch.zeros(2, pages, page_size, kv_heads, dim, dtype=torch.bfloat16)
        for index in range(total):
            page, offset = table[index // page_size], index % page_size
            cache[0, page, offset], cache[1, page, offset] = key[index], value[index]
        npu_cache, npu_query = cache.npu(), query.npu()
        block_table = table[None].int().npu()
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[length], seq_lens_list=[total], causal=True,
            seq_lens=torch.tensor([total], dtype=torch.int32, device="npu"),
            seq_lens_cpu=torch.tensor([total], dtype=torch.int32),
            query_start_loc=torch.tensor([0, length], dtype=torch.int32, device="npu"))
        outputs = []
        # 与原生 verifier 相同的 FIA 调用；只切 query，可见 KV 的末端逐段推进。
        for rows, part in verification_chunks(metadata):
            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=npu_query[rows].contiguous(),
                key=npu_cache[0].view(pages, page_size, -1).contiguous(),
                value=npu_cache[1].view(pages, page_size, -1).contiguous(),
                atten_mask=causal_mask, block_table=block_table, input_layout="TND",
                block_size=page_size, actual_seq_lengths=part.actual_seq_lengths_q,
                actual_seq_lengths_kv=part.seq_lens_list, num_heads=heads,
                num_key_value_heads=kv_heads, scale=dim ** -.5, sparse_mode=3)
            outputs.append(output)
        actual = torch.cat(outputs).float().cpu()
        allowed = torch.arange(total)[None] <= past + torch.arange(length)[:, None]
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.float().transpose(0, 1)[None],
            key.float().transpose(0, 1)[None].repeat_interleave(heads // kv_heads, dim=1),
            value.float().transpose(0, 1)[None].repeat_interleave(heads // kv_heads, dim=1),
            attn_mask=allowed)[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=.025, atol=.005)
        torch.testing.assert_close(npu_cache.cpu(), cache, rtol=0, atol=0)
        results.append(dict(past=past, queries=length, pages=pages,
                            output_max_abs=float((actual - expected).abs().max()), cache_unchanged=True))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
