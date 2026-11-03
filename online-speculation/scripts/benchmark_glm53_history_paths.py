#!/usr/bin/env python3
"""分别计时分页快照、KPool 打分/Top-K 与选中 latent 的 gather。"""

import argparse
from functools import partial
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from blockspec.vllm_ascend.glm53_native import read_pages
from blockspec.vllm_ascend.glm53_ops import gather_history, select_history
from benchmark_glm53_attention_paths import measure


def page_loop(cache, table, length, *, block_size):
    """逐页 clone 的基线；避免对非连续 arena 直接 index_select。"""
    split = cache.shape[1] // block_size
    pieces = []
    for page, index in enumerate(table[0, :(length + block_size - 1) // block_size].cpu().tolist()):
        physical, subpage = divmod(index, split)
        offset = subpage * block_size
        width = min(block_size, length - page * block_size)
        pieces.append(cache[physical, offset:offset + width, 0].clone(memory_format=torch.contiguous_format))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.npu.set_device(0)
    torch.manual_seed(5311)
    results = []
    for dim, pitch in ((128, 128), (512, 576)):
        arena = torch.randn(512, 128, 1, pitch, device="npu", dtype=torch.bfloat16)
        cache = arena[..., :dim]
        table = torch.randperm(4096).to(device="npu", dtype=torch.int32)[None]
        for length in (7, 65, 129, 257, 2055, 8195, 32771):
            old = partial(page_loop, cache, table, length, block_size=16)
            new = partial(read_pages, cache, table, length, block_size=16)
            assert torch.equal(new(), old())
            results.append(dict(case="snapshot", length=length, dim=dim, pitch=pitch,
                                page_loop=measure(old, args.iterations),
                                current=measure(new, args.iterations), exact=True))
    for length in (7, 2055, 8195, 32771):
        history = torch.randn(length, 512, device="npu", dtype=torch.bfloat16)
        pools = torch.randn(length // 4, 128, device="npu", dtype=torch.bfloat16)
        query = torch.randn(8, 32, 128, device="npu", dtype=torch.bfloat16)
        weights = torch.randn(8, 32, device="npu", dtype=torch.bfloat16)
        def score(query=query, pools=pools, weights=weights):
            scores = torch.einsum("qhd,pd->qhp", query.float(), pools.float())
            return (scores.relu() * weights.float().unsqueeze(-1)).sum(1)
        scores = score()
        selected = select_history(query, weights, pools, length, topk=2048)
        results.append(dict(case="retrieval", length=length,
                            score=measure(score, args.iterations),
                            topk=measure(partial(scores.topk, min(len(pools), 512), dim=-1), args.iterations),
                            select=measure(partial(select_history, query, weights, pools, length, topk=2048), args.iterations),
                            gather_advanced=measure(lambda history=history, selected=selected: history[selected], args.iterations),
                            gather_current=measure(partial(gather_history, history, selected), args.iterations)))
    report = dict(scope="synthetic local operators, not model or service throughput", results=results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
