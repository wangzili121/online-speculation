#!/usr/bin/env python3
"""Exercise the draft MLA inside its mHC and TP communication boundary."""

import argparse
import json
import sys
from pathlib import Path


def make_program(capacity, table_width, ranks):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D
    from blockspec.pypto_glm53_backbone import make_backbone_mla

    draft = pl.jit(auto_scope=False)(make_backbone_mla(capacity, table_width)._func)

    @pl.jit.host
    def entry(
        stream: pl.Tensor, heads: pl.Tensor, ar_key: pl.Tensor, ar_value: pl.Tensor,
        mapping: pl.Tensor, scale: pl.Tensor, base: pl.Tensor, norm: pl.Tensor,
        history: pl.Tensor, tables: pl.Tensor, pool: pl.Tensor, blocks: pl.Tensor,
        start: pl.Tensor, count: pl.Tensor, tail: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, ids: pl.Tensor,
        pre: pl.InOut[pl.Tensor], post: pl.InOut[pl.Tensor], residual: pl.InOut[pl.Tensor],
        collapsed: pl.InOut[pl.Tensor], normed: pl.InOut[pl.Tensor],
        partial: pl.InOut[pl.Tensor], reduced: pl.InOut[pl.Tensor], result: pl.Out[pl.Tensor],
    ):
        exchange = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        ready = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            exchange_view = pld.window(exchange, [capacity, D], dtype=pl.FP32)
            ready_view = pld.window(ready, [ranks, 1], dtype=pl.INT32)
            draft(stream[rank], heads[rank], ar_key[rank], ar_value[rank], mapping[rank],
                  scale[rank], base[rank], norm[rank], history[rank], tables[rank],
                  pool[rank], blocks[rank], start[rank], count[rank], tail[rank],
                  tail_count[rank], prefix[rank], ids[rank], pre[rank], post[rank],
                  residual[rank], collapsed[rank], normed[rank], partial[rank], reduced[rank],
                  result[rank], exchange_view, ready_view, rank, 1, 1, device=rank)

    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = len(args.devices)
    if ranks not in (1, 2, 4, 8) or len(set(args.devices)) != ranks:
        parser.error("use one, two, four or eight distinct devices")
    if args.capacity < 16 or args.capacity % 16 or args.output.exists():
        parser.error("use a fresh output and a capacity divisible by 16")
    # Keep the real TP8 MLA shard geometry, while testing communication with
    # the available number of ranks. This is a component-level integration test.
    sys.argv += ["--tp", str(ranks), "--ep", str(ranks)]

    import torch
    from models.glm5_3_flash.config import D, HC_MULT, HC_DIM, MIX_HC
    from models.glm5_3_flash.mhc import hc_post
    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig
    from blockspec.pypto_glm53_head import head_layout, initialize_rank_head
    from validate_glm53_pypto_mla import draft_fixture

    torch.set_num_threads(4)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
        device_ids=args.devices, num_sub_workers=0))
    layout, _ = head_layout()
    checkpoint = Checkpoint(args.checkpoint)
    packed = [initialize_rank_head(checkpoint, 3, rank) for rank in range(ranks)]
    fixtures = []
    for rank, slab in enumerate(packed):
        weights = tuple(slab[offset:offset + size].view(shape).clone()
                        for offset, size, shape in layout.values())
        fixture, _ = draft_fixture(638, 174, weights)
        fixtures.append(fixture)
    c = args.capacity
    source = fixtures[0][0]
    stream = torch.zeros((ranks, c, HC_MULT, D), dtype=torch.bfloat16)
    stream[:, :16] = source[None, :, None, :] / 2
    stacked = lambda index: torch.stack([f[index] for f in fixtures])
    # Poison the neighboring head and page row so the test exercises nonzero
    # layer/request offsets as well as the individual projection offsets.
    head_bank = torch.full((ranks, 2, packed[0].numel()), float("nan"), dtype=torch.bfloat16)
    head_bank[:, 1] = torch.stack(packed)
    page_tables = torch.full((ranks, 2, fixtures[0][14].numel()), -1, dtype=torch.int32)
    page_tables[:, 1] = stacked(14)
    tensors = [stream, head_bank, stacked(6), stacked(8),
               torch.zeros((ranks, MIX_HC, HC_DIM), dtype=torch.bfloat16),
               torch.zeros((ranks, 3)), torch.zeros((ranks, MIX_HC)),
               torch.ones((ranks, D), dtype=torch.bfloat16), stacked(13), page_tables,
               stacked(15), stacked(16), *[stacked(i) for i in range(17, 22)],
               torch.ones((ranks, c), dtype=torch.int32),
               torch.empty((ranks, c, HC_MULT)), torch.empty((ranks, c, HC_MULT)),
               torch.empty((ranks, c, HC_MULT, HC_MULT)),
               torch.empty((ranks, c, D), dtype=torch.bfloat16),
               torch.empty((ranks, c, D), dtype=torch.bfloat16),
               torch.empty((ranks, c, D)), torch.empty((ranks, c, D), dtype=torch.bfloat16),
               torch.empty_like(stream)]
    resident = [t.share_memory_() for t in tensors]
    originals = [t.clone() for t in resident[:18]]
    report = {"passed": False, "compiled": False, "executed": False,
              "devices": args.devices, "capacity": c, "tp8_weight_shards": list(range(ranks))}
    try:
        print("GLM_BACKBONE_COMPILE", flush=True)
        program = make_program(c, tensors[9].shape[-1], ranks).compile(*resident, config=config)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        with DistributedWorker([program], config=config, persistent=True) as worker:
            handles = [worker.alloc_stacked_tensor(t) for t in resident]
            print("GLM_BACKBONE_EXECUTE", flush=True)
            worker.run(program, *handles, config=config)
            report["executed"] = True
            for host, handle in zip(resident, handles, strict=True):
                for rank, shard in enumerate(handle.shards):
                    worker.copy_from(host[rank].data_ptr(), shard.data_ptr, host[rank].nbytes,
                                     worker_id=rank)
        for actual, original in zip(resident[:18], originals, strict=True):
            if not torch.equal(actual.view(torch.uint8), original.view(torch.uint8)):
                raise AssertionError("the draft changed an input")
        partial = resident[-3][:, :6].double().sum(0).bfloat16()
        for rank in range(ranks):
            torch.testing.assert_close(resident[-2][rank, :6], partial, rtol=0, atol=0)
            if resident[-2][rank, 6:].count_nonzero():
                raise AssertionError("the draft did not clear padded reduction rows")
            expected = hc_post(resident[-2][rank], stream[rank], resident[19][rank], resident[20][rank])
            torch.testing.assert_close(resident[-1][rank], expected, atol=0.015625, rtol=0.01)
            if not torch.isfinite(resident[-1][rank]).all():
                raise AssertionError("the draft produced nonfinite residuals")
        report.update(passed=True, tp_reduction_exact=True, padding_zero=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
