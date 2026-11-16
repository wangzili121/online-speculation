#!/usr/bin/env python3
"""Execute packed-cache GLM Orthrus MLA against a joint-softmax Torch golden."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from blockspec.pypto_glm53_mla import (
    HEADS, HEAD_TILE, LATENT, PAGE, QUERY_ROWS, TILE, VALUE,
    make_history_selector, make_indexed_mla_program, make_mla_program,
)

PAGE_ORDER = (2, 0, 4, 1, 3)


def fixture(anchor, width, seed, selected=None):
    from models.glm5_3_flash.attention import dequant_mla_cache, pack_mla_cache

    generator = torch.Generator().manual_seed(seed)
    def random(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    q_history, q_draft = (random((QUERY_ROWS, HEADS, LATENT), 0.5) for _ in range(2))
    ar_value, draft_value = (random((HEADS, VALUE, LATENT), 0.04) for _ in range(2))
    table = torch.tensor(PAGE_ORDER, dtype=torch.int32)
    cache = pack_mla_cache(random((table.numel() * PAGE, LATENT)))
    block = random((QUERY_ROWS, LATENT))
    indices = torch.full((QUERY_ROWS, width), -1, dtype=torch.int32)
    history_bias = torch.full((QUERY_ROWS, width), -torch.inf)
    if selected is None:
        lengths = torch.tensor([min(anchor, width, count) for count in (0, 1, 127, 129, 511, 2051)],
                               dtype=torch.int32)
        for row, count in enumerate(lengths.tolist()):
            indices[row, :count] = torch.randperm(anchor, generator=generator)[:count].sort().values.int()
            history_bias[row, :count] = 0.0
    else:
        count = selected.shape[1]
        lengths = torch.full((QUERY_ROWS,), count, dtype=torch.int32)
        indices[:, :count] = selected
        history_bias[:, :count] = 0.0
    block_bias = torch.full((1, TILE), -torch.inf)
    block_bias[0, :QUERY_ROWS] = 0.0

    logical = indices.long().clamp_min(0)
    physical = table[logical // PAGE].long() * PAGE + logical % PAGE
    history = dequant_mla_cache(cache)[physical].float()
    qk_history = torch.einsum("qhd,qnd->qhn", q_history.float(), history) * VALUE ** -0.5
    qk_draft = torch.einsum("qhd,nd->qhn", q_draft.float(), block.float()) * VALUE ** -0.5
    joint = torch.cat((qk_history + history_bias[:, None], qk_draft), dim=-1).softmax(-1)
    v_history = torch.einsum("qhn,qnd->qhd", joint[..., :width], history)
    v_block = torch.einsum("qhn,nd->qhd", joint[..., width:], block.float())
    expected = (torch.einsum("qhd,hvd->qhv", v_history, ar_value.float()) +
                torch.einsum("qhd,hvd->qhv", v_block, draft_value.float())).flatten(0, 1)
    q_padding = (0, 0, 0, HEAD_TILE - HEADS)
    tensors = [
        F.pad(q_history, q_padding).flatten(0, 1), F.pad(q_draft, q_padding).flatten(0, 1),
        cache, table, indices, history_bias, lengths,
        F.pad(block, (0, 0, 0, TILE - QUERY_ROWS)), block_bias,
        ar_value.flatten(0, 1), draft_value.flatten(0, 1), torch.empty_like(expected),
    ]
    return tensors, expected


def selection_fixture(anchor, seed, width, query=None, weights=None):
    from models.glm5_3_flash.config import INDEX_DIM, INDEX_H, INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE, KPOOL_SELECT_K

    generator = torch.Generator().manual_seed(seed)
    if query is None:
        query = torch.randn((QUERY_ROWS, INDEX_H, INDEX_DIM), generator=generator).bfloat16()
        weights = torch.randn((QUERY_ROWS, INDEX_H), generator=generator)
    block_size = INDEX_STATE_BLOCK_SIZE
    blocks = torch.tensor(PAGE_ORDER, dtype=torch.int32) * block_size
    cache = torch.randn((blocks.numel() * block_size, INDEX_DIM), generator=generator).bfloat16()
    pools, tail = divmod(anchor, INDEX_KPOOL)
    logical = torch.arange(pools)
    candidates = cache[blocks[logical // block_size].long() + logical % block_size]
    scores = torch.einsum("qhd,nd->qhn", query.float(), candidates.float()).relu()
    pool_ids = (scores * weights[:, :, None]).sum(1).topk(min(pools, KPOOL_SELECT_K), dim=1).indices
    picked = (pool_ids.unsqueeze(-1) * INDEX_KPOOL + torch.arange(INDEX_KPOOL)).flatten(1)
    remainder = torch.arange(pools * INDEX_KPOOL, anchor).expand(QUERY_ROWS, -1)
    expected = torch.cat((picked, remainder), dim=1).sort(1).values.int()
    tensors = [query, weights, cache, blocks, torch.zeros(QUERY_ROWS, dtype=torch.int32)]
    tensors.extend(torch.full((QUERY_ROWS,), item, dtype=torch.int32) for item in (pools, anchor - tail, tail, anchor))
    tensors.append(torch.empty((QUERY_ROWS, width), dtype=torch.int32))
    return tensors, expected


def draft_fixture(anchor, seed, checkpoint_head=None):
    from models.glm5_3_flash.attention import dequant_mla_cache, pack_mla_cache
    from models.glm5_3_flash.config import D, INDEX_DIM, INDEX_H, Q_LORA, TOPK_INDEX_WIDTH
    from models.glm5_3_flash.indexer import WEIGHTS_SCALE
    from models.glm5_3_flash.rmsnorm import rms_norm

    generator = torch.Generator().manual_seed(seed)

    def matrix(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).bfloat16()

    x = matrix((16, D), 0.5)
    if checkpoint_head is None:
        q_a, q_b, kv_a, out = (
            matrix(shape, 0.02) for shape in ((Q_LORA, D), (HEADS * VALUE, Q_LORA), (LATENT, D), (D, HEADS * VALUE))
        )
        q_norm, kv_norm = torch.ones(Q_LORA).bfloat16(), torch.ones(LATENT).bfloat16()
        ar_key, draft_key, ar_value, draft_value = (matrix((HEADS * VALUE, LATENT), 0.04) for _ in range(4))
        indexer_q, indexer_gate = matrix((Q_LORA, INDEX_H * INDEX_DIM), 0.02), matrix((D, INDEX_H), 0.02)
    else:
        q_a, q_norm, q_b, kv_a, kv_norm, draft_key, draft_value, out, indexer_q, indexer_gate = (
            weight.clone() for weight in checkpoint_head
        )
        ar_key, ar_value = draft_key, draft_value
    x_active = x[:QUERY_ROWS].float()
    resid = rms_norm((x_active @ q_a.float().T).bfloat16(), q_norm).float()
    q = (resid @ q_b.float().T).bfloat16().float().reshape(QUERY_ROWS, HEADS, VALUE)
    kv = rms_norm((x_active @ kv_a.float().T).bfloat16(), kv_norm).float()
    qi = (resid @ indexer_q.float()).bfloat16().reshape(QUERY_ROWS, INDEX_H, INDEX_DIM)
    gate = (x_active @ indexer_gate.float()).bfloat16().float() * WEIGHTS_SCALE
    select, picked = selection_fixture(anchor, seed, TOPK_INDEX_WIDTH, qi, gate)
    table = torch.tensor(PAGE_ORDER, dtype=torch.int32)
    cache = pack_mla_cache(matrix((table.numel() * PAGE, LATENT), 1.0))
    physical = table[picked.long() // PAGE].long() * PAGE + picked.long() % PAGE
    past = dequant_mla_cache(cache)[physical].float()
    ar_query = torch.einsum("qhd,hdl->qhl", q, ar_key.float().reshape(HEADS, VALUE, LATENT)).bfloat16().float()
    d_query = torch.einsum("qhd,hdl->qhl", q, draft_key.float().reshape(HEADS, VALUE, LATENT)).bfloat16().float()
    qk_ar = torch.einsum("qhl,qnl->qhn", ar_query, past)
    qk_d = torch.einsum("qhl,nl->qhn", d_query, kv)
    weights = (torch.cat((qk_ar, qk_d), dim=-1) * VALUE ** -0.5).softmax(-1)
    ar_weights, d_weights = weights.split((picked.shape[1], QUERY_ROWS), dim=-1)
    ar_mix = torch.einsum("qhn,qnl->qhl", ar_weights, past)
    d_mix = torch.einsum("qhn,nl->qhl", d_weights, kv)
    mixed = (torch.einsum("qhl,hvl->qhv", ar_mix, ar_value.float().reshape(HEADS, VALUE, LATENT)) +
             torch.einsum("qhl,hvl->qhv", d_mix, draft_value.float().reshape(HEADS, VALUE, LATENT)))
    expected = F.pad(mixed.flatten(1).bfloat16().float() @ out.float().T, (0, 0, 0, 16 - QUERY_ROWS))
    block_bias = torch.full((1, TILE), -torch.inf)
    block_bias[:, :QUERY_ROWS] = 0.0
    inputs = [x, q_a, q_norm, q_b, kv_a, kv_norm, ar_key, draft_key, ar_value, draft_value, out,
              indexer_q, indexer_gate, cache, table, *select[2:-1], block_bias, torch.empty_like(expected)]
    return inputs, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--selection", type=int, default=2176)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--selector", action="store_true", help="validate the read-only target indexer stage")
    modes.add_argument("--indexed", action="store_true", help="validate indexer and MLA in a single dispatch")
    modes.add_argument("--draft", action="store_true", help="validate full rank-local MLA draft attention")
    parser.add_argument("--checkpoint", type=Path, help="initialize draft layer 43 from the real W8A8 checkpoint")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.selection < TILE or args.selection % TILE:
        parser.error("use a new output path and a positive multiple of 128 selection rows")
    if args.checkpoint is not None and not args.draft:
        parser.error("checkpoint validation requires --draft")

    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig

    torch.set_num_threads(4)
    mode = ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=mode,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    if config.execution_mode != mode:
        raise RuntimeError("PyPTO overrode the requested execution mode")
    report = {"platform": config.platform, "passed": False, "compiled": False, "executed": False,
              "mode": next((key for key in ("draft", "indexed", "selector") if getattr(args, key)), "attention"),
              "selection_capacity": args.selection, "cases": []}
    worker, handles = None, []
    try:
        if args.draft:
            from blockspec.pypto_glm53_draft_attention import make_draft_attention_program

            entry = make_draft_attention_program()
            weights = None
            if args.checkpoint is not None:
                from models.glm5_3_flash.utils import Checkpoint
                from blockspec.pypto_glm53_head import head_layout, initialize_rank_head

                packed = initialize_rank_head(Checkpoint(args.checkpoint), 43, 7)
                layout, _ = head_layout()
                weights = tuple(packed[offset:offset + count].view(shape).clone()
                                for offset, count, shape in layout.values())
                report["checkpoint_layer"] = 43
                report["checkpoint_tp_rank"] = 7

            def make_fixture(anchor, seed):
                return draft_fixture(anchor, seed, weights)
        elif args.selector:
            from pypto import language as pl

            select, width = make_history_selector()

            @pl.jit.host
            def entry(q: pl.Tensor, w: pl.Tensor, cache: pl.Tensor, blocks: pl.Tensor, start: pl.Tensor,
                      count: pl.Tensor, tail: pl.Tensor, tail_count: pl.Tensor, prefix: pl.Tensor,
                      indices: pl.Out[pl.Tensor]):
                select(q, w, cache, blocks, start, count, tail, tail_count, prefix, indices, device=0)

            def make_fixture(anchor, seed):
                return selection_fixture(anchor, seed, width)
        elif args.indexed:
            entry, width = make_indexed_mla_program()

            def make_fixture(anchor, seed):
                selector_args, picked = selection_fixture(anchor, seed, width)
                attention_args, expected = fixture(anchor, (width + TILE - 1) // TILE * TILE, seed, picked)
                return attention_args[:4] + selector_args[:-1] + attention_args[7:], expected
        else:
            entry = make_mla_program(args.selection)

            def make_fixture(anchor, seed):
                return fixture(anchor, args.selection, seed)

        sample, _ = make_fixture(0, 174)
        resident = [tensor.share_memory_() for tensor in sample]
        program = entry.compile(*resident, config=config)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        worker = DistributedWorker([program], config=config, inherited_host_tensors=resident)
        for tensor in resident:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        for case, anchor in enumerate((0, 17, 639, 640, 642, 2055)):
            tensors, expected = make_fixture(anchor, 174 + case)
            for src, host, handle in zip(tensors[:-1], resident[:-1], handles[:-1], strict=True):
                host.copy_(src)
                worker.copy_to(handle.data_ptr, host.data_ptr(), host.nbytes)
            worker.run(program, *handles, config=config)
            for host, handle in zip(resident, handles, strict=True):
                worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)
            for actual, original in zip(resident[:-1], tensors[:-1], strict=True):
                if not torch.equal(actual.view(torch.uint8), original.view(torch.uint8)):
                    raise AssertionError("GLM MLA changed a read-only input")
            output = resident[-1]
            if args.selector:
                count = expected.shape[1]
                torch.testing.assert_close(output[:, :count].sort(1).values, expected, rtol=0, atol=0)
                if not bool((output[:, count:] == -1).all()):
                    raise AssertionError("indexer exposed an invalid row outside the committed prefix")
                result = {"anchor": anchor, "selected": count, "inputs_byte_exact": True}
            else:
                error = output - expected
                result = {"anchor": anchor, "max_abs": float(error.abs().max()), "inputs_byte_exact": True}
                if args.draft:
                    # The complete sublayer rounds at several BF16 boundaries.
                    # Compare the residual stream as well as its largest residual
                    # to keep near-zero elements from dominating the error metric.
                    result["relative_l2"] = float(error.norm() / expected.norm())
                    result["max_relative_to_peak"] = float(error.abs().max() / expected.abs().max())
                    if (not torch.isfinite(output).all() or not torch.equal(output[QUERY_ROWS:], expected[QUERY_ROWS:])
                            or result["relative_l2"] > 0.002 or result["max_relative_to_peak"] > 0.004):
                        raise AssertionError(result)
                else:
                    torch.testing.assert_close(output, expected, atol=3e-5, rtol=2e-5)
                if not args.indexed and not args.draft:
                    result["selected_max"] = int(tensors[6].max())
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        report.update(passed=True, executed=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                try:
                    for handle in reversed(handles):
                        worker.free_tensor(handle)
                finally:
                    worker.close()
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
