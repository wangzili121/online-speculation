#!/usr/bin/env python3
"""Exercise the complete resident MLA replay and update on real GLM weight shards."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

import torch

from blockspec.pypto_glm53_head import head_layout, initialize_rank_head
from blockspec.pypto_glm53_mla import HEADS, LATENT, QUERY_ROWS, VALUE
from blockspec.pypto_glm53_online_mla import ResidentMLAUpdate, prepare_mla_update

from validate_glm53_pypto_mla import fixture, selection_fixture


def validate(args, report):
    from models.glm5_3_flash.config import D, FLASH, TOPK_INDEX_WIDTH
    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, StackedDeviceTensor

    class Compiler:
        def compile(self, _name, stage):
            return stage.compile(config=config)

    ranks = len(args.devices)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
        device_ids=args.devices, num_sub_workers=0,
    ))
    torch.set_num_threads(4)
    indexer, picked = selection_fixture(642, 91, TOPK_INDEX_WIDTH)
    attention, _ = fixture(642, 2176, 91, picked)
    cache, table = attention[2:4]
    pool, pool_blocks = indexer[2:4]
    api = SimpleNamespace(RANKS=ranks, D=D, TABLE_MAX=len(table), POOL_PAGE_ROWS=len(pool_blocks),
                          LATENT_ROWS=cache.shape[0], MLA_CACHE_WIDTH=cache.shape[1], POOL_TABLE_ROWS=pool.shape[0])
    programs = prepare_mla_update(api, Compiler())
    report["programs_compiled"] = list(programs)
    checkpoint = Checkpoint(args.checkpoint)
    layer, ordinal = FLASH.dsa_layer_ids[-1], len(FLASH.dsa_layer_ids) - 1
    layout, total = head_layout()
    packed = torch.stack([initialize_rank_head(checkpoint, layer, rank) for rank in range(ranks)])
    norm = checkpoint.read(checkpoint.layer_key(layer, "input_layernorm.weight")).bfloat16()
    generator = torch.Generator().manual_seed(57)
    collapsed = (torch.randn(16, D, generator=generator) * 0.5).bfloat16()
    upstream = torch.randn(16, D, generator=generator) * 0.01
    upstream[QUERY_ROWS - 1:] = 0
    meta = dict(zip(("seg_start", "pool_count_tok", "tail_start", "tail_count", "kv_len_tok"), indexer[4:-1], strict=True))
    meta.update(block_table=table.unsqueeze(0), pool_blocks=pool_blocks)

    with DistributedWorker(list(programs.values()), config=config, persistent=True, reset_persistent_windows=False) as worker:
        def alloc(shape, dtype):
            handles = [worker.alloc_tensor(shape, dtype, worker_id=i) for i in range(ranks)]
            return StackedDeviceTensor(handles, (ranks, *shape), tuple(range(ranks)))

        def upload(handle, values, *, offset=0):
            for rank, (dest, source) in enumerate(zip(handle.shards, values, strict=True)):
                host = source.contiguous().share_memory_()
                worker.copy_to(dest.data_ptr, host.data_ptr(), host.nbytes, dst_offset=offset, worker_id=rank)

        def replicated(value):
            dev = alloc(value.shape, value.dtype)
            upload(dev, [value] * ranks)
            return dev

        def download(handle, rank):
            shard = handle.shards[rank]
            out = torch.empty(shard.shape, dtype=shard.dtype).share_memory_()
            worker.copy_from(out.data_ptr(), shard.data_ptr - shard.byte_offset, out.nbytes,
                             src_offset=shard.byte_offset, worker_id=rank)
            return out

        live = alloc((ordinal + 1, total), torch.bfloat16)
        upload(live, packed, offset=ordinal * total * packed.element_size())
        runner = SimpleNamespace(_api=api, auxiliary_programs=programs, config=config,
                                 stacked={"orthrus_head": live}, _run_l3=worker.run)
        session = SimpleNamespace(runner=runner, runtime=worker, allocate=alloc)
        keys = {}
        for name in ("key", "value"):
            start, length, shape = layout[name]
            view = packed[:, start:start + length].reshape(ranks, HEADS, VALUE, LATENT)
            handle = alloc(tuple(view.shape[1:]), view.dtype)
            upload(handle, view)
            keys[f"mla_w_{name[0]}"] = handle
        weights = dict(keys, mla_latent_cache=replicated(cache), index_pool_cache=replicated(pool),
                       norm_input_norm=replicated(norm))
        suffix = SimpleNamespace(collapse=replicated(collapsed), _weight=lambda name, _ordinal: weights[name])
        updater = ResidentMLAUpdate(session, suffix, 16, 1e-4, 0.01)
        upstream = replicated(upstream)
        report["steps"] = []
        earlier = packed
        trainable = updater.gradient.shape[-1]
        for _ in range(2):
            updater.step(upstream, meta, slot=0)
            after = torch.cat([download(updater.live, rank) for rank in range(ranks)], dim=0)
            if torch.equal(after[..., :trainable], earlier[..., :trainable]):
                raise AssertionError("resident update left the live MLA weights unchanged")
            torch.testing.assert_close(after[..., trainable:], packed[..., trainable:], atol=0, rtol=0)
            gradients = [download(updater.gradient, rank).flatten() for rank in range(ranks)]
            for name, (start, count, _shape) in layout.items():
                if start < trainable and any(not g[start:start + count].count_nonzero() for g in gradients):
                    raise AssertionError(f"missing native gradient for draft projection {name}")
            report["steps"].append({"step": updater.step_count,
                                    "largest_gradient": max(float(g.abs().max()) for g in gradients)})
            earlier = after
        previous_context = [download(updater.context, rank) for rank in range(ranks)]
        updater.replay(meta, slot=0)
        for rank, previous in enumerate(previous_context):
            current = download(updater.context, rank)
            if not torch.isfinite(current).all() or torch.equal(current, previous):
                raise AssertionError("replay did not produce a new finite context with the published head")
            torch.testing.assert_close(download(weights["mla_latent_cache"], rank), cache, atol=0, rtol=0)
        if args.profile_steps:
            samples = defaultdict(list)
            names = {id(program): name for name, program in programs.items()}

            def timed_run(program, *bindings, **options):
                start = perf_counter_ns()
                worker.run(program, *bindings, **options)
                samples[names[id(program)]].append((perf_counter_ns() - start) / 1e6)

            runner._run_l3 = timed_run
            for _ in range(2):
                updater.step(upstream, meta, slot=0)
            samples.clear()
            step_ms = []
            for _ in range(args.profile_steps):
                start = perf_counter_ns()
                updater.step(upstream, meta, slot=0)
                step_ms.append((perf_counter_ns() - start) / 1e6)
            report["profile"] = {
                "warmup_steps": 2, "measured_steps": args.profile_steps,
                "step_ms_median": median(step_ms),
                "dispatch_ms_median": {name: median(values) for name, values in samples.items()},
                "step_ms_samples": step_ms, "dispatch_ms_samples": dict(samples),
            }

        rejected = torch.tensor([float("nan")], dtype=torch.float32)
        upload(updater.gradient, [rejected] * ranks)
        try:
            updater._check_gradients()
        except FloatingPointError:
            report["nonfinite_gradient_rejected"] = True
        else:
            raise AssertionError("the resident gradient guard accepted a NaN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, default=0, help="time this many extra resident steps after warmup")
    args = parser.parse_args()
    if args.profile_steps < 0:
        parser.error("--profile-steps must be nonnegative")
    if args.output.exists() or len(set(args.devices)) != len(args.devices):
        parser.error("use a fresh output path and distinct devices")
    report = {"ranks": len(args.devices), "passed": False}
    try:
        validate(args, report)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
