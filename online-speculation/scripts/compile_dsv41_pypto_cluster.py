#!/usr/bin/env python3
"""Compile the real A3 base and wide programs with serving-shaped metadata only."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from pypto_serving_a3.cache import CachePlan
from pypto_serving_a3.config import A3Config
from pypto_serving_a3.metadata import rank_metadata
from pypto_serving_a3.source import KernelSource
from pypto_serving_a3.state import RequestLedger


def compile_geometry(source, output, capacity, tokens, *, materialize):
    model = dict(text_config=asdict(source.module("config").FLASH), image_token_id=129264)
    model = json.loads(json.dumps(model))
    config = A3Config(4, 512, 128, capacity, 1, capacity)
    vocabulary = model["text_config"]["vocab_size"]
    hashed = source.module("engram_host").make_layout()
    constants = dict(token_map=torch.arange(vocabulary), multipliers=hashed.multipliers,
                     primes=hashed.primes, offsets=hashed.offsets,
                     layer_ids=[1, 14], pad_id=model["text_config"]["engram_pad_token_id"])
    templates = source.module("prefill_prepare").cache_templates(constants)
    cache_plan = CachePlan.from_native_templates(source, templates, config)
    ledger = RequestLedger(config, cache_plan, vocabulary)
    pages = {group.name: list(range(group.max_blocks_per_seq)) for group in cache_plan.groups}
    handle = ledger.admit("compile-probe", 0, tokens, pages, processed_length=tokens).handle
    chunks = (ledger.prepare_chunk(handle, [3] * tokens, 0, 0, "prefill"),)
    actual, _ = ledger.input_contract(chunks)
    payloads = [dict(payload, hash_constants=constants)
                for payload in rank_metadata(source, cache_plan, ledger, chunks)]
    receipts = [dict(rank=rank, payload_sha256="compile-only", report_sha256="compile-only")
                for rank in range(config.ep)]
    shards = [dict(head_range=[lane * 6, (lane + 1) * 6]) for lane in range(config.tp)]
    cache = dict(cache_ABI=payloads, cache_ABI_sources=receipts, engram_contracts={
        layer: dict(topology=config.topology, model_config=model,
                    hash_constants=constants, engram_shards=shards)
        for layer in (1, 14)
    })
    contract = dict(topology=config.topology, model_config=model, capacity=capacity, scheduler_capacity=8192,
                    required_modes=dict(enable_expert_parallel=True, enable_fused_mc2=True,
                        enable_dsa_cp=True, enable_flashcomm1=True, enable_shared_expert_dp=True),
                    fresh_shared_initialization={"topk_indices": -1}, input_contract_sha256="compile-only")
    program = source.module("prefill_fwd").compile_prefill(
        dict(input_contract=actual, prepare_input_stage=True),
        cache=cache, config=contract, output=output,
    )
    # ResidentNetworkRuntime.register also builds the actual callable images.
    compiled = {id(entry.compiled): entry.compiled for entries in program.stages.values()
                for entry in entries.values()}
    if len(compiled) > 64:
        raise ValueError("A compiled A3 phase exceeds the resident worker's 64 program slots")
    if materialize:
        for index, executable in enumerate(compiled.values(), 1):
            print(json.dumps(dict(stage="materialize_callable", capacity=capacity,
                                  entry=index, entries=len(compiled))), flush=True)
            executable.chip_callable
    return dict(capacity=capacity, tokens=tokens, stages=len(program.stages), compiled_entries=len(compiled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=("base", "wide", "both", "short"), default="both")
    parser.add_argument("--pto-only", action="store_true", help="Stop before building callable images")
    args = parser.parse_args()
    source = KernelSource.resolve(cluster=True)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    shapes = {"base": (16, 16), "wide": (128, 32), "short": (16, 8)}
    names = ("base", "wide") if args.shape == "both" else (args.shape,)
    programs = [compile_geometry(source, args.output / name, *shapes[name], materialize=not args.pto_only)
                for name in names]
    print(json.dumps(dict(passed=True, compile_only=True, callable_images=not args.pto_only,
                          programs=programs)), flush=True)


if __name__ == "__main__":
    main()
