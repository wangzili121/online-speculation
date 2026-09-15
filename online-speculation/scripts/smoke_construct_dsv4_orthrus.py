#!/usr/bin/env python3
"""Construct the DSV4 Orthrus draft on 1-4 NPUs with dummy weights."""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--master-port", type=int, default=29871)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument(
        "--load-format",
        choices=("dummy", "auto"),
        default="dummy",
        help="use auto to read the real Orthrus head shards",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp = args.tensor_parallel_size or world_size
    if tp != world_size:
        raise SystemExit(
            f"tensor parallel size {tp} must match torchrun world size {world_size}"
        )
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    from blockspec.vllm_ascend.plugin import register

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.model,
        skip_tokenizer_init=True,
        tokenizer_mode="deepseek_v4",
        load_format=args.load_format,
        quantization="ascend",
        tensor_parallel_size=tp,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        speculative_config={
            "model": args.head,
            "method": "orthrus",
            "num_speculative_tokens": 31,
            "parallel_drafting": True,
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
            "enforce_eager": True,
        },
    )
    vllm_config = engine_args.create_engine_config()
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        raise RuntimeError("vLLM did not create a speculative config")

    register()
    torch.npu.set_device(local_rank)
    init_method = "env://"
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
            backend="hccl",
        )
        ensure_model_parallel_initialized(tp, 1)
        model = get_model(
            vllm_config=vllm_config,
            model_config=speculative_config.draft_model_config,
            load_config=vllm_config.load_config,
        )

    parameters = list(model.named_parameters())
    attention_parameters = [
        (name, parameter)
        for name, parameter in parameters
        if ".self_attn." in name
    ]

    loaded = None
    if tp == 1 and args.load_format == "dummy":
        def checkpoint_named_parameters():
            for name, parameter in attention_parameters:
                source_name = name.removeprefix("model.").replace(
                    ".self_attn.", ".attn.", 1
                )
                yield source_name, parameter.detach()

        with set_current_vllm_config(vllm_config):
            loaded = model.load_weights(checkpoint_named_parameters())
        if len(loaded) != len(attention_parameters):
            raise RuntimeError(
                f"loader mapped {len(loaded)} of "
                f"{len(attention_parameters)} attention parameters"
            )
    local_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for _, parameter in parameters
    )
    total_parameter_bytes = torch.tensor(
        local_parameter_bytes, dtype=torch.int64, device=f"npu:{local_rank}"
    )
    if world_size > 1:
        torch.distributed.all_reduce(total_parameter_bytes)
    result = {
        "status": "ok",
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "layers": len(model.model.layers),
        "tensor_parallel_size": tp,
        "attention_parameters": len(attention_parameters),
        "loader_roundtrip_parameters": len(loaded) if loaded is not None else None,
        "load_format": args.load_format,
        "local_parameter_bytes": local_parameter_bytes,
        "sum_parameter_bytes_across_ranks": int(total_parameter_bytes.item()),
        "orthrus_quant_aliases": sum(
            name.startswith("orthrus.layers.")
            for name in vllm_config.quant_config.quant_description
        ),
        "device": str(parameters[0][1].device),
    }
    if rank == 0:
        print(json.JSONEncoder(indent=2).encode(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
