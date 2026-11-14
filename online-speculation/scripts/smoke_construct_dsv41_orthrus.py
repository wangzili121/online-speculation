#!/usr/bin/env python3
"""Construct the full 40-layer V4.1 Orthrus head with dummy weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="V4.1 target config directory")
    parser.add_argument("--head", required=True, help="Orthrus head config directory")
    parser.add_argument("--master-port", type=int, default=29881)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument(
        "--exercise-loader",
        action="store_true",
        help="stream all checkpoint-format tensors through the TP loader",
    )
    parser.add_argument("--load-head", action="store_true", help="load the actual safetensors instead of a dummy head")
    args = parser.parse_args()
    if args.load_head and args.exercise_loader:
        parser.error("--load-head and --exercise-loader are mutually exclusive")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp = args.tensor_parallel_size or world_size
    if tp != world_size:
        raise ValueError("tensor parallel size must match torchrun world size")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    import torch
    from vllm.config import LoadConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    from blockspec.vllm_ascend.dsv41_contract import V41Topology, head_shapes
    from blockspec.vllm_ascend.head_contract import read_json
    from blockspec.vllm_ascend.plugin import register

    register()
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.model,
        skip_tokenizer_init=True,
        load_format="dummy",
        tensor_parallel_size=tp,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        speculative_config={
            "model": args.head,
            # vLLM 0.32 recognizes this parallel-scheduler family.  The head
            # architecture selects the Orthrus implementation in our plugin.
            "method": "dflash",
            "num_speculative_tokens": 31,
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
            "enforce_eager": True,
        },
    )
    vllm_config = engine_args.create_engine_config()
    speculative = vllm_config.speculative_config
    if speculative is None:
        raise RuntimeError("vLLM did not create the speculative configuration")

    torch.npu.set_device(local_rank)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="hccl",
        )
        ensure_model_parallel_initialized(tp, 1)
        model = get_model(
            vllm_config=vllm_config,
            model_config=speculative.draft_model_config,
            load_config=LoadConfig(load_format="safetensors") if args.load_head else vllm_config.load_config,
        )

    target_config = read_json(Path(args.model) / "config.json")
    expected = head_shapes(target_config, V41Topology.from_config(target_config))
    parameters = {
        name.removeprefix("model."): parameter
        for name, parameter in model.named_parameters()
    }
    missing = sorted(expected.keys() - parameters.keys())
    extra = sorted(parameters.keys() - expected.keys())
    if missing or extra:
        raise RuntimeError(
            f"head/runtime names mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    text = target_config["text_config"]
    groups = text["o_groups"]
    o_lora_rank = text["o_lora_rank"]

    def local_shape(name, checkpoint_shape):
        if name.endswith(".attn.attn_sink"):
            return (checkpoint_shape[0] // tp,)
        if name.endswith(".attn.wq_b.weight") and ".indexer." not in name:
            return (checkpoint_shape[0] // tp, checkpoint_shape[1])
        if name.endswith(".attn.wo_a.weight"):
            return (groups // tp, checkpoint_shape[1], o_lora_rank)
        if name.endswith(".attn.wo_b.weight"):
            return (checkpoint_shape[0], checkpoint_shape[1] // tp)
        return checkpoint_shape

    mismatched = sorted(
        name
        for name, shape in expected.items()
        if tuple(parameters[name].shape) != local_shape(name, shape)
    )
    if mismatched:
        details = [
            (name, tuple(parameters[name].shape), local_shape(name, expected[name]))
            for name in mismatched[:5]
        ]
        raise RuntimeError(f"head/runtime shape mismatch: {details}")

    # Stream one source tensor at a time so TP sharding is tested without
    # materializing the roughly 10 GiB exchange artifact in host memory.
    loaded = None
    loader_content_checks = None
    if not args.load_head and (tp == 1 or args.exercise_loader):
        signatures = {
            "layers.0.attn.attn_sink": 0,
            "layers.0.attn.wq_b.weight": 0,
            "layers.0.attn.wo_a.weight": 0,
            "layers.0.attn.wo_b.weight": 1,
        }

        def checkpoint_values():
            for name, shape in expected.items():
                dtype = torch.float32 if name.endswith(".attn_sink") else torch.bfloat16
                axis = signatures.get(name)
                if axis is None:
                    value = torch.zeros(shape, dtype=dtype)
                else:
                    coordinate = torch.arange(shape[axis]).remainder(127).to(dtype)
                    view = [1] * len(shape)
                    view[axis] = shape[axis]
                    value = coordinate.view(view).expand(shape)
                yield name, value

        loaded = model.load_weights(checkpoint_values())
    if loaded is not None and len(loaded) != len(expected):
        raise RuntimeError(f"loader accepted {len(loaded)} of {len(expected)} tensors")

    first = model.model.layers[0].attn
    expected_wo_shape = (
        first.n_local_groups,
        first.n_local_heads * first.head_dim // first.n_local_groups,
        first.o_lora_rank,
    )
    if tuple(first.wo_a.weight.shape) != expected_wo_shape:
        raise RuntimeError(
            f"wo_a post-load layout is {tuple(first.wo_a.weight.shape)}, expected {expected_wo_shape}"
        )
    if loaded is not None:
        local_heads = first.attn_sink.numel()
        local_q_rows = first.wq_b.weight.shape[0]
        local_groups = first.wo_a.weight.shape[0]
        local_o_columns = first.wo_b.weight.shape[1]
        probes = {
            "attn_sink": (
                first.attn_sink,
                torch.arange(rank * local_heads, (rank + 1) * local_heads).remainder(127),
            ),
            "wq_b": (
                first.wq_b.weight[:, 0],
                torch.arange(rank * local_q_rows, (rank + 1) * local_q_rows).remainder(127),
            ),
            "wo_a": (
                first.wo_a.weight[:, 0, :].reshape(-1),
                torch.arange(
                    rank * local_groups * first.o_lora_rank,
                    (rank + 1) * local_groups * first.o_lora_rank,
                ).remainder(127),
            ),
            "wo_b": (
                first.wo_b.weight[0],
                torch.arange(
                    rank * local_o_columns,
                    (rank + 1) * local_o_columns,
                ).remainder(127),
            ),
        }
        failed = [
            name
            for name, (actual, expected_values) in probes.items()
            if not torch.equal(actual.float().cpu(), expected_values.float())
        ]
        if failed:
            raise RuntimeError(f"TP loader content mismatch on rank {rank}: {failed}")
        loader_content_checks = len(probes)

    if args.load_head:
        from safetensors import safe_open
        from blockspec.vllm_ascend.head_contract import find_weight_index

        path = Path(args.head)
        mapping = read_json(find_weight_index(path))["weight_map"]
        name = f"layers.{len(model.model.layers) - 1}.attn.attn_sink"
        value = parameters[name]
        with safe_open(path / mapping[name], framework="pt", device="cpu") as shard:
            expected_sink = shard.get_tensor(name).chunk(tp)[rank]
        torch.testing.assert_close(value.detach().cpu(), expected_sink, atol=0, rtol=0)

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    total_parameter_bytes = torch.tensor(parameter_bytes, dtype=torch.int64, device=f"npu:{local_rank}")
    if world_size > 1:
        torch.distributed.all_reduce(total_parameter_bytes)
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "class": f"{type(model).__module__}.{type(model).__name__}",
                    "layers": len(model.model.layers),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "parameter_bytes": parameter_bytes,
                    "sum_parameter_bytes_across_ranks": int(total_parameter_bytes.item()),
                    "tensor_parallel_size": tp,
                    "contract_tensors": len(expected),
                    "loaded_head_from_safetensors": args.load_head,
                    "loader_roundtrip_tensors": len(loaded) if loaded is not None else None,
                    "loader_content_checks_per_rank": loader_content_checks,
                    "wo_a_layout": list(first.wo_a.weight.shape),
                    "device": str(next(model.parameters()).device),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
