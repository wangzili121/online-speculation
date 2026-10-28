#!/usr/bin/env python3
"""使用 Torch-NPU 算子运行一层 Orthrus-DSV4 diffusion attention。"""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from blockspec.vllm_ascend.torch_npu_dsv4_layer import (
    NativeDSV4Config,
    TorchNPUOrthrusDSV4Attention,
    make_synthetic_history,
)


def run_demo(config: NativeDSV4Config, block_size: int) -> dict:
    """对一个 Orthrus draft 块分别运行 SWA、HCA 和 CSA。"""
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    batch = 1
    layer = TorchNPUOrthrusDSV4Attention(config, device=device, dtype=dtype)
    # 用随机输入模拟共享 backbone 生成的 K 位置块。
    hidden = torch.randn(
        batch, block_size, config.hidden_size, device=device, dtype=dtype
    ) * 0.02
    positions = torch.arange(
        1024, 1024 + block_size, device=device, dtype=torch.long
    ).view(batch, block_size)
    # 用模拟 cache 代替主模型已经接受的历史。
    history = make_synthetic_history(
        config,
        batch=batch,
        device=device,
        dtype=dtype,
    )
    rows = []
    with torch.inference_mode():
        # DSV4 混合使用三种层，因此 Orthrus 需要全部支持。
        for kind in ("swa", "hca", "csa"):
            result = layer(hidden, positions, history, kind)
            rows.append(
                {
                    "kind": kind,
                    "output_shape": list(result.output.shape),
                    "topk_shape": (
                        list(result.topk_indices.shape)
                        if result.topk_indices is not None
                        else None
                    ),
                }
            )
    torch.npu.synchronize()
    return {
        "config": {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "index_heads": config.index_heads,
            "index_head_dim": config.index_head_dim,
            "index_topk": config.index_topk,
            "window_size": config.window_size,
            "block_size": block_size,
        },
        "paths": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261027)

    result = run_demo(NativeDSV4Config.dsv4_flash(), args.block_size)
    print(
        json.dumps(
            {
                "device": "npu:0",
                "dtype": "torch.bfloat16",
                "native_ops": [
                    "torch_npu.npu_rms_norm",
                    "torch_npu.npu_rotary_mul",
                    "torch_npu.npu_lightning_indexer",
                    "torch_npu.npu_fused_infer_attention_score",
                    "torch_npu.npu_transpose_batchmatmul",
                ],
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
