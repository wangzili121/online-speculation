#!/usr/bin/env python3
"""GLM-5.3-Flash 的两类 draft Attention，使用真实维度和现成 Torch/NPU 算子。

这是独立的计算 demo，不包含共享 mHC/MoE、分词、采样和服务调度。
权重及历史为合成数据；推理接入使用相同的 MLA/KDA 核心函数。
"""

import argparse
import json
import math

import torch
from torch import nn

from blockspec.vllm_ascend.glm53_ops import (
    kda_convolve, kda_native, kda_reference, select_history, sparse_draft_attention,
)


class DraftAttentionDemo(nn.Module):
    """一个 TP8 分片的几何：8 个 MLA/KDA head，indexer 仍有完整的 32 个 head。"""

    def __init__(self):
        super().__init__()
        self.heads, self.dim, self.rank = 8, 128, 512
        self.qkv_a = nn.Linear(4096, 1536 + 512, bias=False)
        self.q_norm, self.kv_norm = nn.RMSNorm(1536, eps=1e-5), nn.RMSNorm(512, eps=1e-5)
        self.q_b = nn.Linear(1536, 8 * 256, bias=False)
        self.index_q = nn.Linear(1536, 32 * 128, bias=False)
        self.index_weight = nn.Linear(4096, 32, bias=False)
        self.mla_o = nn.Linear(8 * 256, 4096, bias=False)
        for name in ("ar_key", "ar_value", "draft_key", "draft_value"):
            # 两套展开权重故意不同，不能把历史当成 draft latent 来解释。
            self.register_buffer(name, torch.randn(8, 256, 512) / math.sqrt(512))
        self.qkvbfg = nn.Linear(4096, 3 * 8 * 128 + 8 + 2 * 128, bias=False)
        self.f_b, self.g_b = nn.Linear(128, 8 * 128, bias=False), nn.Linear(128, 8 * 128, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(3 * 8 * 128, 1, 4) * .1)
        self.register_buffer("a_log", torch.zeros(8))
        self.register_buffer("dt_bias", torch.full((8 * 128,), -3.))
        self.kda_norm = nn.RMSNorm(128, eps=1e-5)
        self.kda_o = nn.Linear(8 * 128, 4096, bias=False)
        self.backward_gate = nn.Parameter(torch.full((8,), math.atanh(.25)))

    def mla(self, hidden, history, pools, *, backend):
        """完整 MLA 分支：投影 -> 历史检索 -> 历史/双向块联合 Attention -> O 投影。"""
        q_resid, latent = self.qkv_a(hidden).split((1536, 512), -1)
        q_resid = self.q_norm(q_resid)
        query = self.q_b(q_resid).reshape(-1, 8, 256)
        block = self.kv_norm(latent)
        index_q = self.index_q(q_resid).reshape(-1, 32, 128)
        weights = self.index_weight(hidden) * (128**-.5 * 32**-.5)
        # ORTHRUS：所有行止于同一个历史边界，临时块不参加历史 Top-K。
        indices = select_history(index_q, weights, pools, len(history), topk=2048)
        output = sparse_draft_attention(
            query, history, block, indices, ar_key=self.ar_key, ar_value=self.ar_value,
            draft_key=self.draft_key, draft_value=self.draft_value, backend=backend,
        )
        return self.mla_o(output.flatten(1))

    def _convolve(self, rows, tail, backend):
        # 宽度 4 的逐通道卷积，读取 3 行历史；不把新尾部写回 target。
        mixed = kda_convolve(rows, tail, self.conv_weight[:, 0],
                             backend="auto" if backend == "npu" else "torch")
        return mixed.reshape(-1, 3, 8, 128).unbind(1)

    def kda(self, hidden, tail, state, *, backend, bidirectional):
        """KDA 分支：临时历史递推，可选零状态反向块，在 gated norm 前合并。"""
        rows, beta, f_a, g_a = self.qkvbfg(hidden).split((3 * 8 * 128, 8, 128, 128), -1)
        gate = self.f_b(f_a).reshape(-1, 8, 128)
        core = kda_native if backend == "npu" else kda_reference

        def run(qkv, left, initial, raw_gate, raw_beta):
            q, k, v = self._convolve(qkv, left, backend)
            output, _ = core(q, k, v,
                             raw_gate.contiguous(), raw_beta.contiguous(), initial,
                             self.a_log, self.dt_bias)
            return output

        # ORTHRUS：从已接受位置的 AR 状态出发，但只更新 core 内部的临时副本。
        output = run(rows, tail, state, gate, beta)
        if bidirectional:
            reverse = run(rows.flip(0), torch.zeros_like(tail), torch.zeros_like(state),
                          gate.flip(0), beta.flip(0))
            output = output + self.backward_gate.tanh()[None, :, None] * reverse.flip(0)
        # GLM 的输出门是 sigmoid；不是将两路分别归一化后相加。
        out_gate = self.g_b(g_a).reshape(-1, 8, 128).sigmoid()
        return self.kda_o((self.kda_norm(output) * out_gate).flatten(1))

    def forward(self, hidden, history, pools, tail, state, *, backend="torch", bidirectional=False):
        return dict(mla=self.mla(hidden, history, pools, backend=backend),
                    kda=self.kda(hidden, tail, state, backend=backend, bidirectional=bidirectional))


def make_case(block_size=8, prefix_length=2051, *, device="cpu", dtype=torch.float32):
    """合成权重和只读历史；2051 覆盖 512 个完整池及 3 个尾部 token。"""
    if block_size < 1 or prefix_length < 0:
        raise ValueError("positive block size and nonnegative prefix length required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5308)
        model = DraftAttentionDemo().eval().requires_grad_(False).to(device=device, dtype=dtype)
        hidden = torch.randn(block_size, 4096).to(device=device, dtype=dtype)
        history = torch.randn(prefix_length, 512).to(device=device, dtype=dtype) * .1
        pools = torch.randn(prefix_length // 4, 128).to(device=device, dtype=dtype) * .1
        tail = torch.randn(3, 3 * 8 * 128).to(device=device, dtype=dtype) * .1
        state = torch.randn(8, 128, 128).to(device=device, dtype=torch.float32) * .01
    # state 是 FP32，a_log/dt_bias 与原生 KDA 的接口一致，保持 FP32。
    model.a_log = model.a_log.float()
    model.dt_bias = model.dt_bias.float()
    return model, (hidden, history, pools, tail, state)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "npu"), default="torch")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=2051)
    parser.add_argument("--kda-mode", choices=("causal", "bidirectional"), default="causal")
    args = parser.parse_args()
    if args.backend == "npu":
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import enable_custom_op

        if not 1 <= args.block_size <= 8:
            parser.error("the current native KDA entry supports block sizes 1..8")
        torch.npu.set_device(0)
        if not enable_custom_op():
            raise RuntimeError("native Ascend operators are unavailable")
    model, inputs = make_case(args.block_size, args.prefix_length,
                              device="npu" if args.backend == "npu" else "cpu",
                              dtype=torch.bfloat16 if args.backend == "npu" else torch.float32)
    outputs = model(*inputs, backend=args.backend, bidirectional=args.kda_mode == "bidirectional")
    print(json.dumps(dict(backend=args.backend, block_size=args.block_size, kda_mode=args.kda_mode,
                          prefix_length=args.prefix_length,
                          outputs={name: dict(shape=list(value.shape), mean=float(value.float().mean()))
                                   for name, value in outputs.items()},
                          scope="synthetic attention branches; not full-model serving acceptance"), indent=2))


if __name__ == "__main__":
    main()
