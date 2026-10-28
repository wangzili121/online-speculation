"""使用 Torch-NPU 算子实现一层 Orthrus-DSV4 Attention

Orthrus 让 K 个 draft 位置都能读取主模型已接受的历史，以及临时 draft 块内
的全部位置：

    draft hidden states -> diffusion Q/KV 投影
                        -> 主模型历史 + 双向 draft 块
                        -> LSE 合并 -> diffusion 输出投影

历史张量模拟只读的主模型 cache。本模块参照 ``dsv4.py``，但不依赖 vLLM
metadata。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch_npu
from torch import nn

from .dsv4_hybrid import merge_attention_segments


AttentionKind = Literal["swa", "hca", "csa"]


@dataclass(frozen=True)
class NativeDSV4Config:
    """一层 diffusion attention 所需的 DSV4 结构参数。"""

    # 主 Attention 和 MLA 风格低秩投影的维度。
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    # CSA indexer 的维度以及 SWA 历史窗口大小。
    index_heads: int
    index_head_dim: int
    index_topk: int
    window_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    @classmethod
    def dsv4_flash(cls) -> "NativeDSV4Config":
        """DeepSeek-V4-Flash-0731 模型配置中的真实维度。"""
        return cls(
            hidden_size=4096,
            num_heads=64,
            head_dim=512,
            rope_head_dim=64,
            q_lora_rank=1024,
            o_lora_rank=1024,
            o_groups=8,
            index_heads=64,
            index_head_dim=128,
            index_topk=512,
            window_size=128,
        )

    def validate(self) -> None:
        if self.head_dim < self.rope_head_dim or self.index_head_dim < self.rope_head_dim:
            raise ValueError("RoPE dimensions exceed an attention head")
        if self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be even")
        if self.num_heads % self.o_groups:
            raise ValueError("num_heads must be divisible by o_groups")


@dataclass(frozen=True)
class SyntheticTargetHistory:
    """只读主模型 cache：原始 KV、HCA C128、CSA C4 和 index key。"""

    raw_kv: torch.Tensor
    c4_kv: torch.Tensor
    c128_kv: torch.Tensor
    index_key_pages: torch.Tensor
    index_block_table: torch.Tensor
    index_key_lengths: torch.Tensor


@dataclass(frozen=True)
class NativeLayerOutput:
    """临时 draft 输出；CSA 还返回选中的历史位置。"""

    output: torch.Tensor
    topk_indices: torch.Tensor | None


def make_synthetic_history(
    config: NativeDSV4Config,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    c4_length: int | None = None,
    c128_length: int = 16,
    index_block_size: int = 128,
) -> SyntheticTargetHistory:
    """构造布局与 DSV4 兼容的模拟 cache。"""
    c4_length = c4_length or max(config.index_topk + index_block_size, index_block_size)
    if c4_length < config.index_topk:
        raise ValueError("C4 history must cover index_topk")
    pages = math.ceil(c4_length / index_block_size)
    padded_c4 = pages * index_block_size

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype) * 0.02

    raw_kv = randn(batch, config.window_size, 1, config.head_dim)
    c4_kv = randn(batch, c4_length, 1, config.head_dim)
    c128_kv = randn(batch, c128_length, 1, config.head_dim)
    index_key_pages = randn(
        batch * pages, index_block_size, 1, config.index_head_dim
    )
    index_block_table = torch.arange(
        batch * pages, device=device, dtype=torch.int32
    ).view(batch, pages)
    index_key_lengths = torch.full(
        (batch,), c4_length, device=device, dtype=torch.int32
    )
    if padded_c4 != c4_length:
        # 分页 index cache 保留 padding，但有效长度会排除这些位置。
        index_key_pages = index_key_pages.contiguous()
    return SyntheticTargetHistory(
        raw_kv=raw_kv,
        c4_kv=c4_kv,
        c128_kv=c128_kv,
        index_key_pages=index_key_pages,
        index_block_table=index_block_table,
        index_key_lengths=index_key_lengths,
    )


def _normalize_fia_lse(lse: torch.Tensor, batch: int, queries: int) -> torch.Tensor:
    if lse.ndim == 4 and lse.shape[0] == batch and lse.shape[2] == queries:
        return lse.transpose(1, 2)
    if lse.ndim == 4 and lse.shape[:2] == (batch, queries):
        return lse
    raise RuntimeError(f"unexpected FIA LSE shape: {tuple(lse.shape)}")


def _native_fia_segment(
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_lengths: list[int],
    key_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算一段历史或块内 Attention，并返回输出和 LSE。"""
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        key_value.contiguous(),
        key_value.contiguous(),
        num_heads=query.shape[2],
        num_key_value_heads=key_value.shape[2],
        input_layout="BSND",
        sparse_mode=0,
        scale=query.shape[-1] ** -0.5,
        atten_mask=None,
        block_table=None,
        block_size=0,
        actual_seq_lengths=query_lengths,
        actual_seq_lengths_kv=key_lengths,
        softmax_lse_flag=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("FIA did not return output and softmax LSE")
    output, lse = result[:2]
    return output, _normalize_fia_lse(lse, query.shape[0], query.shape[1])


class TorchNPUOrthrusDSV4Attention(nn.Module):
    """DSV4 单层中的 Orthrus diffusion attention 分支。"""

    def __init__(
        self,
        config: NativeDSV4Config,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        factory = {"device": device, "dtype": dtype}

        # ORTHRUS 改动：增加独立于 AR 分支的可训练 diffusion Q 投影。
        self.wq_a = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False, **factory
        )
        self.q_norm = nn.Parameter(torch.ones(config.q_lora_rank, **factory))
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.num_heads * config.head_dim,
            bias=False,
            **factory,
        )

        # ORTHRUS 改动：生成临时块 K/V，不写入主模型 cache。
        self.wkv = nn.Linear(
            config.hidden_size, config.head_dim, bias=False, **factory
        )
        self.kv_norm = nn.Parameter(torch.ones(config.head_dim, **factory))

        # ORTHRUS 改动：为每个 draft 位置生成 CSA 检索 query。
        self.index_wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_heads * config.index_head_dim,
            bias=False,
            **factory,
        )
        self.index_weights = nn.Linear(
            config.hidden_size, config.index_heads, bias=False, **factory
        )

        # DSV4 分组输出投影，将结果映射回共享 backbone 的 hidden size。
        group_dim = config.num_heads * config.head_dim // config.o_groups
        self.wo_a = nn.Parameter(
            torch.empty(config.o_groups, group_dim, config.o_lora_rank, **factory)
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
            **factory,
        )
        self.attn_sink = nn.Parameter(
            torch.zeros(config.num_heads, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            "head_norm", torch.ones(config.head_dim, **factory), persistent=False
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.wq_a,
            self.wq_b,
            self.wkv,
            self.index_wq_b,
            self.index_weights,
            self.wo_b,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.wo_a, mean=0.0, std=0.01)

    def _rotary_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0,
                    config.rope_head_dim,
                    2,
                    device=positions.device,
                    dtype=torch.float32,
                )
                / config.rope_head_dim
            )
        )
        angles = positions.reshape(-1, 1).float() * inv_freq.view(1, -1)
        angles = angles.repeat_interleave(2, dim=-1)
        shape = (positions.numel(), 1, 1, config.rope_head_dim)
        return angles.cos().view(shape), angles.sin().view(shape)

    def _partial_rope(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        rope_dim = self.config.rope_head_dim
        flat = value.reshape(-1, value.shape[-2], value.shape[-1])
        nope, rope = flat[..., :-rope_dim], flat[..., -rope_dim:]
        rotated = torch_npu.npu_rotary_mul(
            rope.reshape(rope.shape[0], rope.shape[1], 1, rope_dim).float(),
            cos,
            -sin if inverse else sin,
            rotary_mode="interleave",
        ).to(value.dtype)
        return torch.cat((nope, rotated.reshape_as(rope)), dim=-1).view_as(value)

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """为 K 个 draft 位置生成 diffusion Q 和块内 K/V。"""
        config = self.config
        batch, block, _ = hidden_states.shape
        q_a = self.wq_a(hidden_states)
        qr, _ = torch_npu.npu_rms_norm(
            q_a, self.q_norm, epsilon=config.rms_norm_eps
        )
        query = self.wq_b(qr).view(
            batch, block, config.num_heads, config.head_dim
        )
        query, _ = torch_npu.npu_rms_norm(
            query, self.head_norm, epsilon=config.rms_norm_eps
        )
        local_kv = self.wkv(hidden_states)
        local_kv, _ = torch_npu.npu_rms_norm(
            local_kv, self.kv_norm, epsilon=config.rms_norm_eps
        )
        local_kv = local_kv.view(batch, block, 1, config.head_dim)
        cos, sin = self._rotary_tables(positions)
        return (
            self._partial_rope(query, cos, sin),
            self._partial_rope(local_kv, cos, sin),
            qr,
            hidden_states,
        )

    def _native_indexer(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        history: SyntheticTargetHistory,
    ) -> torch.Tensor:
        """使用 diffusion query 为每个 draft 位置选择 C4 历史。"""
        config = self.config
        batch, block, _ = hidden_states.shape
        query = self.index_wq_b(qr).view(
            batch, block, config.index_heads, config.index_head_dim
        )
        cos, sin = self._rotary_tables(positions)
        query = self._partial_rope(query, cos, sin)
        weights = self.index_weights(hidden_states) * (
            config.index_head_dim**-0.5 * config.index_heads**-0.5
        )
        query_lengths = torch.full(
            (batch,), block, dtype=torch.int32, device=hidden_states.device
        )
        result = torch_npu.npu_lightning_indexer(
            query=query,
            key=history.index_key_pages,
            weights=weights,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_key=history.index_key_lengths,
            block_table=history.index_block_table,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=config.index_topk,
            sparse_mode=0,
        )
        indices = result[0]
        if indices.ndim == 4:
            if indices.shape[2] != 1:
                raise RuntimeError(f"unexpected indexer output: {tuple(indices.shape)}")
            indices = indices[:, :, 0]
        if indices.shape != (batch, block, config.index_topk):
            raise RuntimeError(f"unexpected indexer output: {tuple(indices.shape)}")
        return indices.long()

    @staticmethod
    def _gather_c4(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """从 C4 cache 中取出每个 draft query 选中的历史行。"""
        batch, queries, selected = indices.shape
        source = cache[:, None].expand(-1, queries, -1, -1, -1)
        gather_index = indices[..., None, None].expand(
            batch, queries, selected, cache.shape[2], cache.shape[3]
        )
        return torch.gather(source, 2, gather_index)

    def _output_projection(
        self,
        attention: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """执行 DSV4 inverse partial RoPE 和分组输出投影。"""
        config = self.config
        batch, block, _, _ = attention.shape
        cos, sin = self._rotary_tables(positions)
        attention = self._partial_rope(attention, cos, sin, inverse=True)
        grouped = attention.reshape(
            batch * block,
            config.o_groups,
            config.num_heads * config.head_dim // config.o_groups,
        )
        low_rank = torch_npu.npu_transpose_batchmatmul(
            grouped,
            self.wo_a,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
        return self.wo_b(low_rank.reshape(batch, block, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        history: SyntheticTargetHistory,
        kind: AttentionKind,
    ) -> NativeLayerOutput:
        """为 K 位置 draft 块执行一层 SWA、HCA 或 CSA。"""
        config = self.config
        batch, block, hidden = hidden_states.shape
        if hidden != config.hidden_size or positions.shape != (batch, block):
            raise ValueError("hidden_states or positions have incompatible shapes")
        query, local_kv, qr, hidden_states = self._project(hidden_states, positions)
        query_lengths = [block] * batch
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []

        # K 个 query 读取同一份已接受历史，但不修改它。
        raw_out, raw_lse = _native_fia_segment(
            query,
            history.raw_kv,
            query_lengths,
            [history.raw_kv.shape[1]] * batch,
        )
        outputs.append(raw_out)
        lses.append(raw_lse)

        topk_indices: torch.Tensor | None = None
        if kind == "hca":
            # HCA 额外读取 C128 压缩历史。
            compressed_out, compressed_lse = _native_fia_segment(
                query,
                history.c128_kv,
                query_lengths,
                [history.c128_kv.shape[1]] * batch,
            )
            outputs.append(compressed_out)
            lses.append(compressed_lse)
        elif kind == "csa":
            # CSA 为每个 draft 位置选择并读取 C4 历史。
            topk_indices = self._native_indexer(
                qr, hidden_states, positions, history
            )
            selected_c4 = self._gather_c4(history.c4_kv, topk_indices)
            flat_query = query.reshape(
                batch * block, 1, config.num_heads, config.head_dim
            )
            flat_c4 = selected_c4.reshape(
                batch * block, config.index_topk, 1, config.head_dim
            )
            compressed_out, compressed_lse = _native_fia_segment(
                flat_query,
                flat_c4,
                [1] * (batch * block),
                [config.index_topk] * (batch * block),
            )
            outputs.append(compressed_out.view_as(query))
            lses.append(
                compressed_lse.view(batch, block, config.num_heads, 1)
            )
        elif kind != "swa":
            raise ValueError(f"unknown attention kind: {kind}")

        # ORTHRUS 改动：不加因果 mask，K 个 draft 位置可以互相读取。
        local_out, local_lse = _native_fia_segment(
            query, local_kv, query_lengths, [block] * batch
        )
        outputs.append(local_out)
        lses.append(local_lse)

        # DSV4 attention sink 与其他部分共享 softmax 分母。
        outputs.append(torch.zeros_like(local_out))
        lses.append(
            self.attn_sink.view(1, 1, -1, 1).expand(batch, block, -1, -1)
        )
        # ORTHRUS 改动：用 LSE 精确合并只读历史与双向 draft 块。
        attention, _ = merge_attention_segments(outputs, lses)

        # 候选验证以及 cache 提交/回滚由推理引擎负责。
        output = self._output_projection(attention, positions)
        return NativeLayerOutput(
            output=output,
            topk_indices=topk_indices,
        )
