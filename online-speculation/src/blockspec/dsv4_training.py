"""BF16 training model and artifact export for the DSV4 Orthrus head.

The serving model is deliberately not reused as a trainer.  vLLM-Ascend owns
the W8A8 target and its paged caches; this module wraps a differentiable BF16
Transformers DSV4 model, freezes it, and trains only the diffusion attention
twins that are consumed by ``OrthrusDSV4DraftModel``.
"""

from __future__ import annotations

import copy
from collections import Counter
import json
import math
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vllm_ascend.head_contract import (
    BLOCK_SIZE,
    FORMAT_NAME,
    FORMAT_VERSION,
    HISTORY_BACKEND,
    NUM_SPECULATIVE_TOKENS,
    find_weight_index,
    read_json,
    read_safetensors_header,
    shard_manifest,
    target_identity,
    tensor_schema,
    validate_manifest_header,
    write_json,
)


_LAYER_TYPES = {
    "sliding_attention": 1,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}

_QUANTIZED_WEIGHT_DTYPES = frozenset(
    {"F8_E4M3", "F8_E5M2", "I8", "U8"}
)


def validate_bf16_training_base(directory: Path) -> dict:
    """Validate real shard dtypes instead of trusting stale quantization metadata."""
    directory = Path(directory).resolve()
    index = read_json(find_weight_index(directory))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("training base needs a nonempty safetensors weight map")
    names = set(weight_map)
    paired_scales = {
        name
        for name in names
        if (
            name.endswith(".scale")
            and f"{name.removesuffix('.scale')}.weight" in names
        )
        or (
            name.endswith((".weight_scale", ".weight_offset"))
            and name.rsplit("_", 1)[0] in names
        )
    }
    dtype_counts: Counter[str] = Counter()
    quantized_weights = []
    by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    for shard, expected in sorted(by_shard.items()):
        header = read_safetensors_header(directory / shard)
        actual = set(header) - {"__metadata__"}
        if actual != expected:
            raise ValueError(
                f"training-base index/header mismatch in {shard}: "
                f"missing={sorted(expected - actual)[:4]}, "
                f"extra={sorted(actual - expected)[:4]}"
            )
        for name in expected:
            item = header[name]
            if name in paired_scales:
                continue
            dtype = str(item.get("dtype"))
            if not name.endswith(".weight"):
                continue
            dtype_counts[dtype] += 1
            if dtype in _QUANTIZED_WEIGHT_DTYPES:
                quantized_weights.append(name)
    if paired_scales or quantized_weights:
        raise ValueError(
            "the trainer needs dequantized floating-point DSV4 shards; found "
            f"{len(quantized_weights)} quantized weights and "
            f"{len(paired_scales)} paired quantization scales"
        )
    if not dtype_counts or not set(dtype_counts).issubset({"BF16", "F16", "F32", "F64"}):
        raise ValueError(f"unsupported DSV4 training weight dtypes: {dict(dtype_counts)}")
    return {
        "weight_tensors": sum(dtype_counts.values()),
        "weight_dtypes": dict(sorted(dtype_counts.items())),
        "shards": len(by_shard),
        "stale_quantization_metadata": bool(
            read_json(directory / "config.json").get("quantization_config")
        ),
    }


def _normalized_layout(values) -> list[int]:
    return [1 if int(value) in (0, 1) else int(value) for value in values]


def _rotate_half(value: Tensor) -> Tensor:
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rope(
    value: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1
) -> Tensor:
    """Apply DSV4 interleaved partial RoPE to ``[B,H,S,D]`` or ``[B,S,H,D]``."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = value[..., :-rope_dim], value[..., -rope_dim:]
    rotated = (rope.float() * cos + _rotate_half(rope).float() * sin).to(value.dtype)
    return torch.cat((nope, rotated), dim=-1)


def _linear(module: nn.Module, value: Tensor) -> Tensor:
    output = module(value)
    return output[0] if isinstance(output, tuple) else output


def _compress_clean(module: nn.Module, hidden: Tensor) -> Tensor:
    """Stateless target compressor matching Transformers' HCA/CSA layout."""
    batch, sequence, _ = hidden.shape
    rate = int(module.compress_rate)
    usable = sequence // rate * rate
    if usable == 0:
        return hidden.new_zeros((batch, 0, int(module.head_dim)))
    kv = _linear(module.kv_proj, hidden[:, :usable])
    gate = _linear(module.gate_proj, hidden[:, :usable])
    windows = usable // rate
    kv = kv.view(batch, windows, rate, -1)
    gate = gate.view(batch, windows, rate, -1) + module.position_bias
    head_dim = int(module.head_dim)
    if kv.shape[-1] == head_dim:
        pooled_kv, pooled_gate = kv, gate
    elif kv.shape[-1] == 2 * head_dim:
        pooled_kv = kv.new_zeros((batch, windows, 2 * rate, head_dim))
        pooled_gate = gate.new_full(
            (batch, windows, 2 * rate, head_dim), float("-inf")
        )
        pooled_kv[:, :, rate:] = kv[..., head_dim:]
        pooled_gate[:, :, rate:] = gate[..., head_dim:]
        if windows > 1:
            pooled_kv[:, 1:, :rate] = kv[:, :-1, :, :head_dim]
            pooled_gate[:, 1:, :rate] = gate[:, :-1, :, :head_dim]
    else:
        raise ValueError("unexpected DSV4 compressor projection width")
    weights = pooled_gate.softmax(dim=2, dtype=torch.float32).to(pooled_kv.dtype)
    compressed = _linear(module.kv_norm, (pooled_kv * weights).sum(dim=2))
    positions = torch.arange(windows, device=hidden.device) * rate
    positions = positions.unsqueeze(0).expand(batch, -1)
    cos, sin = module.rotary_emb(
        compressed, position_ids=positions, layer_type=module.rope_layer_type
    )
    return _apply_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)


def _gather_rows(value: Tensor, indices: Tensor) -> Tensor:
    """Gather ``value[B,S,D]`` with an arbitrary ``indices[B,...]`` grid."""
    if value.ndim != 3 or indices.ndim < 2 or value.shape[0] != indices.shape[0]:
        raise ValueError("gather expects value[B,S,D] and indices[B,...]")
    batch = torch.arange(value.shape[0], device=value.device)
    batch = batch.view(value.shape[0], *((1,) * (indices.ndim - 1))).expand_as(indices)
    return value[batch, indices]


def _history_indices(anchors: Tensor, width: int, sequence: int) -> tuple[Tensor, Tensor]:
    width = min(width, sequence)
    offsets = torch.arange(width, device=anchors.device)
    starts = (anchors - width).clamp_min(0)
    lengths = anchors - starts
    indices = (starts[..., None] + offsets).clamp_max(sequence - 1)
    return indices, offsets.view(*((1,) * anchors.ndim), -1) < lengths[..., None]


def _compressed_indices(
    anchors: Tensor, rate: int, available: int
) -> tuple[Tensor, Tensor, Tensor]:
    lengths = torch.div(anchors, rate, rounding_mode="floor").clamp_max(available)
    offsets = torch.arange(available, device=anchors.device)
    indices = offsets.view(*((1,) * anchors.ndim), -1).expand(*anchors.shape, -1)
    valid = indices < lengths[..., None]
    return indices, valid, lengths


class DSV4DiffusionAttention(nn.Module):
    """Trainable attention twin for one frozen DSV4 decoder layer."""

    def __init__(self, target: nn.Module, rotary_emb: nn.Module):
        super().__init__()
        self.layer_type = str(target.layer_type)
        if self.layer_type not in _LAYER_TYPES:
            raise ValueError(f"unsupported DSV4 layer type: {self.layer_type}")
        self.rope_layer_type = str(target.rope_layer_type)
        self.num_heads = int(target.num_heads)
        self.head_dim = int(target.head_dim)
        self.sliding_window = int(target.sliding_window)
        self.scaling = float(target.scaling)
        self.o_groups = int(target.config.o_groups)
        self.q_a_proj = copy.deepcopy(target.q_a_proj)
        self.q_a_norm = copy.deepcopy(target.q_a_norm)
        self.q_b_proj = copy.deepcopy(target.q_b_proj)
        self.q_b_norm = copy.deepcopy(target.q_b_norm)
        self.kv_proj = copy.deepcopy(target.kv_proj)
        self.kv_norm = copy.deepcopy(target.kv_norm)
        self.o_a_proj = copy.deepcopy(target.o_a_proj)
        self.o_b_proj = copy.deepcopy(target.o_b_proj)
        self.sinks = nn.Parameter(target.sinks.detach().clone())
        self.rotary_emb = copy.deepcopy(rotary_emb)
        self.indexer_q_b_proj: nn.Module | None = None
        self.indexer_weights_proj: nn.Module | None = None
        self.index_n_heads = self.index_head_dim = self.index_topk = 0
        self.index_scaling = self.index_weights_scaling = 0.0
        if self.layer_type == "compressed_sparse_attention":
            indexer = target.compressor.indexer
            self.indexer_q_b_proj = copy.deepcopy(indexer.q_b_proj)
            self.indexer_weights_proj = copy.deepcopy(indexer.scorer.weights_proj)
            self.index_n_heads = int(indexer.num_heads)
            self.index_head_dim = int(indexer.head_dim)
            self.index_topk = int(indexer.index_topk)
            self.index_scaling = float(indexer.scorer.softmax_scale)
            self.index_weights_scaling = float(indexer.scorer.weights_scaling)

    @property
    def compress_ratio(self) -> int:
        return _LAYER_TYPES[self.layer_type]

    def _query(self, hidden: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        batch, sequence, _ = hidden.shape
        residual = _linear(self.q_a_norm, _linear(self.q_a_proj, hidden))
        query = _linear(self.q_b_proj, residual).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        query = _linear(self.q_b_norm, query)
        return _apply_rope(query, cos, sin), residual

    def _local_kv(self, hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, sequence, _ = hidden.shape
        value = _linear(self.kv_norm, _linear(self.kv_proj, hidden))
        value = value.view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        return _apply_rope(value, cos, sin).squeeze(1)

    def _target_swa(self, target: nn.Module, hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, sequence, _ = hidden.shape
        value = _linear(target.kv_norm, _linear(target.kv_proj, hidden))
        value = value.view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        return _apply_rope(value, cos, sin).squeeze(1)

    def _indexer_scores(
        self,
        hidden: Tensor,
        q_residual: Tensor,
        compressed_keys: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        assert self.indexer_q_b_proj is not None
        assert self.indexer_weights_proj is not None
        batch, sequence, _ = hidden.shape
        query = _linear(self.indexer_q_b_proj, q_residual).view(
            batch, sequence, self.index_n_heads, self.index_head_dim
        ).transpose(1, 2)
        query = _apply_rope(query, cos, sin).transpose(1, 2)
        scores = torch.einsum("bqhd,bld->bqhl", query.float(), compressed_keys.float())
        scores = F.relu(scores) * self.index_scaling
        weights = _linear(self.indexer_weights_proj, hidden).float()
        weights = weights * self.index_weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)

    def _target_indexer_scores(
        self,
        target: nn.Module,
        hidden: Tensor,
        compressed_keys: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        indexer = target.compressor.indexer
        residual = _linear(target.q_a_norm, _linear(target.q_a_proj, hidden))
        batch, sequence, _ = hidden.shape
        query = _linear(indexer.q_b_proj, residual).view(
            batch, sequence, indexer.num_heads, indexer.head_dim
        ).transpose(1, 2)
        query = _apply_rope(query, cos, sin).transpose(1, 2)
        scores = torch.einsum("bqhd,bld->bqhl", query.float(), compressed_keys.float())
        scores = F.relu(scores) * float(indexer.scorer.softmax_scale)
        weights = _linear(indexer.scorer.weights_proj, hidden).float()
        weights = weights * float(indexer.scorer.weights_scaling)
        return (scores * weights.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        block_hidden: Tensor,
        clean_hidden: Tensor,
        anchors: Tensor,
        positions: Tensor,
        target: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        """Attend each K-row block to read-only clean history and its own block."""
        batch, blocks, block_size, hidden_size = block_hidden.shape
        if anchors.shape != (batch, blocks) or clean_hidden.shape[:1] != (batch,):
            raise ValueError("aligned clean history, blocks and anchors required")
        flat_hidden = block_hidden.reshape(batch * blocks, block_size, hidden_size)
        flat_positions = positions.reshape(batch * blocks, block_size)
        cos, sin = self.rotary_emb(
            flat_hidden, position_ids=flat_positions, layer_type=self.rope_layer_type
        )
        query, q_residual = self._query(flat_hidden, cos, sin)
        local_kv = self._local_kv(flat_hidden, cos, sin)

        clean_positions = torch.arange(
            clean_hidden.shape[1], device=clean_hidden.device
        ).unsqueeze(0).expand(batch, -1)
        clean_cos, clean_sin = self.rotary_emb(
            clean_hidden,
            position_ids=clean_positions,
            layer_type=self.rope_layer_type,
        )
        clean_swa = self._target_swa(target, clean_hidden, clean_cos, clean_sin)
        swa_indices, swa_valid = _history_indices(
            anchors, self.sliding_window, clean_hidden.shape[1]
        )
        swa = _gather_rows(clean_swa, swa_indices).reshape(
            batch * blocks, swa_indices.shape[-1], self.head_dim
        )
        swa_valid = swa_valid.reshape(batch * blocks, -1)

        key_segments = [
            swa[:, None].expand(-1, block_size, -1, -1),
            local_kv[:, None].expand(-1, block_size, -1, -1),
        ]
        valid_segments = [
            swa_valid[:, None].expand(-1, block_size, -1),
            torch.ones(
                (batch * blocks, block_size, block_size),
                dtype=torch.bool,
                device=block_hidden.device,
            ),
        ]
        indexer_loss = flat_hidden.new_zeros((), dtype=torch.float32)

        if self.compress_ratio > 1:
            compressed = _compress_clean(target.compressor, clean_hidden)
            available = compressed.shape[1]
            _, compressed_valid, compressed_lengths = _compressed_indices(
                anchors, self.compress_ratio, available
            )
            if self.compress_ratio == 4 and available:
                indexer = target.compressor.indexer
                index_keys = _compress_clean(indexer, clean_hidden)
                expanded_index_keys = index_keys[:, None].expand(
                    -1, blocks, -1, -1
                ).reshape(batch * blocks, available, self.index_head_dim)
                student_scores = self._indexer_scores(
                    flat_hidden, q_residual, expanded_index_keys, cos, sin
                )
                valid = compressed_valid.reshape(batch * blocks, 1, available)
                masked_scores = student_scores.masked_fill(~valid, float("-inf"))
                topk = min(self.index_topk, available)
                top_values, top_indices = masked_scores.topk(topk, dim=-1)
                expanded = compressed[:, None].expand(-1, blocks, -1, -1)
                expanded = expanded.reshape(batch * blocks, available, self.head_dim)
                selected = torch.gather(
                    expanded[:, None].expand(-1, block_size, -1, -1),
                    2,
                    top_indices[..., None].expand(-1, -1, -1, self.head_dim),
                )
                selected_valid = torch.isfinite(top_values)
                key_segments.append(selected)
                valid_segments.append(selected_valid)

                clean_block = _gather_rows(clean_hidden, positions).reshape(
                    batch * blocks, block_size, hidden_size
                )
                teacher_scores = self._target_indexer_scores(
                    target, clean_block, expanded_index_keys, cos, sin
                ).detach()
                score_mask = valid.expand(-1, block_size, -1)
                if score_mask.any():
                    indexer_loss = F.smooth_l1_loss(
                        student_scores[score_mask], teacher_scores[score_mask]
                    )
            elif self.compress_ratio == 128 and available:
                indices = torch.arange(available, device=anchors.device)
                indices = indices.view(1, 1, -1).expand(batch, blocks, -1)
                gathered = _gather_rows(compressed, indices).reshape(
                    batch * blocks, available, self.head_dim
                )
                key_segments.append(
                    gathered[:, None].expand(-1, block_size, -1, -1)
                )
                valid_segments.append(
                    compressed_valid.reshape(batch * blocks, 1, available).expand(
                        -1, block_size, -1
                    )
                )

        keys = torch.cat(key_segments, dim=2)
        valid = torch.cat(valid_segments, dim=2)
        logits = torch.einsum("bhqd,bqtd->bhqt", query.float(), keys.float())
        logits = logits * self.scaling
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        sink = self.sinks.float().view(1, self.num_heads, 1, 1)
        sink = sink.expand(batch * blocks, -1, block_size, -1)
        logits = torch.cat((logits, sink), dim=-1)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probabilities = logits.softmax(dim=-1, dtype=torch.float32)[..., :-1]
        output = torch.einsum("bhqt,bqtd->bqhd", probabilities, keys.float())
        output = output.to(flat_hidden.dtype)
        output = _apply_rope(output, cos, -sin, unsqueeze_dim=2)
        grouped = output.reshape(batch * blocks, block_size, self.o_groups, -1)
        output = _linear(self.o_a_proj, grouped).flatten(2)
        output = _linear(self.o_b_proj, output)
        return output.reshape(batch, blocks, block_size, hidden_size), indexer_loss

    def export_tensors(self) -> dict[str, Tensor]:
        tensors = {
            "attn_sink": self.sinks,
            "wq_a.weight": self.q_a_proj.weight,
            "q_norm.weight": self.q_a_norm.weight,
            "wq_b.weight": self.q_b_proj.weight,
            "wkv.weight": self.kv_proj.weight,
            "kv_norm.weight": self.kv_norm.weight,
            "wo_a.weight": self.o_a_proj.weight,
            "wo_b.weight": self.o_b_proj.weight,
        }
        if self.compress_ratio == 4:
            assert self.indexer_q_b_proj is not None
            assert self.indexer_weights_proj is not None
            tensors.update(
                {
                    "indexer.wq_b.weight": self.indexer_q_b_proj.weight,
                    "indexer.weights_proj.weight": self.indexer_weights_proj.weight,
                }
            )
        return tensors


class OrthrusDSV4TrainingModel(nn.Module):
    """Frozen HF DSV4 plus trainable SWA/C4/C128 diffusion attention twins."""

    def __init__(self, target: nn.Module, *, block_size: int = BLOCK_SIZE, mask_token_id: int):
        super().__init__()
        if block_size < 2 or mask_token_id < 0:
            raise ValueError("block_size >= 2 and a nonnegative mask token are required")
        if not hasattr(target, "model") or not hasattr(target, "lm_head"):
            raise TypeError("a Transformers DeepseekV4ForCausalLM target is required")
        layer_types = list(getattr(target.config, "layer_types", []))
        if len(layer_types) != len(target.model.layers):
            raise ValueError("DSV4 config must declare one layer type per decoder layer")
        unknown = sorted(set(layer_types) - set(_LAYER_TYPES))
        if unknown:
            raise ValueError(f"unsupported DSV4 layer types: {unknown}")
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.head = nn.ModuleList(
            DSV4DiffusionAttention(layer.self_attn, target.model.rotary_emb)
            for layer in target.model.layers
        ).requires_grad_(True)
        self.target = target.requires_grad_(False).eval()

    @property
    def config(self):
        return self.target.config

    @property
    def compress_ratios(self) -> list[int]:
        configured = list(getattr(self.config, "compress_ratios", []) or [])
        if len(configured) >= len(self.head):
            ratios = [int(value) for value in configured[: len(self.head)]]
            if _normalized_layout(ratios) != [layer.compress_ratio for layer in self.head]:
                raise ValueError("DSV4 compress_ratios and layer_types disagree")
            return ratios
        return [layer.compress_ratio for layer in self.head]

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()
        self.head.train(mode)
        return self

    def head_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    @property
    def input_device(self) -> torch.device:
        return self.target.model.embed_tokens.weight.device

    @property
    def output_device(self) -> torch.device:
        return self.target.lm_head.weight.device

    def place_layerwise(
        self,
        layer_devices: list[str | torch.device],
        *,
        input_device: str | torch.device | None = None,
        output_device: str | torch.device | None = None,
    ) -> "OrthrusDSV4TrainingModel":
        """Place frozen layers and their trainable twins for sequential model parallelism."""
        if len(layer_devices) != len(self.head):
            raise ValueError("layer device map must contain one entry per DSV4 layer")
        devices = [torch.device(device) for device in layer_devices]
        input_device = torch.device(input_device or devices[0])
        output_device = torch.device(output_device or devices[-1])
        self.target.model.embed_tokens.to(input_device)
        for layer, head, device in zip(
            self.target.model.layers, self.head, devices, strict=True
        ):
            layer.to(device)
            head.to(device)
        self.target.model.hc_head.to(output_device)
        self.target.model.norm.to(output_device)
        self.target.lm_head.to(output_device)
        return self

    def _causal_mask(self, batch: int, sequence: int, dtype: torch.dtype, device) -> Tensor:
        query = torch.arange(sequence, device=device)[:, None]
        key = torch.arange(sequence, device=device)[None, :]
        allowed = (key <= query) & (key > query - int(self.config.sliding_window))
        mask = torch.zeros((sequence, sequence), dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask.view(1, 1, sequence, sequence).expand(batch, -1, -1, -1)

    def _attention_step(
        self,
        layer: nn.Module,
        head: DSV4DiffusionAttention,
        clean: Tensor,
        draft: Tensor,
        input_ids: Tensor,
        block_tokens: Tensor,
        anchors: Tensor,
        positions: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = clean.dtype
        with torch.no_grad():
            clean_post, clean_comb, clean_collapsed = layer.attn_hc(clean)
            clean_normed = layer.input_layernorm(clean_collapsed)
            clean_positions = torch.arange(
                clean.shape[1], device=clean.device
            ).unsqueeze(0).expand(clean.shape[0], -1)
            clean_position_embeddings = {
                name: head.rotary_emb(
                    clean_normed, position_ids=clean_positions, layer_type=name
                )
                for name in ("main", "compress")
            }
            clean_attention, _ = layer.self_attn(
                clean_normed,
                position_embeddings=clean_position_embeddings,
                position_ids=clean_positions,
                attention_mask=attention_mask,
                past_key_values=None,
            )
            clean = clean_post.to(dtype).unsqueeze(-1) * clean_attention.unsqueeze(-2) + torch.matmul(
                clean_comb.to(dtype).transpose(-1, -2), clean
            )

        batch, blocks, block_size, copies, hidden_size = draft.shape
        flat = draft.reshape(batch * blocks, block_size, copies, hidden_size)
        draft_post, draft_comb, draft_collapsed = layer.attn_hc(flat)
        draft_normed = layer.input_layernorm(draft_collapsed)
        draft_attention, indexer_loss = head(
            draft_normed.reshape(batch, blocks, block_size, hidden_size),
            clean_normed,
            anchors,
            positions,
            layer.self_attn,
        )
        draft_attention = draft_attention.reshape(batch * blocks, block_size, hidden_size)
        flat = draft_post.to(flat.dtype).unsqueeze(-1) * draft_attention.unsqueeze(-2) + torch.matmul(
            draft_comb.to(flat.dtype).transpose(-1, -2), flat
        )
        draft = flat.reshape(batch, blocks, block_size, copies, hidden_size)

        with torch.no_grad():
            clean_post, clean_comb, clean_collapsed = layer.ffn_hc(clean)
            clean_mlp = layer.mlp(
                layer.post_attention_layernorm(clean_collapsed), input_ids=input_ids
            )
            clean = clean_post.to(dtype).unsqueeze(-1) * clean_mlp.unsqueeze(-2) + torch.matmul(
                clean_comb.to(dtype).transpose(-1, -2), clean
            )

        flat = draft.reshape(batch * blocks, block_size, copies, hidden_size)
        draft_post, draft_comb, draft_collapsed = layer.ffn_hc(flat)
        draft_mlp = layer.mlp(
            layer.post_attention_layernorm(draft_collapsed),
            input_ids=block_tokens.reshape(batch * blocks, block_size),
        )
        flat = draft_post.to(flat.dtype).unsqueeze(-1) * draft_mlp.unsqueeze(-2) + torch.matmul(
            draft_comb.to(flat.dtype).transpose(-1, -2), flat
        )
        return clean, flat.reshape_as(draft), indexer_loss

    def hidden_states(self, input_ids: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if input_ids.ndim != 2 or anchors.ndim != 2 or anchors.shape[0] != input_ids.shape[0]:
            raise ValueError("input_ids[B,S] and anchors[B,A] required")
        batch, sequence = input_ids.shape
        if anchors.dtype != torch.long or ((anchors < 0) | (anchors + self.block_size > sequence)).any():
            raise ValueError("every integer anchor must have a complete block")
        input_ids = input_ids.to(self.input_device)
        anchors = anchors.to(self.input_device)
        offsets = torch.arange(self.block_size, device=self.input_device)
        positions = anchors[..., None] + offsets
        block_tokens = input_ids.new_full(positions.shape, self.mask_token_id)
        block_tokens[:, :, 0] = input_ids.gather(1, anchors)
        with torch.no_grad():
            clean_embed = self.target.model.embed_tokens(input_ids)
            clean = clean_embed.unsqueeze(2).expand(
                -1, -1, int(self.config.hc_mult), -1
            ).contiguous()
            block_embed = self.target.model.embed_tokens(block_tokens)
        draft = block_embed.unsqueeze(3).expand(
            -1, -1, -1, int(self.config.hc_mult), -1
        ).contiguous()
        indexer_losses = []
        for layer, head in zip(self.target.model.layers, self.head, strict=True):
            device = next(head.parameters()).device
            clean = clean.to(device)
            draft = draft.to(device)
            layer_input_ids = input_ids.to(device)
            layer_block_tokens = block_tokens.to(device)
            layer_anchors = anchors.to(device)
            layer_positions = positions.to(device)
            attention_mask = self._causal_mask(
                batch, sequence, clean.dtype, device
            )
            clean, draft, indexer_loss = self._attention_step(
                layer,
                head,
                clean,
                draft,
                layer_input_ids,
                layer_block_tokens,
                layer_anchors,
                layer_positions,
                attention_mask,
            )
            indexer_losses.append(indexer_loss.to(self.output_device))
        clean = clean.to(self.output_device)
        draft = draft.to(self.output_device)
        with torch.no_grad():
            clean = self.target.model.norm(self.target.model.hc_head(clean))
        batch, blocks, block_size, copies, hidden_size = draft.shape
        draft = draft.reshape(batch * blocks, block_size, copies, hidden_size)
        draft = self.target.model.norm(self.target.model.hc_head(draft))
        draft = draft.reshape(batch, blocks, block_size, hidden_size)
        indexer_loss = torch.stack(indexer_losses).mean() if indexer_losses else draft.new_zeros(())
        return clean, draft, indexer_loss

    def loss(
        self,
        input_ids: Tensor,
        anchors: Tensor,
        *,
        objective: str = "forward_kl",
        chunk_rows: int = 32,
        indexer_weight: float = 0.01,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if objective not in ("forward_kl", "cross_entropy"):
            raise ValueError("objective must be forward_kl or cross_entropy")
        if chunk_rows < 1 or not math.isfinite(indexer_weight) or indexer_weight < 0:
            raise ValueError("positive chunks and finite nonnegative indexer weight required")
        clean, draft, indexer_loss = self.hidden_states(input_ids, anchors)
        input_ids = input_ids.to(clean.device)
        anchors = anchors.to(clean.device)
        offsets = torch.arange(self.block_size - 1, device=clean.device)
        teacher_positions = anchors[..., None] + offsets
        teacher_hidden = _gather_rows(clean, teacher_positions).reshape(-1, clean.shape[-1])
        student_hidden = draft[:, :, :-1].reshape(-1, draft.shape[-1])
        labels = _gather_rows(input_ids[..., None], teacher_positions + 1).reshape(-1)
        terms = []
        count = student_hidden.shape[0]

        def projected_loss(student_rows: Tensor, teacher_rows: Tensor, targets: Tensor) -> Tensor:
            student_logits = _linear(self.target.lm_head, student_rows).float()
            if objective == "cross_entropy":
                return F.cross_entropy(student_logits, targets)
            with torch.no_grad():
                teacher_logits = _linear(self.target.lm_head, teacher_rows).float()
                teacher_log = teacher_logits.log_softmax(-1)
            student_log = student_logits.log_softmax(-1)
            return (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()

        for start in range(0, count, chunk_rows):
            stop = min(start + chunk_rows, count)
            term = checkpoint(
                projected_loss,
                student_hidden[start:stop],
                teacher_hidden[start:stop],
                labels[start:stop],
                use_reentrant=False,
            )
            terms.append(term * ((stop - start) / count))
        token_loss = torch.stack(terms).sum()
        total = token_loss + indexer_loss * indexer_weight
        return total, {
            "token_loss": token_loss.detach(),
            "indexer_loss": indexer_loss.detach(),
        }

    def export_state(self) -> list[dict[str, Tensor]]:
        return [
            {name: value.detach() for name, value in layer.export_tensors().items()}
            for layer in self.head
        ]


def load_bf16_head(model: OrthrusDSV4TrainingModel, directory: Path) -> None:
    """Load a complete trained/initialized v2 BF16 head into the trainer."""
    from safetensors import safe_open

    directory = Path(directory).resolve()
    manifest = read_json(directory / "orthrus_head_manifest.json")
    validate_manifest_header(manifest)
    if manifest.get("precision") != "bf16":
        raise ValueError("training can only resume from the BF16 exchange head")
    if _normalized_layout(manifest.get("compress_ratios", [])) != _normalized_layout(
        model.compress_ratios
    ):
        raise ValueError("head artifact and BF16 training model use different hybrid layouts")
    index = read_json(find_weight_index(directory))
    weight_map = index.get("weight_map", {})
    expected = {
        f"layers.{layer}.attn.{name}": parameter
        for layer, state in enumerate(model.export_state())
        for name, parameter in state.items()
    }
    if set(weight_map) != set(expected):
        missing = sorted(set(expected) - set(weight_map))
        extra = sorted(set(weight_map) - set(expected))
        raise ValueError(f"head tensor set mismatch: missing={missing[:4]}, extra={extra[:4]}")
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    loaded = {}
    for shard, names in by_shard.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in names:
                value = handle.get_tensor(name)
                if value.shape != expected[name].shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid BF16 head tensor: {name}")
                loaded[name] = value
    with torch.no_grad():
        for name, destination in expected.items():
            destination.copy_(loaded[name].to(destination.device, destination.dtype))


def export_bf16_head(
    model: OrthrusDSV4TrainingModel,
    output: Path,
    *,
    serving_target: Path,
    training_base: Path,
    mask_token_id: int,
    base_model_id: str,
    training_metadata: dict | None = None,
) -> Path:
    """Write a trained v2 BF16 exchange artifact ready for independent calibration."""
    from safetensors.torch import save_file

    if model.block_size != BLOCK_SIZE:
        raise ValueError(f"a deployable DSV4 head must be trained with K={BLOCK_SIZE}")
    output = Path(output).resolve()
    serving_target = Path(serving_target).resolve()
    training_base = Path(training_base).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    serving_config = json.loads((serving_target / "config.json").read_text(encoding="utf-8"))
    ratios = [int(value) for value in serving_config.get("compress_ratios", [])]
    if _normalized_layout(ratios) != _normalized_layout(model.compress_ratios):
        raise ValueError(
            f"training/serving hybrid layouts differ: {model.compress_ratios} != {ratios}"
        )
    weight_map: dict[str, str] = {}
    states = model.export_state()
    for index, state in enumerate(states):
        tensors = {
            f"layers.{index}.attn.{name}": value.to(dtype=torch.bfloat16, device="cpu").contiguous()
            for name, value in state.items()
        }
        filename = f"orthrus-head-{index + 1:05d}-of-{len(states):05d}.safetensors"
        save_file(tensors, output / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in tensors})
    index_name = "model.safetensors.index.json"
    write_json(output / index_name, {"metadata": {}, "weight_map": weight_map})
    schema, total_size = tensor_schema(output, weight_map)
    write_json(
        output / index_name,
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    head_config = dict(serving_config)
    head_config.update(
        {
            "architectures": ["OrthrusDSV4DraftModel"],
            "compress_ratios": ratios,
            "mask_token_id": int(mask_token_id),
            "pard_token": int(mask_token_id),
            "orthrus_block_size": BLOCK_SIZE,
            "orthrus_head_format": FORMAT_NAME,
            "orthrus_head_format_version": FORMAT_VERSION,
            "orthrus_shared_backbone": True,
            "parallel_drafting": True,
            "sample_from_anchor": False,
        }
    )
    write_json(output / "config.json", head_config)
    training_identity = {
        "path": str(training_base),
        "config_sha256": target_identity(training_base)["config_sha256"],
    }
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "artifact_role": "trained",
        "precision": "bf16",
        "base_model": target_identity(serving_target, base_model_id),
        "training_base": training_identity,
        "training": dict(training_metadata or {}),
        "block_size": BLOCK_SIZE,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "mask_token_id": int(mask_token_id),
        "num_hidden_layers": len(states),
        "compress_ratios": ratios,
        "history_backend": HISTORY_BACKEND,
        "tensor_count": len(schema),
        "total_size": total_size,
        "tensor_schema": schema,
        "shards": shard_manifest(output, weight_map.values()),
        "quantization": {
            "format": "bf16-exchange",
            "source_description_sha256": None,
            "head_calibration_required": True,
            "independently_calibrated": False,
            "scale_origin": "unquantized",
            "shares_target_scale_storage": False,
        },
    }
    write_json(output / "orthrus_head_manifest.json", manifest)
    return output
