"""DeepSeek-V4 Orthrus diffusion head for vLLM-Ascend 0.26.

The target owns the complete DSV4 backbone and every historical cache. The
draft owns only attention/head parameters and reads SWA/C4/C128 history without
writing it. FIA handles the local window/block and large-page C128 history;
native sparse attention handles C4 and small-page C128. Each exposes an LSE.
"""

from __future__ import annotations

import os
import re
import weakref
from collections.abc import Iterable, Sequence
from types import SimpleNamespace

import torch
from torch import nn
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import PPMissingLayer
# The 0.26 rc1 monolithic package needs its operator registrations completed
# before DeviceOperator imports submodules from the same package.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import dsa_v1 as _dsa_runtime
from vllm_ascend.device.device_op import DeviceOperator
try:
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention
except ImportError:  # 0.26 release image before the package split
    from vllm_ascend.models.deepseek_v4 import DeepseekV4Attention

try:
    from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
except ImportError:  # 0.26 release image before the package split
    from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.ops.dsa import _build_kv_cache
from vllm_ascend.utils import enable_dsa_cp

from .dsv4_hybrid import (
    assert_cache_unchanged,
    cache_checksum,
    c4_history_indices,
    gather_paged_cache,
    history_and_block_mask,
    merge_attention_outputs,
    tail_logical_indices,
)
from .head_contract import BLOCK_SIZE, FORMAT_NAME, NUM_SPECULATIVE_TOKENS
from .trace import instant


_HEAD_WEIGHT = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.(?:attn|self_attn)\."
    r"(?:attn_sink|wq_a|q_norm|q_norm_without_weight|wq_b|wkv|kv_norm|"
    r"wo_a|wo_b|indexer\.(?:wq_b|weights_proj))(?:\.|$)"
)


def _install_orthrus_quant_aliases(quant_config, ratios: list[int]) -> None:
    if quant_config is None:
        return
    description = getattr(quant_config, "quant_description", None)
    if not isinstance(description, dict):
        return
    aliases = {}
    for name, value in description.items():
        normalized = name.removeprefix("model.")
        if _HEAD_WEIGHT.match(normalized):
            aliases["orthrus." + normalized.replace(".attn.", ".self_attn.", 1)] = value
    # The vendor DeepseekV4Attention constructor always creates target-side
    # compressors before Orthrus can detach them. They are absent from the head
    # artifact by contract, so give only those throwaway constructor modules a
    # local FLOAT policy. No compressor survives in named_parameters().
    for layer, ratio in enumerate(ratios):
        if ratio <= 1:
            continue
        roots = [f"orthrus.layers.{layer}.self_attn.compressor"]
        if ratio == 4:
            roots.append(f"orthrus.layers.{layer}.self_attn.indexer.compressor")
        for root in roots:
            aliases[f"{root}.wkv.weight"] = "FLOAT"
            aliases[f"{root}.wgate.weight"] = "FLOAT"
    description.update(aliases)


def _alias_cache(module: object | None, target_name: str) -> None:
    if module is not None:
        setattr(module, "kv_sharing_target_layer_name", target_name)


def _linear(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    output = module(value)
    return output[0] if isinstance(output, tuple) else output


def _request_metadata(metadata):
    helper = getattr(_dsa_runtime, "_require_req_metadata", None)
    if helper is not None:
        return helper(metadata)
    for name in ("prefill", "decode"):
        request = getattr(metadata, name, None)
        if request is not None:
            return request
    raise RuntimeError("DSV4 metadata has neither a prefill nor decode request view")


def _metadata_for_target(target_wrapper, context_metadata, compress_ratio: int):
    """Normalize the release and nightly DSA metadata layouts."""
    prefix = target_wrapper.prefix
    entries = [
        value
        for name, value in sorted(context_metadata.items())
        if name.startswith(prefix)
    ]
    expected = 5 if compress_ratio == 4 else 3 if compress_ratio == 128 else 1
    if len(entries) != expected:
        raise RuntimeError(
            f"Orthrus expected {expected} target metadata entries under {prefix!r}, "
            f"got {len(entries)}"
        )
    if compress_ratio == 4:
        attention, _, _, indexer, swa = entries
    elif compress_ratio == 128:
        attention, _, swa = entries
        indexer = None
    else:
        (swa,) = entries
        attention = None
        indexer = None
    return SimpleNamespace(attention=attention, swa=swa, indexer=indexer)


def _query_readonly(
    draft_impl,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_a: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project a diffusion query without constructing or updating draft KV."""
    import torch_npu

    if q_a is None:
        q_a = _linear(draft_impl.wq_a, hidden_states)
    if _dsa_runtime._is_w8a8_dynamic(draft_impl.wq_b):
        qr, qr_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
            q_a, draft_impl.q_norm.weight, epsilon=draft_impl.eps
        )
        query = torch_npu.npu_quant_matmul(
            qr,
            draft_impl.wq_b.weight,
            draft_impl.wq_b.weight_scale,
            pertoken_scale=qr_scale,
            bias=draft_impl.wq_b.bias,
            output_dtype=hidden_states.dtype,
        ).unflatten(-1, (draft_impl.n_local_heads, draft_impl.head_dim))
    else:
        qr = _linear(draft_impl.q_norm, q_a)
        query = _linear(draft_impl.wq_b, qr).unflatten(
            -1, (draft_impl.n_local_heads, draft_impl.head_dim)
        )
        qr_scale = None
    query = DeviceOperator.apply_dsa_q_rms(
        query, draft_impl.eps, draft_impl.q_norm_without_weight
    )
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        query.unsqueeze(1),
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
    )
    return query, qr, qr_scale


def _qkv_readonly(draft_impl, hidden_states, cos, sin):
    """Share input quantization across Q/KV, without a target-cache write."""
    import torch_npu

    if (_dsa_runtime._is_w8a8_dynamic(draft_impl.wq_a)
            and _dsa_runtime._is_w8a8_dynamic(draft_impl.wkv)):
        hs_int8, hs_scale = torch_npu.npu_dynamic_quant(hidden_states)

        def project(module):
            return torch_npu.npu_quant_matmul(
                hs_int8, module.weight, module.weight_scale,
                pertoken_scale=hs_scale, bias=module.bias,
                output_dtype=hidden_states.dtype,
            )

        q_a, kv = project(draft_impl.wq_a), project(draft_impl.wkv)
    else:
        q_a = _linear(draft_impl.wq_a, hidden_states)
        kv = _linear(draft_impl.wkv, hidden_states)
    query, qr, qr_scale = _query_readonly(
        draft_impl, hidden_states, cos, sin, q_a=q_a
    )
    kv = draft_impl.kv_norm(kv).view(-1, 1, draft_impl.head_dim)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        kv.unsqueeze(1), cos, sin, rotary_mode="interleave",
        partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
    )
    return query, qr, qr_scale, kv


class _NoTargetState(nn.Module):
    """Placeholder proving that target compressor state is not in the head."""

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Orthrus attempted to execute a target-only compressor")


def _normalize_fia_lse(lse: torch.Tensor, batch: int, queries: int) -> torch.Tensor:
    """Normalize Ascend FIA BSND LSE to ``[B,Q,H,1]``."""
    if lse.ndim == 4 and lse.shape[0] == batch and lse.shape[2] == queries:
        return lse.transpose(1, 2)
    if lse.ndim == 4 and lse.shape[:2] == (batch, queries):
        return lse
    raise RuntimeError(f"unexpected FIA LSE shape: {tuple(lse.shape)}")


def _fia_segment(
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_lengths: Sequence[int] | torch.Tensor,
    key_lengths: Sequence[int] | torch.Tensor,
    scale: float,
    *,
    allow_empty: bool = False,
    block_table: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    if query.ndim != 4 or key_value.ndim != 4:
        raise ValueError("Orthrus FIA expects BSND query and key/value")
    kv_heads = key_value.shape[2]
    block_size = key_value.shape[1] if block_table is not None else 0
    # FIA paged KV is [blocks, heads, block_size, dim]. DSV4 has one
    # KV head, so this transpose only changes the view, without copying KV.
    if block_table is not None:
        key_value = key_value.transpose(1, 2)
    key_value = key_value.contiguous()
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        key_value,
        key_value,
        num_heads=query.shape[2],
        num_key_value_heads=kv_heads,
        input_layout="BSND",
        sparse_mode=0,
        scale=scale,
        atten_mask=atten_mask,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths=query_lengths,
        actual_seq_lengths_kv=key_lengths,
        softmax_lse_flag=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Ascend FIA did not return output and LSE")
    output, lse = result[:2]
    lse = _normalize_fia_lse(lse, query.shape[0], query.shape[1])
    empty = None
    if allow_empty and torch.is_tensor(key_lengths):
        empty = key_lengths.to(device=output.device) == 0
    elif not torch.is_tensor(key_lengths) and any(length == 0 for length in key_lengths):
        empty = torch.tensor(
            [length == 0 for length in key_lengths], dtype=torch.bool, device=output.device
        )
    if empty is not None:
        output = torch.where(
            empty.view(-1, 1, 1, 1), torch.zeros_like(output), output
        )
        lse = torch.where(
            empty.view(-1, 1, 1, 1),
            torch.full_like(lse, float("-inf")),
            lse,
        )
    return output, lse


def _swa_block_segment(query, history, local, lengths, scale):
    # Both branches use the same query and KV format. One softmax avoids two
    # FIA launches and an intermediate normalized output; only padding is masked.
    kv = torch.cat((history, local), dim=1)
    mask = history_and_block_mask(lengths, history.shape[1], local.shape[1])
    batch = query.shape[0]
    return _fia_segment(
        query, kv, [query.shape[1]] * batch, [kv.shape[1]] * batch, scale,
        atten_mask=mask,
    )


def _c128_segment(query, cache, block_table, lengths, maximum, scale, *, backend=None):
    backend = backend or os.environ.get("ORTHRUS_C128_BACKEND", "auto")
    if backend not in {"auto", "sparse", "gather"}:
        raise ValueError("ORTHRUS_C128_BACKEND must be auto, sparse or gather")
    # FIA supports large pages directly; sparse attention also reads the
    # smaller DSV4 pages without materializing a contiguous history buffer.
    if backend == "auto" and 128 <= cache.shape[1] <= 1024 and cache.shape[1] % 128 == 0:
        return _fia_segment(
            query, cache, [BLOCK_SIZE] * query.shape[0], lengths, scale,
            allow_empty=True, block_table=block_table,
        )
    if backend != "gather":
        width = max(128, (maximum + 127) // 128 * 128)
        offsets = torch.arange(width, device=lengths.device, dtype=torch.int32)
        indices = torch.where(offsets[None, None, :] < lengths[:, None, None], offsets, -1)
        indices = indices.expand(-1, query.shape[1], -1).contiguous()
        return _sparse_history_segment(query, cache, block_table, indices, lengths, scale)
    offsets = torch.arange(max(maximum, 1), device=lengths.device)
    indices = torch.where(offsets[None, :] < lengths[:, None], offsets[None, :], -1)
    kv = gather_paged_cache(cache, block_table, indices)
    return _fia_segment(
        query, kv, [BLOCK_SIZE] * query.shape[0], lengths, scale, allow_empty=True,
    )


def _c4_segment(query, cache, block_table, indices, lengths, scale, *, validate=False):
    batch, width, heads, dim = query.shape
    backend = os.environ.get("ORTHRUS_C4_BACKEND", "sparse")
    if backend == "gather":
        counts = (indices >= 0).sum(-1)
        kv = gather_paged_cache(cache, block_table, indices, validate=validate)
        out, lse = _fia_segment(
            query.reshape(batch * width, 1, heads, dim),
            kv.reshape(batch * width, indices.shape[-1], 1, dim),
            [1] * (batch * width), counts.flatten(), scale, allow_empty=True,
        )
        return out.view_as(query), lse.view(batch, width, heads, 1)
    if backend != "sparse":
        raise ValueError("ORTHRUS_C4_BACKEND must be sparse or gather")
    if validate:
        logical_blocks = indices.clamp_min(0).long() // cache.shape[1]
        if int(logical_blocks.max()) >= block_table.shape[1]:
            raise ValueError("C4 index exceeds block table capacity")
        batch_ids = torch.arange(batch, device=indices.device)[:, None, None]
        physical = block_table[batch_ids, logical_blocks]
        if bool(((indices >= 0) & ((physical < 0) | (physical >= cache.shape[0]))).any()):
            raise ValueError("C4 index maps to an invalid physical cache block")
    return _sparse_history_segment(query, cache, block_table, indices, lengths, scale)


def _sparse_history_segment(query, cache, block_table, indices, lengths, scale):
    import torch_npu

    batch, width, heads, _ = query.shape
    counts = (indices >= 0).sum(-1)

    # The MLA API requires a separate RoPE pair. DSV4 already includes RoPE in
    # its 512-d query/key, so a zero auxiliary query preserves q@k exactly.
    # key_rope is a strided read-only view, not another KV allocation.
    auxiliary_query = torch.zeros(batch, width, heads, 64, device=query.device, dtype=query.dtype)
    out, maximum, total = torch_npu.npu_sparse_flash_attention(
        query, cache, cache, indices.to(torch.int32).unsqueeze(2), scale,
        block_table=block_table,
        actual_seq_lengths_query=torch.full_like(lengths, width),
        actual_seq_lengths_kv=lengths,
        query_rope=auxiliary_query, key_rope=cache[..., -64:],
        sparse_mode=0, attention_mode=2,
        layout_query="BSND", layout_kv="PA_BSND", return_softmax_lse=True,
    )
    lse = (maximum + total.log()).squeeze(1).unsqueeze(-1)
    empty = (counts == 0)[..., None, None]
    return out.masked_fill(empty, 0), lse.masked_fill(empty, -torch.inf)


class _OrthrusAttentionLayer(nn.Module):
    def __init__(self, vllm_config, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._full_checked_batches: set[int] = set()
        self.compress_ratio = int(config.compress_ratios[layer_idx])
        self.self_attn = DeepseekV4Attention(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=config.rope_parameters[
                "original_max_position_embeddings"
            ],
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"orthrus.layers.{layer_idx}.self_attn",
        )
        self._register_target_cache_aliases()
        self._drop_target_only_modules()
        self._target_attention_ref: weakref.ReferenceType[nn.Module] | None = None

    @property
    def _target_prefix(self) -> str:
        return f"model.layers.{self.layer_idx}.self_attn"

    def _register_target_cache_aliases(self) -> None:
        attention = self.self_attn
        wrapper = attention.dsa_attn
        _alias_cache(wrapper.swa_cache_layer, f"{self._target_prefix}.swa_cache")
        if self.compress_ratio > 1:
            _alias_cache(wrapper.dsa_attn, f"{self._target_prefix}.attn")
            assert attention.compressor is not None
            _alias_cache(
                attention.compressor.state_cache,
                f"{self._target_prefix}.compressor.state_cache",
            )
        if self.compress_ratio == 4:
            assert attention.indexer is not None
            assert attention.indexer.compressor is not None
            _alias_cache(attention.indexer.k_cache, f"{self._target_prefix}.indexer.k_cache")
            _alias_cache(
                attention.indexer.compressor.state_cache,
                f"{self._target_prefix}.indexer.compressor.state_cache",
            )

    def _drop_target_only_modules(self) -> None:
        """Remove compressor weights while retaining the diffusion-side indexer."""
        attention = self.self_attn
        wrapper = attention.dsa_attn
        impl = wrapper.dsa_attn.impl
        if self.compress_ratio > 1:
            missing = _NoTargetState()
            attention.compressor = missing
            wrapper.compressor = missing
            impl.compressor = missing
            for name in (
                "compressor_ape",
                "compressor_wkv",
                "compressor_wgate",
                "compressor_norm",
            ):
                if hasattr(impl, name):
                    setattr(impl, name, None)
        if self.compress_ratio == 4:
            assert attention.indexer is not None
            attention.indexer.compressor = _NoTargetState()
            for name in (
                "indexer_compress",
                "indexcom_ape",
                "indexcom_wkv",
                "indexcom_wgate",
                "indexcom_norm",
            ):
                if hasattr(impl, name):
                    setattr(impl, name, None)

    def bind_target(self, target_layer: nn.Module) -> None:
        target_attention = target_layer.self_attn
        if int(target_attention.compress_ratio) != self.compress_ratio:
            raise ValueError(
                f"layer {self.layer_idx} compress ratio mismatch: "
                f"head={self.compress_ratio}, target={target_attention.compress_ratio}"
            )
        self._target_attention_ref = weakref.ref(target_attention)

    def _target_attention(self) -> nn.Module:
        target = self._target_attention_ref() if self._target_attention_ref else None
        if target is None:
            raise RuntimeError("Orthrus diffusion head is not bound to the DSV4 target")
        return target

    @staticmethod
    def _c4_topk_readonly(
        indexer,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        target_kv: tuple[torch.Tensor, ...],
        metadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run the diffusion-side indexer over all committed C4 history."""
        import torch_npu

        request = _request_metadata(metadata)
        indexer_wq_b = indexer.wq_b
        if _dsa_runtime._is_w8a8_dynamic(indexer_wq_b) and qr_scale is not None:
            query = torch_npu.npu_quant_matmul(
                qr,
                indexer_wq_b.weight,
                indexer_wq_b.weight_scale,
                pertoken_scale=qr_scale,
                bias=indexer_wq_b.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            query = _linear(indexer_wq_b, qr)
        query = query.view(-1, indexer.n_heads, indexer.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[indexer.head_dim - indexer.rope_head_dim, indexer.head_dim],
        )
        query = _dsa_runtime.rotate_activation(query, metadata.hadamard)
        query, query_scale = DeviceOperator.indexer_quantize_query(query)
        _, _, _, indexer_key, indexer_scale, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
            target_kv, 4
        )
        weights = _linear(indexer.weights_proj, hidden_states) * (
            indexer.softmax_scale * indexer.n_heads**-0.5
        )
        # Each query reads the same committed prefix. The target's causal QLI
        # metadata would shorten the history for the first K-1 query rows.
        qli_metadata = getattr(request, "orthrus_qli_metadata", None)
        if qli_metadata is None:
            qli_metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=request.query_start_loc[1:],
                actual_seq_lengths_key=request.seq_lens,
                num_heads_q=indexer.n_heads,
                num_heads_k=1,
                head_dim=indexer.head_dim,
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=request.seq_lens.shape[0],
                max_seqlen_q=BLOCK_SIZE,
                max_seqlen_k=request.max_seq_lens,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=indexer.index_topk,
                sparse_mode=0,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(query.device),
            )
            request.orthrus_qli_metadata = qli_metadata
        topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query=query,
            key=indexer_key,
            weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(
                query_scale
            ),
            key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(
                indexer_scale
            ),
            actual_seq_lengths_query=request.query_start_loc[1:],
            actual_seq_lengths_key=request.seq_lens,
            block_table=request.block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=indexer.index_topk,
            sparse_mode=0,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk

    def _history_segments(
        self,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        target_kv: tuple[torch.Tensor, ...],
        layer_metadata,
        layer_name: str,
        local_kv: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        draft_impl = self.self_attn.dsa_attn.dsa_attn.impl
        validate_cache = os.environ.get("ORTHRUS_VERIFY_CACHE_CHECKSUM", "0") in {"1", "full"}
        compress_cache, swa_cache, state_cache, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
            target_kv, self.compress_ratio
        )
        swa_req = _request_metadata(layer_metadata.swa)
        seq_lens = swa_req.seq_lens
        batch = int(seq_lens.shape[0])
        if query.shape[0] != batch * BLOCK_SIZE:
            raise RuntimeError(
                f"Orthrus requires a dense Bx{BLOCK_SIZE} query block, got "
                f"{query.shape[0]} tokens for {batch} requests"
            )
        query_bsnd = query.view(batch, BLOCK_SIZE, query.shape[1], query.shape[2])
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []

        swa_indices = tail_logical_indices(seq_lens, draft_impl.window_size)
        swa_lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, draft_impl.window_size))
        swa_kv = gather_paged_cache(
            swa_cache,
            swa_req.block_table,
            swa_indices,
            validate=validate_cache,
        )
        out, lse = _swa_block_segment(
            query_bsnd,
            swa_kv,
            local_kv.view(batch, BLOCK_SIZE, 1, local_kv.shape[-1]),
            swa_lengths,
            draft_impl.softmax_scale,
        )
        outputs.append(out)
        lses.append(lse)

        if self.compress_ratio > 1:
            common = layer_metadata.attention
            assert common is not None and compress_cache is not None and state_cache is not None
            compressed_req = _request_metadata(common)
            if self.compress_ratio == 4:
                assert draft_impl.indexer is not None
                assert layer_metadata.indexer is not None
                indexer_metadata = layer_metadata.indexer
                if hasattr(indexer_metadata, "compressor"):
                    indexer_metadata = indexer_metadata.compressor.cache
                indexer_req = _request_metadata(indexer_metadata)
                topk = self._c4_topk_readonly(
                    draft_impl.indexer,
                    hidden_states,
                    qr,
                    qr_scale,
                    target_kv,
                    layer_metadata.indexer,
                    indexer_req.cos[layer_name][: hidden_states.shape[0]],
                    indexer_req.sin[layer_name][: hidden_states.shape[0]],
                )
                if topk is None:
                    raise RuntimeError("C4 diffusion-side indexer returned no candidates")
                if topk.ndim == 3:
                    if topk.shape[1] != 1:
                        raise RuntimeError(f"unexpected C4 top-k shape: {tuple(topk.shape)}")
                    topk = topk[:, 0]
                topk, _ = c4_history_indices(topk.view(batch, BLOCK_SIZE, -1), seq_lens)
                out, lse = _c4_segment(
                    query_bsnd,
                    compress_cache,
                    compressed_req.block_table,
                    topk,
                    torch.div(seq_lens, 4, rounding_mode="floor"),
                    draft_impl.softmax_scale,
                    validate=validate_cache,
                )
                outputs.append(out)
                lses.append(lse)
            else:
                compressed_lengths = torch.div(seq_lens, 128, rounding_mode="floor")
                maximum = getattr(compressed_req, "max_seq_lens", None)
                maximum = (int(maximum) // 128 if maximum is not None
                           else compressed_req.block_table.shape[1] * compress_cache.shape[1])
                out, lse = _c128_segment(
                    query_bsnd,
                    compress_cache,
                    compressed_req.block_table,
                    compressed_lengths,
                    maximum,
                    draft_impl.softmax_scale,
                )
                outputs.append(out)
                lses.append(lse)
        return outputs, lses

    def _attention_readonly(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        context = get_forward_context()
        if context is None or context.attn_metadata is None:
            return torch.zeros_like(hidden_states)
        target_attention = self._target_attention()
        target_wrapper = target_attention.dsa_attn
        target_impl = target_wrapper.dsa_attn.impl
        draft_impl = self.self_attn.dsa_attn.dsa_attn.impl
        layer_name = target_wrapper.dsa_attn.layer_name
        if hasattr(target_impl, "_get_layer_metadata"):
            layer_metadata = target_impl._get_layer_metadata(
                layer_name, context.attn_metadata
            )
        else:
            layer_metadata = _metadata_for_target(
                target_wrapper, context.attn_metadata, self.compress_ratio
            )
        target_kv = _build_kv_cache(target_wrapper, context)
        _, swa_cache, _, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
            target_kv, self.compress_ratio
        )
        common = layer_metadata.attention or layer_metadata.swa
        req = _request_metadata(common)
        swa_req = _request_metadata(layer_metadata.swa)
        num_tokens = hidden_states.shape[0]
        cos = req.cos[layer_name][:num_tokens]
        sin = req.sin[layer_name][:num_tokens]
        guard_mode = os.environ.get("ORTHRUS_VERIFY_CACHE_CHECKSUM", "0")
        debug = guard_mode in {"1", "full"}
        batch = num_tokens // BLOCK_SIZE
        full_check = guard_mode == "full" and batch not in self._full_checked_batches
        checksums = []
        if debug:
            torch.npu.synchronize()
            anchors = positions.view(batch, BLOCK_SIZE)[:, 0]
            if not torch.equal(anchors, swa_req.seq_lens.to(anchors.dtype)):
                raise RuntimeError("Orthrus history length does not match anchor position")
            for index, cache in enumerate(target_kv):
                if torch.is_tensor(cache):
                    checksums.append((index, cache_checksum(cache, full=full_check)))

        query, qr, qr_scale, local_kv = _qkv_readonly(
            draft_impl, hidden_states, cos, sin
        )
        batch = num_tokens // BLOCK_SIZE
        outputs, lses = self._history_segments(
            query, hidden_states, qr, qr_scale, target_kv, layer_metadata, layer_name,
            local_kv,
        )
        sink_lse = draft_impl.attn_sink.view(1, 1, -1, 1).expand(
            batch, BLOCK_SIZE, -1, -1
        )
        outputs.append(torch.zeros_like(outputs[0]))
        lses.append(sink_lse)
        merged = merge_attention_outputs(outputs, lses)
        merged = merged.reshape(num_tokens, query.shape[1], query.shape[2])
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            merged.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
        )
        output = torch.empty_like(hidden_states)
        draft_impl._forward_o_proj(merged, output)
        if debug:
            torch.npu.synchronize()
            for index, before in checksums:
                assert_cache_unchanged(before, target_kv[index], f"layer-{self.layer_idx}:{index}")
            if full_check:
                self._full_checked_batches.add(batch)
        instant("orthrus.dsv4.attention", {
            "layer": self.layer_idx, "compress_ratio": self.compress_ratio,
            "batch": batch, "block_size": BLOCK_SIZE,
            "query_shape": list(query.shape), "segments": len(outputs),
            "cache_guard": "full" if full_check else "sampled" if debug else "off",
            "checked_elements": sum(before.numel() for _, before in checksums),
        })
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_layer: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states.clone()
        hidden_states, post, comb = target_layer.hc_pre(
            hidden_states,
            target_layer.hc_attn_fn,
            target_layer.hc_attn_scale,
            target_layer.hc_attn_base,
        )
        hidden_states = target_layer.input_layernorm(hidden_states)
        hidden_states = self._attention_readonly(positions, hidden_states)
        hidden_states = target_layer.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = target_layer.hc_pre(
            hidden_states,
            target_layer.hc_ffn_fn,
            target_layer.hc_ffn_scale,
            target_layer.hc_ffn_base,
        )
        hidden_states = target_layer.post_attention_layernorm(hidden_states)
        hidden_states = target_layer.mlp(hidden_states, input_ids)
        return target_layer.hc_post(hidden_states, residual, post, comb)


class _OrthrusDeepseekV4Model(nn.Module):
    def __init__(self, vllm_config, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.embed_tokens = PPMissingLayer()
        self.layers = nn.ModuleList(
            [_OrthrusAttentionLayer(vllm_config, config, index) for index in range(config.num_hidden_layers)]
        )
        self._target_model_ref: weakref.ReferenceType[nn.Module] | None = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        target_inner = getattr(target_model, "model", None)
        target_layers = getattr(target_inner, "layers", None)
        if target_inner is None or target_layers is None or len(target_layers) != len(self.layers):
            raise ValueError("draft/target DSV4 layer count mismatch")
        self._target_model_ref = weakref.ref(target_inner)
        for draft_layer, target_layer in zip(self.layers, target_layers, strict=True):
            draft_layer.bind_target(target_layer)
        leaked = [
            name
            for name, _ in self.named_parameters()
            if ".compressor." in name
        ]
        if leaked:
            raise RuntimeError(
                "target compressor parameters leaked into Orthrus draft: "
                + ", ".join(leaked[:8])
            )

    def _target_model(self) -> nn.Module:
        target = self._target_model_ref() if self._target_model_ref else None
        if target is None:
            raise RuntimeError("Orthrus draft is not bound to the live DSV4 target")
        return target

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        names = []
        for layer in self.layers:
            prefix = layer._target_prefix
            names.append(f"{prefix}.swa_cache")
            if layer.compress_ratio > 1:
                names.extend((f"{prefix}.attn", f"{prefix}.compressor.state_cache"))
            if layer.compress_ratio == 4:
                names.extend(
                    (
                        f"{prefix}.indexer.k_cache",
                        f"{prefix}.indexer.compressor.state_cache",
                    )
                )
        return names

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        target = self._target_model()
        hidden_states = target.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for draft_layer, target_layer in zip(self.layers, target.layers, strict=True):
            hidden_states = draft_layer(positions, hidden_states, target_layer, input_ids)
        return target.hc_head(
            hidden_states, target.hc_head_fn, target.hc_head_scale, target.hc_head_base
        )


class OrthrusDSV4DraftModel(DSparkDeepseekV4ForCausalLM):
    """Attention-only K32 diffusion head with a shared DSV4 backbone."""

    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)
        # The decorated DSpark parent installs its compile wrapper in __call__,
        # but invoking its constructor would allocate a second draft backbone.
        # Orthrus is intentionally eager, so initialize the inherited wrapper
        # contract without constructing the parent model.
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.do_not_compile = True
        if get_pp_group().world_size != 1:
            raise NotImplementedError("DSV4 Orthrus currently requires PP=1")
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if additional_config.get("enable_dsa_cp"):
            raise NotImplementedError(
                "DSV4 Orthrus does not yet support DSA context parallel; "
                "remove enable_dsa_cp and use the TP-only correctness backend"
            )
        if additional_config.get("enable_flashcomm1"):
            raise NotImplementedError(
                "DSV4 Orthrus does not yet support FlashComm1 sequence parallel; "
                "remove enable_flashcomm1 and use the TP-only correctness backend"
            )
        speculative = vllm_config.speculative_config
        if speculative is None or speculative.num_speculative_tokens != NUM_SPECULATIVE_TOKENS:
            raise ValueError("DSV4 Orthrus requires exactly 31 speculative tokens")
        config = speculative.draft_model_config.hf_config
        if getattr(config, "orthrus_head_format", None) != FORMAT_NAME:
            raise ValueError("DSV4 Orthrus requires an orthrus-dsv4-head-v2 artifact")
        ratios = list(getattr(config, "compress_ratios", []))[: config.num_hidden_layers]
        if len(ratios) != config.num_hidden_layers or any(r not in (0, 1, 4, 128) for r in ratios):
            raise ValueError("invalid DSV4 hybrid compress_ratios in diffusion head")
        if getattr(config, "sample_from_anchor", None) is not False:
            raise ValueError("Orthrus K32 requires sample_from_anchor=false")
        _install_orthrus_quant_aliases(vllm_config.quant_config, ratios)
        self.config = config
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        self.model = _OrthrusDeepseekV4Model(vllm_config, config)
        self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        self.num_expert_groups = getattr(config, "n_group", 1)
        self._target_ref: weakref.ReferenceType[nn.Module] | None = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        self._target_ref = weakref.ref(target_model)
        self.model.bind_shared_backbone(target_model)
        for name in (
            "expert_weights",
            "moe_layers",
            "moe_mlp_layers",
            "num_expert_groups",
            "num_moe_layers",
            "num_logical_experts",
            "num_physical_experts",
            "num_local_physical_experts",
            "num_routed_experts",
            "num_shared_experts",
            "num_redundant_experts",
        ):
            if hasattr(target_model, name):
                setattr(self, name, getattr(target_model, name))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return [False]

    def combine_hidden_states(self, states: torch.Tensor) -> torch.Tensor:
        return states[..., : self.config.hidden_size].contiguous()

    def precompute_and_store_context_kv(self, *args, **kwargs) -> None:
        del args, kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        if inputs_embeds is not None:
            raise ValueError("DSV4 Orthrus does not accept separate inputs_embeds")
        return self.model(input_ids=input_ids, positions=positions)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        target = self._target_ref() if self._target_ref else None
        if target is None:
            raise RuntimeError("Orthrus draft is not bound to the target LM head")
        return self.logits_processor(target.lm_head, target.model.norm(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        for source_name, loaded_weight in weights:
            normalized = source_name.removeprefix("model.")
            if normalized.startswith("orthrus.layers."):
                normalized = normalized.removeprefix("orthrus.")
            if not _HEAD_WEIGHT.match(normalized):
                continue
            name = "model." + normalized.replace(".attn.", ".self_attn.", 1)
            if name.endswith(".scale"):
                name = name.removesuffix(".scale") + ".weight_scale"
            param = params.get(name)
            if param is None:
                continue
            if name.endswith(".attn_sink") and not enable_dsa_cp():
                param.data.copy_(loaded_weight.narrow(0, tp_rank * heads_per_rank, heads_per_rank))
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            loaded.add(name)
        expected = {
            name
            for name in params
            if name.startswith("model.layers.") and _HEAD_WEIGHT.match(name.removeprefix("model."))
        }
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"Orthrus head is missing {len(missing)} parameters: {', '.join(missing[:8])}"
            )
        return loaded


# Kept as an import-time error message for stale deployment scripts.
OrthrusDSparkDeepseekV4ForCausalLM = OrthrusDSV4DraftModel
