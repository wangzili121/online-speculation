"""Attention-only Orthrus draft model for DeepSeek-V4.1.

The draft owns only diffusion attention and index-query projections.  Token
embeddings, mHC, Engram, norms, MoE/MLP and the LM head remain target-owned.
Target SWA/long/index pages are read through :mod:`dsv41_runtime`; no draft
operation has a writable target-cache slot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import weakref

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
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Indexer,
    init_attention_projections,
)
from vllm_ascend.utils import normalize_deepseek_v41_config

from .dsv41_contract import BLOCK_SIZE, HEAD_FORMAT, V41Topology
from .dsv41_native_attention import temporary_block
from .dsv41_runtime import (
    V41NativeDraftState,
    bind_target_history,
    run_native_layer,
)


def _value(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    result = module(value)
    return result[0] if isinstance(result, tuple) else result


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    convert = getattr(config, "to_dict", None)
    if callable(convert):
        result = convert()
        if isinstance(result, dict):
            return result
    raise TypeError("DeepSeek-V4.1 config must provide to_dict()")


def _topology_config(config: Any) -> dict[str, Any]:
    """Restore the on-disk wrapper when vLLM exposes only its text config."""
    raw = _config_dict(config)
    flattened_markers = {
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "candidate_source_layer_id",
    }
    if raw.get("model_type") == "deepseek_v41_text" or flattened_markers <= raw.keys():
        text = dict(raw)
        # SpeculativeConfig's generic DeepSeek compatibility pass may rename
        # the flattened draft view to deepseek_mtp.  These structural markers
        # identify the V4.1 text topology more reliably than that runtime tag.
        text["model_type"] = "deepseek_v41_text"
        return {"model_type": "deepseek_v41", "text_config": text}
    return raw


def _grouped_o_weight(
    weight: torch.Tensor,
    groups: int,
    group_width: int,
    o_lora_rank: int,
) -> torch.Tensor:
    expected = (groups, group_width, o_lora_rank)
    if tuple(weight.shape) == expected:
        return weight
    if tuple(weight.shape) != (groups * o_lora_rank, group_width):
        raise ValueError(f"unexpected wo_a layout {tuple(weight.shape)}, expected {expected}")
    return weight.view(groups, o_lora_rank, group_width).transpose(2, 1).contiguous()


class OrthrusV41Attention(nn.Module):
    """One diffusion attention view, without compressor or cache ownership."""

    def __init__(self, vllm_config: Any, config: Any, topology: V41Topology, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.role = topology.layers[layer_idx]
        self.topology = topology
        prefix = f"model.layers.{layer_idx}.attn"
        # A trained head is exchanged in BF16 even when the target checkpoint
        # is W8A8.  Target quantization scales are not valid for draft weights.
        init_attention_projections(self, config, None, prefix, True)
        self.indexer = (
            DeepseekV41Indexer(
                config,
                owns_k=False,
                vllm_config=vllm_config,
                prefix=f"{prefix}.indexer",
                compress_ratio=self.role.ratio,
                quant_config=None,
                is_candidate_source=layer_idx == self.role.candidate_source,
            )
            if layer_idx == self.role.index_source
            else None
        )

    def _project_output(self, value: torch.Tensor) -> torch.Tensor:
        """Run V4.1's grouped low-rank O projection without a cache wrapper."""
        import torch_npu

        tokens = value.shape[0]
        group_width = self.n_local_heads * self.head_dim // self.n_local_groups
        grouped = value.view(tokens, self.n_local_groups, group_width)
        weight = _grouped_o_weight(
            self.wo_a.weight,
            self.n_local_groups,
            group_width,
            self.o_lora_rank,
        )
        projected = torch_npu.npu_transpose_batchmatmul(
            grouped,
            weight,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
        return _value(self.wo_b, projected.reshape(tokens, -1))

    def process_weights_after_loading(self) -> None:
        """Keep the grouped O weight in its native hot-path layout."""
        group_width = self.n_local_heads * self.head_dim // self.n_local_groups
        self.wo_a.weight.data = _grouped_o_weight(
            self.wo_a.weight.data,
            self.n_local_groups,
            group_width,
            self.o_lora_rank,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        anchors: torch.Tensor,
        target_model: nn.Module,
        metadata_by_prefix: Mapping[str, Any] | None,
        state: V41NativeDraftState,
        capture: Any = None,
    ) -> tuple[torch.Tensor, V41NativeDraftState]:
        from vllm_ascend.device.device_op import DeviceOperator

        tokens = hidden_states.shape[0]
        batch = anchors.numel()
        if tokens != batch * BLOCK_SIZE or positions.shape != (tokens,):
            raise ValueError("Orthrus V4.1 requires a dense Bx32 query block")

        target_inner = getattr(target_model, "model", target_model)
        target_attention = target_inner.layers[self.layer_idx].self_attn
        layer_name = target_attention.rotary_emb.layername
        if metadata_by_prefix is None:
            from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa

            cos, sin = get_cos_and_sin_dsa(positions, layer_names=layer_name)
            cos, sin = cos[layer_name], sin[layer_name]
        else:
            metadata = target_attention.v41_impl._get_layer_metadata(metadata_by_prefix)
            cos, sin = metadata.rope(layer_name, tokens)

        qr = self.q_norm(_value(self.wq_a, hidden_states))
        query = _value(self.wq_b, qr).view(tokens, self.n_local_heads, self.head_dim)
        temporary = self.kv_norm(_value(self.wkv, hidden_states)).view(tokens, 1, self.head_dim)
        DeviceOperator.apply_partial_rotary_inplace(
            query,
            cos,
            sin,
            start=self.nope_head_dim,
            end=self.head_dim,
        )
        DeviceOperator.apply_partial_rotary_inplace(
            temporary,
            cos,
            sin,
            start=self.nope_head_dim,
            end=self.head_dim,
        )

        query = query.view(batch, BLOCK_SIZE, self.n_local_heads, self.head_dim)
        temporary = temporary.view(batch, BLOCK_SIZE, self.head_dim)
        if metadata_by_prefix is None:
            # Startup profiling precedes allocation of physical target pages.
            output, lse = temporary_block(query, temporary)
            sink = self.attn_sink.view(1, 1, -1, 1)
            output = output.float() * torch.sigmoid(lse.float() - sink)
        else:
            index_query = index_weights = None
            if self.indexer is not None:
                index_query = self.indexer.project_query(qr)
                self.indexer.apply_query_rope(index_query, cos, sin)
                index_weights = self.indexer.project_weights(hidden_states)

            history = bind_target_history(target_model, metadata_by_prefix, self.topology, self.layer_idx, anchors)
            if capture is not None:
                capture(cos, sin, history, state.topk[self.role.index_source])
            output, _, state = run_native_layer(
                self.topology, self.layer_idx, query, temporary, self.attn_sink, history, state,
                index_query=(
                    None
                    if index_query is None
                    else index_query.view(batch, BLOCK_SIZE, self.indexer.n_heads, self.indexer.width)
                ),
                index_weights=(
                    None
                    if index_weights is None
                    else index_weights.view(batch, BLOCK_SIZE, self.indexer.n_heads)
                ),
            )
        output = output.reshape(tokens, self.n_local_heads, self.head_dim).to(hidden_states.dtype)
        DeviceOperator.apply_partial_rotary_inplace(
            output,
            cos,
            sin,
            start=self.nope_head_dim,
            end=self.head_dim,
            inverse=True,
        )
        return self._project_output(output), state


class OrthrusV41DraftLayer(nn.Module):
    """Keep the runtime module tree identical to the head tensor contract."""

    def __init__(self, vllm_config: Any, config: Any, topology: V41Topology, layer_idx: int) -> None:
        super().__init__()
        self.attn = OrthrusV41Attention(vllm_config, config, topology, layer_idx)

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, V41NativeDraftState]:
        return self.attn(*args, **kwargs)


class OrthrusV41Model(nn.Module):
    """Full Orthrus draft pass with a target-owned V4.1 backbone."""

    def __init__(self, vllm_config: Any, config: Any, topology: V41Topology) -> None:
        super().__init__()
        self.config = config
        self.topology = topology
        self.hc_mult = int(config.hc_mult)
        self.embed_tokens = PPMissingLayer()
        self.layers = nn.ModuleList(
            OrthrusV41DraftLayer(vllm_config, config, topology, layer)
            for layer in range(config.num_hidden_layers)
        )
        self._target_ref: weakref.ReferenceType[nn.Module] | None = None
        self._lookback_token_ids: torch.Tensor | None = None
        self.capture_last_layer = False
        self.last_layer_snapshot = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        inner = getattr(target_model, "model", None)
        if inner is None or len(inner.layers) != len(self.layers):
            raise ValueError("draft and target V4.1 layer counts differ")
        if getattr(inner, "use_sequence_parallel", False):
            raise NotImplementedError("Orthrus V4.1 currently requires sequence parallel to be disabled")
        self._target_ref = weakref.ref(target_model)

    def set_lookback_token_ids(self, value: torch.Tensor | None) -> None:
        self._lookback_token_ids = value

    def process_weights_after_loading(self) -> None:
        for layer in self.layers:
            layer.attn.process_weights_after_loading()

    def _target(self) -> nn.Module:
        target = self._target_ref() if self._target_ref is not None else None
        if target is None:
            raise RuntimeError("Orthrus draft has not been bound to the V4.1 target")
        return target

    @staticmethod
    def _anchors(positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() % BLOCK_SIZE:
            raise ValueError("Orthrus V4.1 positions are not a dense Bx32 block")
        rows = positions.view(-1, BLOCK_SIZE)
        # DFlash offset zero is the known bonus/anchor token at position equal
        # to the committed token count. The scheduler can clamp the trailing
        # positions to the context limit; those rows are never verified.
        return rows[:, 0].to(torch.int32)

    def _prepare_engram(
        self,
        target_inner: nn.Module,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        profiling: bool = False,
    ) -> tuple[Mapping[int, torch.Tensor], torch.Tensor]:
        if not getattr(target_inner, "has_engram", False):
            return {}, torch.empty(0, dtype=torch.bool, device=input_ids.device)
        lookback = self._lookback_token_ids
        if lookback is None and not profiling:
            raise RuntimeError("V4.1 Engram requires request lookback tokens for the Orthrus block")
        batch = input_ids.numel() // BLOCK_SIZE
        if lookback is not None and lookback.shape[0] != batch:
            raise ValueError("Engram lookback rows do not match the Orthrus request batch")
        query_start = torch.arange(batch + 1, dtype=torch.int32, device=input_ids.device) * BLOCK_SIZE
        return target_inner.prepare_engram(
            input_ids,
            positions,
            lookback_token_ids=lookback,
            query_start_loc=query_start,
            force_dummy=profiling,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        target = self._target()
        target_inner = target.model
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            self._anchors(positions)
            positions = torch.arange(BLOCK_SIZE, device=positions.device, dtype=positions.dtype).repeat(
                positions.numel() // BLOCK_SIZE
            )
            input_ids = torch.full_like(input_ids, int(self.config.engram_pad_token_id))
        elif not isinstance(metadata, Mapping):
            raise RuntimeError("Orthrus V4.1 requires target cache metadata")
        anchors = self._anchors(positions)

        lookups, token_mask = self._prepare_engram(target_inner, input_ids, positions, profiling=metadata is None)
        hidden_states = target_inner.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        draft_state = V41NativeDraftState()
        last_target_layer = None

        for i, (draft_attention, target_layer) in enumerate(zip(self.layers, target_inner.layers, strict=True)):
            last_target_layer = target_layer
            if target_layer.engram is not None and token_mask.numel():
                hidden_states = target_layer.engram(
                    hidden_states,
                    lookups[target_layer.layer_idx],
                    token_mask,
                    target_inner.engram_rotation if target_inner.engram_rotated else None,
                )

            residual = hidden_states
            x, post, comb, attn_pre = target_layer.hc_pre(
                hidden_states,
                target_layer.hc_attn_fn,
                target_layer.hc_attn_scale,
                target_layer.hc_attn_base,
                pre_mix,
            )
            x = target_layer.input_layernorm(x)
            capture = None
            if self.capture_last_layer and i == len(self.layers) - 1 and metadata is not None:
                from .dsv41_last_layer import LastLayerReplay

                def capture(cos, sin, history, selection, x=x, residual=residual,
                            pre=attn_pre, post=post, comb=comb):
                    self.last_layer_snapshot = LastLayerReplay.snapshot(
                        x, residual, pre, post, comb, cos, sin, history, selection, self.topology.window,
                    )

            x, draft_state = draft_attention(x, positions, anchors, target, metadata, draft_state, capture)
            hidden_states = target_layer.hc_post(x, residual, post, comb)

            residual = hidden_states
            x, post, comb, pre_mix = target_layer.hc_pre(
                hidden_states,
                target_layer.hc_ffn_fn,
                target_layer.hc_ffn_scale,
                target_layer.hc_ffn_base,
                attn_pre,
            )
            x, x_fp32 = target_layer.rms_norm_cast(x)
            x = target_layer.mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32)
            hidden_states = target_layer.hc_post(x, residual, post, comb)

        if last_target_layer is None:
            raise RuntimeError("V4.1 target has no decoder layers")
        return last_target_layer.hc_collapse(hidden_states, pre_mix)


class OrthrusDSV41DraftModel(nn.Module):
    """vLLM model-registry entry for an ``orthrus-dsv41-head-v1`` artifact."""

    packed_modules_mapping: dict[str, list[str]] = {}
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del prefix
        if get_pp_group().world_size != 1:
            raise NotImplementedError("Orthrus V4.1 currently requires PP=1")
        parallel = vllm_config.parallel_config
        if any(
            int(getattr(parallel, name, 1)) != 1
            for name in ("decode_context_parallel_size", "prefill_context_parallel_size")
        ):
            raise NotImplementedError("Orthrus V4.1 does not yet support DCP or PCP")
        speculative = vllm_config.speculative_config
        if speculative is None or speculative.num_speculative_tokens != BLOCK_SIZE - 1:
            raise ValueError("Orthrus V4.1 requires exactly 31 speculative tokens")
        draft_config = speculative.draft_model_config
        config = normalize_deepseek_v41_config(draft_config.hf_text_config)
        raw = _topology_config(vllm_config.model_config.hf_config)
        topology = V41Topology.from_config(raw)
        if len(topology.layers) != config.num_hidden_layers:
            raise ValueError("Orthrus head and target V4.1 topologies differ")
        head_config = draft_config.hf_config
        if getattr(head_config, "orthrus_head_format", HEAD_FORMAT) != HEAD_FORMAT:
            raise ValueError(f"Orthrus V4.1 requires {HEAD_FORMAT}")
        self.config = config
        self.model = OrthrusV41Model(vllm_config, config, topology)
        self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._target_ref: weakref.ReferenceType[nn.Module] | None = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        self._target_ref = weakref.ref(target_model)
        self.model.bind_shared_backbone(target_model)

    def set_lookback_token_ids(self, value: torch.Tensor | None) -> None:
        self.model.set_lookback_token_ids(value)

    def process_weights_after_loading(self) -> None:
        self.model.process_weights_after_loading()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            raise ValueError("Orthrus V4.1 uses the shared target embedding")
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        target = self._target_ref() if self._target_ref is not None else None
        if target is None:
            raise RuntimeError("Orthrus draft is not bound to the target LM head")
        return self.logits_processor(target.lm_head, target.model.norm(hidden_states))

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def precompute_and_store_context_kv(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # Persistent cache ownership stays entirely with the target model.
        return []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self._load_head_weights(weights, expected_layer=None)

    @torch.no_grad()
    def publish_last_layer(self, weights: Mapping[str, torch.Tensor]) -> None:
        last = len(self.model.layers) - 1
        self._load_head_weights(
            ((f"layers.{last}.attn.{name}", value) for name, value in weights.items()),
            expected_layer=last,
        )

    def live_last_layer_parameters(self):
        return dict(self.model.layers[-1].attn.named_parameters())

    def _load_head_weights(self, weights, *, expected_layer):
        params = dict(self.named_parameters())
        prefix = "model.layers." if expected_layer is None else f"model.layers.{expected_layer}."
        expected = {name for name in params if name.startswith(prefix)}
        loaded: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        local_heads = self.config.num_attention_heads // tp_size
        for source_name, value in weights:
            name = source_name.removeprefix("model.")
            if not name.startswith("layers."):
                continue
            runtime_name = "model." + name
            if runtime_name not in expected:
                continue
            param = params[runtime_name]
            if runtime_name.endswith(".attn.attn_sink"):
                value = value.narrow(0, tp_rank * local_heads, local_heads)
                default_weight_loader(param, value)
            elif runtime_name.endswith(".attn.wo_a.weight") and param.ndim == 3:
                local_groups, group_width, o_lora_rank = param.shape
                start = tp_rank * local_groups * o_lora_rank
                local = value.narrow(0, start, local_groups * o_lora_rank)
                default_weight_loader(
                    param,
                    _grouped_o_weight(local, local_groups, group_width, o_lora_rank),
                )
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, value)
            loaded.add(runtime_name)
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"Orthrus V4.1 head is missing {len(missing)} tensors: {', '.join(missing[:8])}"
            )
        return loaded
