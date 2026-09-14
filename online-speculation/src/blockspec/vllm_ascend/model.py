"""Single-card vLLM model with shared AR/draft Orthrus backbone weights."""

from collections.abc import Iterable
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (SupportsEagle, SupportsEagle3,
                                                    SupportsLoRA, SupportsPP)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer,
                                              maybe_prefix)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .proposer import OrthrusDraftBatch
from .sparse_draft import (compact_verifier_queries, recent_page_plan,
                           reuse_vegas_page_plan, vegas_page_plan,
                           verifier_guided_page_plan)
from .trace import instant


class OrthrusAttention(Qwen3Attention):
    """Ordinary vLLM AR attention plus the checkpoint's draft projections."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, rope_parameters,
                 max_position=4096 * 32, head_dim=None, rms_norm_eps=1e-6,
                 qkv_bias=False, cache_config=None, quant_config=None, prefix="",
                 attn_type=AttentionType.DECODER, dual_chunk_attention_config=None):
        super().__init__(
            hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters, max_position=max_position,
            head_dim=head_dim, rms_norm_eps=rms_norm_eps, qkv_bias=qkv_bias,
            cache_config=cache_config, quant_config=quant_config, prefix=prefix,
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config)
        self.qkv_proj_diff = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads,
            self.total_num_kv_heads, bias=qkv_bias, quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_diff")
        self.o_proj_diff = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj_diff")
        self.q_norm_diff = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm_diff = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.capture_target_query = False
        self.last_target_query = None

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if self.capture_target_query:
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, list):
                raise NotImplementedError(
                    "Orthrus verifier-query capture does not support DBO")
            # vLLM runs dummy forwards with no attention metadata while
            # sizing memory and compiling graphs. A real verifier forward
            # updates this buffer before the proposer can consume it.
            if metadata is not None:
                layer_metadata = metadata[self.attn.layer_name]
                self.last_target_query = compact_verifier_queries(
                    q.view(-1, self.num_heads, self.head_dim),
                    layer_metadata.query_start_loc)
        attended = self.attn(q, k, v)
        return self.o_proj(attended)[0]

    def draft_qkv(self, positions, hidden_states):
        qkv, _ = self.qkv_proj_diff(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm_diff(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm_diff(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        # Drafting runs after the target forward has left Ascend's forward
        # context. The native path is mathematically identical and does not
        # depend on that transient runner context.
        q, k = self.rotary_emb.forward_native(positions, q, k)
        return q, k, v

    def draft_output(self, attended):
        return self.o_proj_diff(attended)[0]


class OrthrusDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, cache_config: CacheConfig | None = None,
                 quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk = getattr(config, "dual_chunk_attention_config", None)
        attn_type = (AttentionType.DECODER if getattr(config, "is_causal", True)
                     else AttentionType.ENCODER_ONLY)
        self.self_attn = OrthrusAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states,
                                                                 residual)
        return self.mlp(hidden_states), residual


class OrthrusModel(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix,
                         decoder_layer_type=OrthrusDecoderLayer)


class OrthrusForCausalLM(
        nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3):
    """AR execution stays native; drafting reuses all non-attention modules."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_proj_diff": ["q_proj_diff", "k_proj_diff", "v_proj_diff"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = OrthrusModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.draft_attention_backend = os.environ.get(
            "ORTHRUS_DRAFT_ATTENTION", "native_gqa")
        if self.draft_attention_backend not in {
                "expanded", "native_gqa", "paged_segmented", "paged_pillar"}:
            raise ValueError(
                "ORTHRUS_DRAFT_ATTENTION must be expanded, native_gqa, or "
                "paged_segmented/paged_pillar, got "
                f"{self.draft_attention_backend!r}")
        self.draft_segment_threshold = int(os.environ.get(
            "ORTHRUS_DRAFT_SEGMENT_THRESHOLD", "24576"))
        if self.draft_segment_threshold <= 0:
            raise ValueError("ORTHRUS_DRAFT_SEGMENT_THRESHOLD must be positive")
        self.pillar_policy = os.environ.get(
            "ORTHRUS_PILLAR_POLICY", "verifier")
        if self.pillar_policy not in {
                "recent", "verifier", "vegas", "vegas_cached"}:
            raise ValueError(
                "ORTHRUS_PILLAR_POLICY must be recent, verifier, vegas, "
                "or vegas_cached")
        self.pillar_budget_ratio = float(os.environ.get(
            "ORTHRUS_PILLAR_BUDGET_RATIO", "0.05"))
        self.pillar_min_tokens = int(os.environ.get(
            "ORTHRUS_PILLAR_MIN_TOKENS", "128"))
        self.pillar_max_tokens = int(os.environ.get(
            "ORTHRUS_PILLAR_MAX_TOKENS", "6144"))
        selector_layer = int(os.environ.get(
            "ORTHRUS_PILLAR_SELECTOR_LAYER", "-1"))
        layer_count = len(self.model.layers)
        if selector_layer < 0:
            selector_layer += layer_count
        if not 0 <= selector_layer < layer_count:
            raise ValueError("ORTHRUS_PILLAR_SELECTOR_LAYER is out of range")
        self.pillar_selector_layer = selector_layer
        if self.draft_attention_backend == "paged_pillar":
            capture_layers = (self.model.layers
                              if self.pillar_policy in {
                                  "vegas", "vegas_cached"}
                              else [self.model.layers[
                                  self.pillar_selector_layer]])
            for layer in capture_layers:
                layer.self_attn.capture_target_query = True
        self._vegas_page_cache = {}

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def _cache(layer, table, request_index, history_length):
        cache = layer.self_attn.attn.kv_cache
        if isinstance(cache, torch.Tensor) and cache.ndim > 0 and cache.shape[0] == 2:
            key_cache, value_cache = cache[0], cache[1]
        elif isinstance(cache, (tuple, list)) and len(cache) >= 2:
            key_cache, value_cache = cache[:2]
        else:
            raise RuntimeError("Orthrus drafting requires a bound target paged KV cache")
        block_size = key_cache.shape[1]
        block_count = (history_length + block_size - 1) // block_size
        block_ids = table.block_table.gpu[request_index, :block_count].long()

        def gather(value):
            selected = value.index_select(0, block_ids)
            return selected.reshape(-1, selected.shape[-2], selected.shape[-1])[:history_length]

        return gather(key_cache), gather(value_cache)

    def _draft_attention(self, layer, hidden_states, positions,
                         table, request_indices, history_lengths):
        attention = layer.self_attn
        active, block, _ = hidden_states.shape
        flat = hidden_states.reshape(active * block, -1)
        q, k, v = attention.draft_qkv(positions.reshape(-1), flat)
        q = q.view(active, block, attention.num_heads, attention.head_dim)
        k = k.view(active, block, attention.num_kv_heads, attention.head_dim)
        v = v.view(active, block, attention.num_kv_heads, attention.head_dim)
        rows = []
        native_attention = None
        paged_attention = None
        if self.draft_attention_backend in {
                "native_gqa", "paged_segmented", "paged_pillar"}:
            import torch_npu
            native_attention = torch_npu.npu_fusion_attention_v3
            paged_attention = torch_npu.npu_fused_infer_attention_score
        if (self.draft_attention_backend in {"paged_segmented", "paged_pillar"}
                and (active > 1
                     or max(history_lengths) >= self.draft_segment_threshold)):
            attended = self._batched_segmented_draft_attention(
                attention=attention,
                query=q,
                local_key=k,
                local_value=v,
                table=table,
                request_indices=request_indices,
                history_lengths=history_lengths,
                paged_attention=paged_attention,
                history_block_table=getattr(
                    self, "_pillar_block_table", None),
                history_attention_lengths=getattr(
                    self, "_pillar_history_lengths", None),
            )
            return attention.draft_output(attended.reshape(
                active * block, -1))
        for local, request_index in enumerate(request_indices):
            history_length = history_lengths[local]
            with torch.profiler.record_function("orthrus.draft.kv_gather"):
                old_k, old_v = self._cache(
                    layer, table, request_index, history_length)
            full_k = torch.cat((old_k, k[local]), dim=0)
            full_v = torch.cat((old_v, v[local]), dim=0)
            if self.draft_attention_backend == "native_gqa":
                with torch.profiler.record_function(
                        "orthrus.draft.attention.native_gqa"):
                    attended = native_attention(
                        q[local][None], full_k[None], full_v[None],
                        attention.num_heads, "BSND", scale=attention.scaling,
                        keep_prob=1.0, sparse_mode=0)[0]
                rows.append(attended[0].reshape(block, -1))
            else:
                repeats = attention.num_heads // attention.num_kv_heads
                if repeats > 1:
                    full_k = full_k.repeat_interleave(repeats, dim=1)
                    full_v = full_v.repeat_interleave(repeats, dim=1)
                with torch.profiler.record_function(
                        "orthrus.draft.attention.expanded"):
                    attended = F.scaled_dot_product_attention(
                        q[local].transpose(0, 1)[None],
                        full_k.transpose(0, 1)[None],
                        full_v.transpose(0, 1)[None],
                        is_causal=False, scale=attention.scaling)
                rows.append(attended[0].transpose(0, 1).reshape(block, -1))
        return attention.draft_output(torch.stack(rows).reshape(active * block, -1))

    @staticmethod
    def _batched_segmented_draft_attention(
            *, attention, query, local_key, local_value, table,
            request_indices, history_lengths, paged_attention,
            history_block_table=None, history_attention_lengths=None):
        """Attend to paged history and local draft KV in two batched ops."""
        cache = attention.attn.kv_cache
        if (isinstance(cache, torch.Tensor) and cache.ndim > 0
                and cache.shape[0] == 2):
            key_cache, value_cache = cache[0], cache[1]
        elif isinstance(cache, (tuple, list)) and len(cache) >= 2:
            key_cache, value_cache = cache[:2]
        else:
            raise RuntimeError(
                "Orthrus drafting requires a bound target paged KV cache")
        active, query_tokens, num_heads, head_dim = query.shape
        block_size = key_cache.shape[1]
        if history_block_table is None:
            block_count = max(
                (length + block_size - 1) // block_size
                for length in history_lengths)
            block_table = table.block_table.gpu[
                request_indices, :block_count].contiguous()
            attention_lengths = history_lengths
            history_scope = "orthrus.draft.attention.paged_history"
        else:
            block_table = history_block_table
            attention_lengths = history_attention_lengths
            history_scope = "orthrus.draft.attention.pillar_history"
        cumulative = [query_tokens * (index + 1) for index in range(active)]
        common = {
            "query": query.reshape(-1, num_heads, head_dim),
            "input_layout": "TND",
            "actual_seq_lengths": cumulative,
            "num_key_value_heads": attention.num_kv_heads,
            "num_heads": attention.num_heads,
            "scale": attention.scaling,
            "sparse_mode": 0,
            "softmax_lse_flag": True,
        }
        with torch.profiler.record_function(history_scope):
            old_output, old_lse = paged_attention(
                key=key_cache.view(key_cache.shape[0], block_size, -1),
                value=value_cache.view(value_cache.shape[0], block_size, -1),
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=attention_lengths,
                **common,
            )
        with torch.profiler.record_function(
                "orthrus.draft.attention.local"):
            local_output, local_lse = paged_attention(
                key=local_key.reshape(
                    -1, attention.num_kv_heads, attention.head_dim),
                value=local_value.reshape(
                    -1, attention.num_kv_heads, attention.head_dim),
                actual_seq_lengths_kv=cumulative,
                **common,
            )
        with torch.profiler.record_function("orthrus.draft.attention.merge"):
            merged_lse = torch.logaddexp(old_lse, local_lse)
            merged = (
                old_output.float() * torch.exp(old_lse - merged_lse)
                + local_output.float() * torch.exp(local_lse - merged_lse)
            )
            return merged.to(old_output.dtype).view_as(query)

    @torch.inference_mode()
    def orthrus_propose(self, *, sampled_token_ids, input_batch,
                        scheduler_output, common_attn_metadata,
                        accepted_token_counts=None, next_token_ids=None,
                        history_lengths=None, draft_request_indices=None):
        del scheduler_output, common_attn_metadata
        if get_pp_group().world_size != 1:
            raise NotImplementedError("the first Orthrus baseline is single-card only")
        batch_size = len(input_batch.req_ids)
        vocabulary = self.config.vocab_size
        async_tokens = torch.is_tensor(sampled_token_ids)
        if async_tokens:
            if sampled_token_ids.ndim != 2 or sampled_token_ids.shape[0] != batch_size:
                raise ValueError(
                    "padded sampled tokens must have shape [requests,tokens]")
            if accepted_token_counts is None:
                accepted_token_counts = (sampled_token_ids >= 0).sum(dim=1)
            active_indices = (list(draft_request_indices)
                              if draft_request_indices is not None else
                              torch.nonzero(accepted_token_counts > 0,
                                            as_tuple=False).flatten().tolist())
            if next_token_ids is None:
                gather_indices = (accepted_token_counts - 1).clamp_min(0).long()
                next_token_ids = sampled_token_ids.gather(
                    1, gather_indices[:, None]).squeeze(1)
        else:
            active_indices = (list(draft_request_indices)
                              if draft_request_indices is not None else
                              [index for index, row in enumerate(sampled_token_ids)
                               if row])
        if not active_indices:
            empty = torch.empty((batch_size, 31, vocabulary),
                                device=self.model.embed_tokens.weight.device)
            token_ids = (torch.full((batch_size, 31), -1, dtype=torch.long,
                                    device=empty.device)
                         if async_tokens else [[] for _ in range(batch_size)])
            return OrthrusDraftBatch(
                token_ids, empty, accepted_token_counts, next_token_ids, [])
        for index in active_indices:
            if (float(input_batch.temperature_cpu[index]) != 1.0
                    or int(input_batch.top_k_cpu[index]) not in (-1, 0, vocabulary)
                    or float(input_batch.top_p_cpu[index]) != 1.0):
                raise ValueError("Orthrus baseline requires T=1, top_k=0, top_p=1")
            if input_batch.generators.get(index) is None:
                raise ValueError("Orthrus baseline requires a request-local seed")
        device = self.model.embed_tokens.weight.device
        mask_id = int(self.config.mask_token_id)
        if async_tokens:
            assert next_token_ids is not None
            anchors = next_token_ids[active_indices].to(device=device)
        else:
            anchors = torch.tensor([
                int(input_batch.token_ids_cpu[
                    index, int(input_batch.num_tokens_no_spec[index]) - 1])
                for index in active_indices
            ], dtype=torch.long, device=device)
        tokens = torch.full((len(active_indices), 32), mask_id,
                            dtype=torch.long, device=device)
        tokens[:, 0] = anchors
        if async_tokens and history_lengths is None:
            # The async scheduler has only advanced the CPU-side sequence by
            # one optimistic placeholder. Correct the draft history with the
            # verifier's actual accepted-token count. This one D2H is kept in
            # the first prototype for correctness; removing it is measured as
            # a separate follow-up rather than hidden in the scheduler A/B.
            counts_cpu = accepted_token_counts.to("cpu").tolist()
            history_lengths = [
                int(input_batch.num_tokens_no_spec[index])
                + int(counts_cpu[index]) - 2
                for index in active_indices
            ]
        elif not async_tokens:
            history_lengths = [int(input_batch.num_tokens_no_spec[index]) - 1
                               for index in active_indices]
        positions = torch.stack([
            torch.arange(length, length + 32, device=device)
            for length in history_lengths])
        self._pillar_block_table = None
        self._pillar_history_lengths = None
        table = input_batch.block_table[0]
        active_block_table = table.block_table.gpu[active_indices]
        if self.draft_attention_backend == "paged_pillar":
            if self.pillar_policy != "vegas":
                selector_attention = self.model.layers[
                    self.pillar_selector_layer].self_attn
                target_query = selector_attention.last_target_query
                if target_query is None:
                    raise RuntimeError(
                        "paged_pillar requires preceding verifier queries")
                active_tensor = torch.tensor(
                    active_indices, dtype=torch.long,
                    device=target_query.device)
                final_query = target_query.index_select(
                    0, active_tensor)[:, -1]
                repeats = (selector_attention.num_heads
                           // selector_attention.num_kv_heads)
                query_by_kv_head = final_query.view(
                    len(active_indices), selector_attention.num_kv_heads,
                    repeats, selector_attention.head_dim).mean(dim=2)
                selector_cache = selector_attention.attn.kv_cache
                key_cache = selector_cache[0]
                with torch.profiler.record_function(
                        "orthrus.draft.pillar.select"):
                    if self.pillar_policy == "verifier":
                        plan = verifier_guided_page_plan(
                            query_by_kv_head, key_cache, active_block_table,
                            history_lengths,
                            budget_ratio=self.pillar_budget_ratio,
                            min_tokens=self.pillar_min_tokens,
                            max_tokens=self.pillar_max_tokens,
                            dense_below_tokens=self.draft_segment_threshold)
                    else:
                        plan = recent_page_plan(
                            active_block_table, history_lengths,
                            key_cache.shape[1],
                            budget_ratio=self.pillar_budget_ratio,
                            min_tokens=self.pillar_min_tokens,
                            max_tokens=self.pillar_max_tokens,
                            dense_below_tokens=self.draft_segment_threshold)
                self._pillar_block_table = plan.block_table
                self._pillar_history_lengths = plan.sequence_lengths
                instant("orthrus.draft.pillar.plan", {
                    "policy": self.pillar_policy,
                    "budget_ratio": self.pillar_budget_ratio,
                    "history_tokens": history_lengths,
                    "attended_history_tokens": plan.sequence_lengths,
                    "selected_blocks": plan.selected_blocks,
                    "total_blocks": plan.total_blocks,
                    "request_ids": [input_batch.req_ids[index]
                                    for index in active_indices],
                })
        with torch.profiler.record_function("orthrus.draft.embedding"):
            hidden_states = self.model.embed_input_ids(tokens).reshape(
                -1, self.config.hidden_size)
        residual = None
        vegas_active_tensor = None
        if self.pillar_policy in {"vegas", "vegas_cached"}:
            vegas_active_tensor = torch.tensor(
                active_indices, dtype=torch.long, device=device)
        request_ids = [input_batch.req_ids[index] for index in active_indices]
        live_request_ids = set(input_batch.req_ids)
        self._vegas_page_cache = {
            key: value for key, value in self._vegas_page_cache.items()
            if key[0] in live_request_ids
        }
        for layer_index, layer in enumerate(self.model.layers):
            if residual is None:
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
            else:
                normalized, residual = layer.input_layernorm(hidden_states,
                                                              residual)
            if (self.draft_attention_backend == "paged_pillar"
                    and self.pillar_policy in {"vegas", "vegas_cached"}):
                target_queries = layer.self_attn.last_target_query
                if target_queries is None:
                    raise RuntimeError(
                        "vegas requires verifier queries for every layer")
                layer_cache = layer.self_attn.attn.kv_cache
                key_cache = layer_cache[0]
                cache_hit = False
                cached_rows = [
                    self._vegas_page_cache.get((request_id, layer_index))
                    for request_id in request_ids
                ]
                if self.pillar_policy == "vegas_cached":
                    try:
                        with torch.profiler.record_function(
                                "orthrus.draft.vegas.reuse"):
                            plan = reuse_vegas_page_plan(
                                cached_rows, active_block_table,
                                history_lengths, key_cache.shape[1],
                                budget_ratio=self.pillar_budget_ratio,
                                min_tokens=self.pillar_min_tokens,
                                max_tokens=self.pillar_max_tokens,
                                dense_below_tokens=(
                                    self.draft_segment_threshold))
                        cache_hit = True
                    except ValueError:
                        plan = None
                else:
                    plan = None
                if plan is None:
                    with torch.profiler.record_function(
                            "orthrus.draft.vegas.select"):
                        plan = vegas_page_plan(
                            target_queries.index_select(
                                0, vegas_active_tensor),
                            key_cache, active_block_table, history_lengths,
                            budget_ratio=self.pillar_budget_ratio,
                            min_tokens=self.pillar_min_tokens,
                            max_tokens=self.pillar_max_tokens,
                            dense_below_tokens=self.draft_segment_threshold)
                    if self.pillar_policy == "vegas_cached":
                        for row, request_id in enumerate(request_ids):
                            old_count = (plan.selected_blocks[row] - 1
                                         if plan.selected_blocks[row]
                                         < plan.total_blocks[row] else 0)
                            if old_count:
                                self._vegas_page_cache[
                                    (request_id, layer_index)] = (
                                        plan.block_table[
                                            row, :old_count].clone())
                self._pillar_block_table = plan.block_table
                self._pillar_history_lengths = plan.sequence_lengths
                instant("orthrus.draft.vegas.plan", {
                    "layer": layer_index,
                    "cache_hit": cache_hit,
                    "budget_ratio": self.pillar_budget_ratio,
                    "selected_blocks": plan.selected_blocks,
                    "total_blocks": plan.total_blocks,
                })
            with torch.profiler.record_function("orthrus.draft.layer"):
                hidden_states = self._draft_attention(
                    layer, normalized.view(len(active_indices), 32, -1),
                    positions, table, active_indices, history_lengths)
                hidden_states, residual = layer.post_attention_layernorm(
                    hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        candidate_hidden = hidden_states.view(len(active_indices), 32, -1)[:, :-1]
        logits = self.compute_logits(candidate_hidden.reshape(-1,
                                                               self.config.hidden_size))
        with torch.profiler.record_function("orthrus.draft.probabilities"):
            probabilities = logits.float().softmax(-1).view(
                len(active_indices), 31, vocabulary)
        active_tokens = []
        with torch.profiler.record_function("orthrus.draft.sample"):
            for local, request_index in enumerate(active_indices):
                exponential = torch.empty_like(probabilities[local]).exponential_(
                    generator=input_batch.generators[request_index])
                active_tokens.append((
                    probabilities[local] / exponential.clamp_min(
                        torch.finfo(torch.float32).tiny)).argmax(-1).tolist())
        if async_tokens:
            all_tokens = torch.full((batch_size, 31), -1, dtype=torch.long,
                                    device=device)
        else:
            all_tokens = [[] for _ in range(batch_size)]
        all_probabilities = probabilities.new_zeros((batch_size, 31, vocabulary))
        for local, request_index in enumerate(active_indices):
            if async_tokens:
                all_tokens[request_index] = torch.tensor(
                    active_tokens[local], dtype=torch.long, device=device)
            else:
                all_tokens[request_index] = active_tokens[local]
            all_probabilities[request_index] = probabilities[local]
        return OrthrusDraftBatch(
            all_tokens, all_probabilities,
            accepted_token_counts=accepted_token_counts,
            next_token_ids=next_token_ids,
            history_lengths=history_lengths)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings
                           else None))
        return loader.load_weights(weights)
