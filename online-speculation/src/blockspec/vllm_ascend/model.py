"""Single-card vLLM model with shared AR/draft Orthrus backbone weights."""

from collections.abc import Iterable

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
        repeats = attention.num_heads // attention.num_kv_heads
        for local, request_index in enumerate(request_indices):
            with torch.profiler.record_function("orthrus.draft.kv_gather"):
                old_k, old_v = self._cache(
                    layer, table, request_index, history_lengths[local])
            full_k = torch.cat((old_k, k[local]), dim=0)
            full_v = torch.cat((old_v, v[local]), dim=0)
            if repeats > 1:
                full_k = full_k.repeat_interleave(repeats, dim=1)
                full_v = full_v.repeat_interleave(repeats, dim=1)
            with torch.profiler.record_function("orthrus.draft.attention"):
                attended = F.scaled_dot_product_attention(
                    q[local].transpose(0, 1)[None],
                    full_k.transpose(0, 1)[None],
                    full_v.transpose(0, 1)[None],
                    is_causal=False, scale=attention.scaling)
            rows.append(attended[0].transpose(0, 1).reshape(block, -1))
        return attention.draft_output(torch.stack(rows).reshape(active * block, -1))

    @torch.inference_mode()
    def orthrus_propose(self, *, sampled_token_ids, input_batch,
                        scheduler_output, common_attn_metadata):
        del scheduler_output, common_attn_metadata
        if get_pp_group().world_size != 1:
            raise NotImplementedError("the first Orthrus baseline is single-card only")
        active_indices = [index for index, row in enumerate(sampled_token_ids)
                          if row]
        batch_size = len(input_batch.req_ids)
        vocabulary = self.config.vocab_size
        if not active_indices:
            empty = torch.empty((batch_size, 31, vocabulary),
                                device=self.model.embed_tokens.weight.device)
            return OrthrusDraftBatch([[] for _ in range(batch_size)], empty)
        for index in active_indices:
            if (float(input_batch.temperature_cpu[index]) != 1.0
                    or int(input_batch.top_k_cpu[index]) not in (-1, 0, vocabulary)
                    or float(input_batch.top_p_cpu[index]) != 1.0):
                raise ValueError("Orthrus baseline requires T=1, top_k=0, top_p=1")
            if input_batch.generators.get(index) is None:
                raise ValueError("Orthrus baseline requires a request-local seed")
        device = self.model.embed_tokens.weight.device
        mask_id = int(self.config.mask_token_id)
        anchors = [int(input_batch.token_ids_cpu[index,
                       int(input_batch.num_tokens_no_spec[index]) - 1])
                   for index in active_indices]
        tokens = torch.full((len(active_indices), 32), mask_id,
                            dtype=torch.long, device=device)
        tokens[:, 0] = torch.tensor(anchors, dtype=torch.long, device=device)
        history_lengths = [int(input_batch.num_tokens_no_spec[index]) - 1
                           for index in active_indices]
        positions = torch.stack([
            torch.arange(length, length + 32, device=device)
            for length in history_lengths])
        with torch.profiler.record_function("orthrus.draft.embedding"):
            hidden_states = self.model.embed_input_ids(tokens).reshape(
                -1, self.config.hidden_size)
        residual = None
        table = input_batch.block_table[0]
        for layer in self.model.layers:
            if residual is None:
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
            else:
                normalized, residual = layer.input_layernorm(hidden_states,
                                                              residual)
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
        all_tokens = [[] for _ in range(batch_size)]
        all_probabilities = probabilities.new_zeros((batch_size, 31, vocabulary))
        for local, request_index in enumerate(active_indices):
            all_tokens[request_index] = active_tokens[local]
            all_probabilities[request_index] = probabilities[local]
        return OrthrusDraftBatch(all_tokens, all_probabilities)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings
                           else None))
        return loader.load_weights(weights)
