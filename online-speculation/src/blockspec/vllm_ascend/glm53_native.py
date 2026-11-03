"""Experimental GLM-Flash draft execution over a shared native backbone.

The hybrid policy is explicit: bidirectional sparse MLA and optional gated
reverse-block KDA. This is not a claim of an author-trained GLM head.
"""

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..parallel.backbone import BackboneOutput
from .glm53_ops import kda_convolve, kda_native, select_history, sparse_draft_attention


HYBRID_POLICY = "experimental_causal_kda_bidirectional_mla_v1"
BIDIRECTIONAL_POLICY = "experimental_residual_bidir_kda_bidirectional_mla_v1"
HYBRID_POLICIES = (HYBRID_POLICY, BIDIRECTIONAL_POLICY)


def clone_projection(module):
    # 不复制自定义 Parameter 的 loader；推理模块仍保留原生量化方法和 scale。
    memo = {id(p): nn.Parameter(p.detach().clone(), requires_grad=False)
            for p in module.parameters()}
    return copy.deepcopy(module, memo)


def cache_tensor(value):
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    # NoPE 的共享 MLA 分配器仍可能保留不占存储的 RoPE 占位。
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and isinstance(value[1], torch.Tensor) and value[1].numel() == 0):
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
        raise ValueError(f"GLM draft requires a bound BF16 target KV tensor, got {type(value).__name__}")
    return value


def cache_rows(tensor, indices):
    """先切请求行再复制；NPU index_select 会物化带 padding 的整块 arena。"""
    if indices.ndim != 1 or not indices.numel():
        raise ValueError("nonempty one-dimensional cache row indices required")
    rows = []
    for index in indices.cpu().tolist():
        if not 0 <= index < tensor.shape[0]:
            raise ValueError("cache row outside the allocated arena")
        rows.append(tensor[index:index + 1].clone(memory_format=torch.contiguous_format))
    return rows[0] if len(rows) == 1 else torch.cat(rows)


def kda_history(attention, metadata, *, accepted, query_length):
    """读取原生 GLM 在接受位置保留的卷积尾部和 VK 状态，不复制其他请求。"""
    if not 1 <= accepted <= query_length or attention.conv_size != 4:
        raise ValueError("invalid GLM KDA accepted boundary or convolution width")
    conv, recurrent = attention.kv_cache
    if attention._conv_state_dim_first:
        conv = conv.transpose(-1, -2)
    speculative = metadata.spec_sequence_masks is not None and metadata.num_spec_decodes > 0
    if speculative:
        state_index = metadata.spec_state_indices_tensor[0, accepted - 1:accepted]
        conv_index = metadata.spec_decode_metadata.spec_causal_conv1d.cache_indices[0, :1]
        offset = accepted - 1
    else:
        if accepted != query_length:
            raise ValueError("non-speculative KDA cannot roll back a partial query")
        if metadata.num_prefills:
            state_index = metadata.prefill_state_indices[:1]
            conv_meta = metadata.non_spec_prefill_metadata.causal_conv1d
        else:
            state_index = metadata.non_spec_state_indices_tensor[:1]
            conv_meta = metadata.non_spec_decode_metadata.causal_conv1d
        conv_index = conv_meta.cache_indices.reshape(-1)[:1]
        offset = 0
    if conv.ndim != 3 or recurrent.ndim != 4 or offset + 3 > conv.shape[1]:
        raise ValueError("unsupported GLM KDA state layout")
    tail = cache_rows(conv, conv_index)[0, offset:offset + 3].contiguous()
    return tail, cache_rows(recurrent, state_index)[0]


def read_pages(cache, table, length, *, block_size=None):
    """复制请求的有效分页前缀，返回 [L, dim]，不会引用可写 cache。"""
    if (cache.ndim != 4 or cache.shape[2] != 1 or table.ndim != 2
            or table.shape[0] != 1 or length < 0):
        raise ValueError("unsupported GLM batch-one paged latent layout")
    block_size = cache.shape[1] if block_size is None else block_size
    if block_size < 1 or cache.shape[1] % block_size:
        raise ValueError("operator pages must divide the physical cache page")
    split = cache.shape[1] // block_size
    pages = (length + block_size - 1) // block_size
    if pages > table.shape[1]:
        raise ValueError("prefix exceeds GLM page table")
    if not pages:
        return cache.new_empty((0, cache.shape[-1]))
    page_ids = table[0, :pages].cpu().tolist()
    for index in page_ids:
        if not 0 <= index < cache.shape[0] * split:
            raise ValueError("operator page outside the physical cache arena")
    pitch = cache.stride(1)
    # 将行末 padding 包含在只读视图中，再批量选行。直接 gather 非连续
    # cache 会让 NPU 先物化整个 arena，不能用 cache.reshape 替代此视图。
    if (pages >= 16 and cache.stride(-1) == 1 and pitch >= cache.shape[-1]
            and cache.stride(0) >= cache.shape[1] * pitch and cache.stride(0) % pitch == 0):
        page_pitch = cache.stride(0) // pitch
        rows = (cache.shape[0] - 1) * page_pitch + cache.shape[1]
        end = (cache.storage_offset() + rows * pitch) * cache.element_size()
        if end <= cache.untyped_storage().nbytes():
            arena = cache.as_strided((rows, pitch), (pitch, 1))
            starts = torch.tensor([i // split * page_pitch + i % split * block_size for i in page_ids])
            indices = (starts[:, None] + torch.arange(block_size)).flatten()[:length].to(cache.device)
            return arena.index_select(0, indices)[:, :cache.shape[-1]].contiguous()
    pieces = []
    for page, index in enumerate(page_ids):
        physical, subpage = divmod(index, split)
        offset = subpage * block_size
        width = min(block_size, length - page * block_size)
        pieces.append(cache[physical, offset:offset + width, 0].clone(memory_format=torch.contiguous_format))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


@dataclass(frozen=True)
class GLMHistory:
    layers: tuple
    length: int


class NativeGLM53History:
    backend = "shared_glm53_target_snapshot"

    def __init__(self, target, runner):
        self.target = target
        self.runner = runner
        names = {id(m): n for n, m in runner.compilation_config.static_forward_context.items()}
        self.entries = []
        for layer in target.model.layers:
            attention = layer.self_attn
            cache_layer = (attention if layer.layer_kind == "kda"
                           else attention.mla_attn.mla_attn)
            self.entries.append((layer.layer_kind, attention, cache_layer, names[id(cache_layer)]))
        self.metadata = self.request_id = None
        self.reads = 0

    def capture(self, metadata, request_id):
        if not isinstance(metadata, dict):
            raise ValueError("GLM eager per-layer attention metadata required")
        self.metadata, self.request_id = metadata, request_id

    def read(self, request_id, length, *, sampled_count, draft_count):
        if self.metadata is None or request_id != self.request_id:
            raise RuntimeError("GLM history does not belong to the current request")
        sparse = next(name for kind, _, _, name in self.entries if kind == "mla")
        meta = self.metadata[sparse]
        if meta.seq_lens.numel() != 1 or meta.query_start_loc.numel() != 2:
            raise ValueError("GLM adapter supports only synchronous batch one")
        query_length = int(meta.query_start_loc[-1])
        accepted = sampled_count if draft_count else query_length
        if (not 1 <= accepted <= query_length
                or draft_count and query_length != draft_count + 1
                or length != int(meta.seq_lens[0]) - query_length + accepted):
            raise RuntimeError("GLM committed-token boundary disagrees with target metadata")
        states = []
        for kind, attention, cache_layer, name in self.entries:
            current = self.metadata[name]
            if kind == "kda":
                states.append(kda_history(attention, current, accepted=accepted, query_length=query_length))
            else:
                if (not torch.equal(current.seq_lens, meta.seq_lens)
                        or not torch.equal(current.query_start_loc, meta.query_start_loc)):
                    raise RuntimeError("GLM MLA layers disagree on the committed history boundary")
                indexer = attention.indexer
                index_meta = self.metadata[indexer.k_cache.prefix]
                latent = read_pages(cache_tensor(cache_layer.kv_cache), current.block_table, length,
                                    block_size=getattr(current, "block_size", None))
                # 完整池用缓存；不足一池的历史 token 总会被选入，不必重新压缩。
                pools = read_pages(cache_tensor(indexer.k_cache.kv_cache), index_meta.block_table,
                                   length // indexer.index_kpool, block_size=getattr(index_meta, "block_size", None))
                states.append((latent, pools))
        self.reads += 1
        return GLMHistory(tuple(states), length)

    def snapshot_target_cache(self):
        """验收用：复制当前请求的有效 KV、卷积和验证 checkpoint 到 CPU。"""
        result = {}
        for kind, attention, cache_layer, name in self.entries:
            current = self.metadata[name]
            if kind == "kda":
                conv, recurrent = cache_layer.kv_cache
                if current.spec_sequence_masks is not None and current.num_spec_decodes > 0:
                    sparse = next(n for k, _, _, n in self.entries if k == "mla")
                    query_length = int(self.metadata[sparse].query_start_loc[-1])
                    indices = current.spec_state_indices_tensor[0, :query_length]
                    conv_indices = current.spec_decode_metadata.spec_causal_conv1d.cache_indices[0, :1]
                else:
                    indices = (current.prefill_state_indices if current.num_prefills
                               else current.non_spec_state_indices_tensor)[:1]
                    metadata = (current.non_spec_prefill_metadata if current.num_prefills
                                else current.non_spec_decode_metadata)
                    conv_indices = metadata.causal_conv1d.cache_indices.reshape(-1)[:1]
                for suffix, tensor, ids in (("conv", conv, conv_indices), ("state", recurrent, indices)):
                    ids = ids.to(tensor.device, torch.long)
                    result[f"{name}.{suffix}"] = cache_rows(tensor, ids).detach().to("cpu", copy=True)
            else:
                length = int(current.seq_lens[0])
                result[f"{name}.latent"] = read_pages(
                    cache_tensor(cache_layer.kv_cache), current.block_table, length,
                    block_size=getattr(current, "block_size", None)).detach().to("cpu", copy=True)
                indexer = attention.indexer
                metadata = self.metadata[indexer.k_cache.prefix]
                result[f"{name}.indexer"] = read_pages(cache_tensor(indexer.k_cache.kv_cache),
                    metadata.block_table, length // indexer.index_kpool,
                    block_size=getattr(metadata, "block_size", None)).detach().to("cpu", copy=True)
                tail = indexer.tail_cache
                tensor = tail.kv_cache
                while isinstance(tensor, (tuple, list)) and len(tensor) == 1:
                    tensor = tensor[0]
                ids = self.metadata[tail.prefix].block_table[0, :1].to(tensor.device, torch.long)
                # 环形槽可能含尚未初始化的位置，按字节比较而非浮点 NaN 相等性。
                result[f"{name}.indexer_tail"] = cache_rows(tensor, ids).detach().contiguous().view(
                    torch.uint8).to("cpu", copy=True)
        return result


class SparseDraft(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.heads, self.q_dim = original.num_local_heads, original.qk_head_dim
        self.v_dim, self.rank = original.v_head_dim, original.kv_lora_rank
        self.q_rank = original.q_lora_rank
        self.index_heads, self.index_dim = original.indexer.n_head, original.indexer.head_dim
        self.pool, self.topk = original.indexer.index_kpool, original.indexer.topk_tokens
        for name in ("fused_qkv_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_layernorm", "o_proj"):
            setattr(self, name, clone_projection(getattr(original, name)))
        # 原生 SFA 加载后可能释放 kv_b_proj，直接读取仍用于 AR 的吸收权重。
        impl = original.mla_attn.mla_attn.impl
        object.__setattr__(self, "ar_key", impl.W_UK_T)
        object.__setattr__(self, "ar_value", impl.W_UV.transpose(-1, -2))
        self.draft_key = nn.Parameter(self.ar_key.detach().clone(), requires_grad=False)
        self.draft_value = nn.Parameter(self.ar_value.detach().clone(), requires_grad=False)
        self.index_query = clone_projection(original.indexer.wq_b)
        # indexer 的融合矩阵中，后半部分才是每个 query 的 head weight 投影。
        self.index_weights = nn.Parameter(
            original.indexer.wk_weights_proj.weight[self.index_dim:].detach().clone(), requires_grad=False)

    def forward(self, hidden, history):
        qkv = self.fused_qkv_a_proj(hidden)[0]
        q, latent = qkv.split((self.q_rank, self.rank), dim=-1)
        q = self.q_a_layernorm(q)
        queries = self.q_b_proj(q)[0].reshape(-1, self.heads, self.q_dim)
        block = self.kv_a_layernorm(latent)
        iq = self.index_query(q)[0].reshape(-1, self.index_heads, self.index_dim)
        iw = F.linear(hidden, self.index_weights) * (self.index_dim**-.5 * self.index_heads**-.5)
        ar_latent, pools = history
        indices = select_history(iq, iw, pools, ar_latent.shape[0], topk=self.topk, pool_size=self.pool)
        attended = sparse_draft_attention(queries, ar_latent, block, indices,
                                         ar_key=self.ar_key, ar_value=self.ar_value,
                                         draft_key=self.draft_key, draft_value=self.draft_value, backend="npu")
        return self.o_proj(attended.flatten(1))[0]


class KDADraft(nn.Module):
    def __init__(self, original, *, bidirectional=False, backward_weight=0.):
        super().__init__()
        self.heads, self.dim = original.local_num_heads, original.head_dim
        self.lower_bound = original.kda_lower_bound
        for name in ("in_proj_qkvbfg_a", "f_b_proj", "g_b_proj", "o_norm", "o_proj"):
            setattr(self, name, clone_projection(getattr(original, name)))
        self.conv = nn.Parameter(torch.cat([getattr(original, n).weight.detach().reshape(-1, original.conv_size)
                                           for n in ("q_conv1d", "k_conv1d", "v_conv1d")]), requires_grad=False)
        self.a_log = nn.Parameter(original.A_log.detach().clone(), requires_grad=False)
        self.dt_bias = nn.Parameter(original.dt_bias.detach().clone(), requires_grad=False)
        self.register_parameter("backward_gate", None)
        self.reverse_calls = 0
        if bidirectional:
            if not math.isfinite(backward_weight) or not -1 < backward_weight < 1:
                raise ValueError("KDA backward weight must be finite and strictly between -1 and 1")
            self.backward_gate = nn.Parameter(torch.full((self.heads,), math.atanh(backward_weight),
                device=original.A_log.device, dtype=torch.float32), requires_grad=False)

    def _convolve(self, rows, tail, dtype):
        convolved = kda_convolve(rows, tail, self.conv, backend="auto")
        return convolved.to(dtype).reshape(-1, 3, self.heads, self.dim).unbind(1)

    def forward(self, hidden, history):
        size = self.heads * self.dim
        projected = self.in_proj_qkvbfg_a(hidden)[0]
        qkv, beta, f_a, g_a = projected.split((3 * size, self.heads, self.dim, self.dim), -1)
        tail, initial = history
        # 只在临时输入上卷积和递推；主模型的卷积及 recurrent state 均不更新。
        q, k, v = self._convolve(qkv, tail, hidden.dtype)
        gate = self.f_b_proj(f_a)[0].reshape(-1, self.heads, self.dim)
        output, _ = kda_native(q.contiguous(), k.contiguous(), v.contiguous(), gate, beta,
                               initial, self.a_log, self.dt_bias, lower_bound=self.lower_bound)
        if self.backward_gate is not None:
            # 反向只读当前块，从零状态开始；不能反向遍历 AR 历史。
            rq, rk, rv = self._convolve(qkv.flip(0), torch.zeros_like(tail), hidden.dtype)
            reverse, _ = kda_native(rq.contiguous(), rk.contiguous(), rv.contiguous(), gate.flip(0),
                beta.flip(0), torch.zeros_like(initial), self.a_log, self.dt_bias, lower_bound=self.lower_bound)
            output = output + self.backward_gate.tanh().to(output.dtype)[None, :, None] * reverse.flip(0)
            self.reverse_calls += 1
        out_gate = self.g_b_proj(g_a)[0].reshape(-1, self.heads, self.dim)
        return self.o_proj(self.o_norm(output[None], out_gate).reshape(hidden.shape[0], -1))[0]


@contextmanager
def replace_attention(layers, drafts, history):
    """仅在一次同步 draft 内替换 Attention；异常时也恢复 AR forward。"""
    originals = []
    try:
        for layer, draft, state in zip(layers, drafts, history.layers, strict=True):
            attention = layer.self_attn
            originals.append((attention, attention.__dict__.get("forward"), "forward" in attention.__dict__))
            def forward(hidden_states, positions, *, branch=draft, snapshot=state):
                del positions
                return branch(hidden_states, snapshot)
            attention.forward = forward
        yield
    finally:
        for attention, forward, existed in reversed(originals):
            if existed:
                attention.forward = forward
            else:
                attention.__dict__.pop("forward", None)


class SharedGLM53Draft(nn.Module):
    def __init__(self, target, config, *, policy=HYBRID_POLICY, backward_weight=0.):
        super().__init__()
        object.__setattr__(self, "target", target)
        self.vllm_config = config
        self.embedding = target.model.embed_tokens
        if policy not in HYBRID_POLICIES:
            raise ValueError("unsupported GLM hybrid policy")
        self.cache_auditor = None
        self.audit_checks = 0
        self.layers = nn.ModuleList(KDADraft(layer.self_attn, bidirectional=policy == BIDIRECTIONAL_POLICY,
                                            backward_weight=backward_weight) if layer.layer_kind == "kda"
                                    else SparseDraft(layer.self_attn) for layer in target.model.layers)

    def forward(self, tokens, *, view, cache, capture_layer=None):
        if view != "draft" or capture_layer is not None or not isinstance(cache, GLMHistory):
            raise ValueError("GLM plugin currently supports inference-only draft execution")
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        positions = torch.arange(cache.length, cache.length + tokens.numel(), device=tokens.device)
        before = None if self.cache_auditor is None else self.cache_auditor()
        # embedding、mHC、norm、MoE 和 LM head 全部调用同一份原生目标模型。
        with replace_attention(self.target.model.layers, self.layers, cache):
            with set_ascend_forward_context(None, self.vllm_config, num_tokens=tokens.numel(),
                                            model_instance=self.target, skip_compiled=True):
                hidden = self.target(tokens.flatten(), positions)
                logits = self.target.compute_logits(hidden)
        if before is not None:
            after = self.cache_auditor()
            if before.keys() != after.keys() or any(
                    before[k].shape != after[k].shape or before[k].dtype != after[k].dtype
                    or not torch.equal(before[k].contiguous().view(torch.uint8),
                                       after[k].contiguous().view(torch.uint8)) for k in before):
                raise RuntimeError("draft execution modified the target request's cache")
            self.audit_checks += 1
        return BackboneOutput(hidden, None, logits[None], None)
