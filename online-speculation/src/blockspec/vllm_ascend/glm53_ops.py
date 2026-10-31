"""GLM-5.3-Flash 的只读 draft 计算，不依赖推理引擎 metadata。"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Glm53Architecture:
    layer_types: tuple[str, ...]
    hidden_size: int
    heads: int
    qk_dim: int
    value_dim: int
    latent_dim: int
    index_heads: int
    index_dim: int
    topk: int
    pool_size: int
    kda_heads: int
    kda_dim: int

    @classmethod
    def from_config(cls, config):
        text = config.get("text_config", config)
        kinds = tuple(text.get("layer_types", ()))
        if (text.get("model_type") != "glm5_next_text"
                or len(kinds) != text.get("num_hidden_layers")
                or set(kinds) != {"linear_attention", "deepseek_sparse_attention"}
                or text.get("qk_rope_head_dim") != 0):
            raise ValueError("GLM-5.3-Flash hybrid NoPE text configuration required")
        linear = text["linear_attn_config"]
        result = cls(kinds, text["hidden_size"], text["num_attention_heads"],
                     text["qk_nope_head_dim"], text["v_head_dim"], text["kv_lora_rank"],
                     text["index_n_heads"], text["index_head_dim"], text["index_topk"],
                     text["index_kpool"], linear["num_heads"], linear["head_dim"])
        if result.pool_size < 1 or result.topk < 1 or result.topk % result.pool_size:
            raise ValueError("index_topk must be divisible by index_kpool")
        return result


def select_history(query, head_weights, pooled_keys, prefix_length, *, topk, pool_size=4):
    """每行选择完整历史池，展开成 token 位置，再追加未满池的历史尾部。

    query: [Q, index_heads, index_dim]；head_weights 已包含 indexer 的缩放。
    pooled_keys 只包含 anchor 之前的完整池，不含临时 draft 的 key。
    """
    if (query.ndim != 3 or head_weights.shape != query.shape[:2]
            or pooled_keys.ndim != 2 or pooled_keys.shape[-1] != query.shape[-1]
            or type(prefix_length) is not int or prefix_length < 0
            or pool_size < 1 or topk < 1 or topk % pool_size):
        raise ValueError("invalid KPool read-only query contract")
    complete = prefix_length // pool_size
    if pooled_keys.shape[0] != complete:
        raise ValueError("pooled keys must end at the committed prefix")
    # ORTHRUS：所有 query 的历史边界相同，不随块内位置向右推进。
    scores = torch.einsum("qhd,pd->qhp", query.float(), pooled_keys.float())
    scores = (scores.relu() * head_weights.float().unsqueeze(-1)).sum(1)
    chosen = scores.topk(min(complete, topk // pool_size), dim=-1).indices
    offsets = torch.arange(pool_size, device=query.device)
    indices = (chosen.unsqueeze(-1) * pool_size + offsets).flatten(1)
    tail = torch.arange(complete * pool_size, prefix_length, device=query.device)
    return torch.cat((indices, tail.expand(query.shape[0], -1)), dim=-1)


def _segment(query, latent, scale, backend):
    """[Q,H,R] 对每行自己的 [Q,S,R] 计算输出及自然对数 LSE。"""
    if latent.shape[1] == 0:
        return torch.zeros_like(query), query.new_full(query.shape[:2], -torch.inf, dtype=torch.float32)
    if backend == "npu":
        import torch_npu

        output, lse = torch_npu.npu_fused_infer_attention_score(
            query[:, None].contiguous(), latent[:, :, None].contiguous(),
            latent[:, :, None].contiguous(), num_heads=query.shape[1],
            num_key_value_heads=1, input_layout="BSND", sparse_mode=0,
            scale=scale, softmax_lse_flag=True,
            actual_seq_lengths=[1] * query.shape[0],
            actual_seq_lengths_kv=[latent.shape[1]] * query.shape[0],
        )[:2]
        return output[:, 0], lse.reshape(query.shape[:2]).float()
    if backend != "torch":
        raise ValueError("backend must be torch or npu")
    scores = torch.einsum("qhr,qsr->qhs", query.float(), latent.float()) * scale
    lse = scores.logsumexp(-1)
    output = torch.einsum("qhs,qsr->qhr", scores.softmax(-1), latent.float())
    return output.to(query.dtype), lse


def sparse_draft_attention(query, history, block, indices, *, ar_key, ar_value,
                           draft_key, draft_value, backend="torch"):
    """稀疏 MLA：主模型低秩历史 + 完整临时双向块，返回 [Q,H,V]。

    query=[Q,H,D]；history=[L,R]；block=[Q,R]；indices=[Q,S]。
    key/value 权重分别为 [H,D,R] / [H,V,R]，不要求 AR 与 draft 相同。
    """
    if (query.ndim != 3 or history.ndim != 2 or block.ndim != 2
            or block.shape[0] != query.shape[0] or history.shape[1] != block.shape[1]
            or indices.ndim != 2 or indices.shape[0] != query.shape[0]):
        raise ValueError("invalid sparse draft tensor layout")
    q_rows, heads, dim = query.shape
    latent_dim = block.shape[1]
    if (ar_key.shape != (heads, dim, latent_dim) or draft_key.shape != ar_key.shape
            or ar_value.shape[0] != heads or ar_value.shape[-1] != latent_dim
            or draft_value.shape != ar_value.shape):
        raise ValueError("incompatible AR/draft latent expansion weights")
    # ORTHRUS：历史按 AR 的 KV 展开权重解释，临时块按 draft 的权重解释。
    selected = history[indices]
    q_ar = torch.einsum("qhd,hdr->qhr", query, ar_key)
    q_draft = torch.einsum("qhd,hdr->qhr", query, draft_key)
    ar_out, ar_lse = _segment(q_ar, selected, dim**-.5, backend)
    draft_out, draft_lse = _segment(q_draft, block.expand(q_rows, -1, -1), dim**-.5, backend)
    ar_out = torch.einsum("qhr,hvr->qhv", ar_out, ar_value)
    draft_out = torch.einsum("qhr,hvr->qhv", draft_out, draft_value)
    # 两段采用同一个 softmax 分母，不是各自归一化后直接相加。
    if backend == "npu" and indices.shape[1]:
        from .dsv4_hybrid import merge_attention_outputs

        return merge_attention_outputs((ar_out, draft_out),
                                       (ar_lse.unsqueeze(-1), draft_lse.unsqueeze(-1)))
    total_lse = torch.logaddexp(ar_lse, draft_lse)
    output = (ar_out.float() * (ar_lse - total_lse).exp().unsqueeze(-1)
              + draft_out.float() * (draft_lse - total_lse).exp().unsqueeze(-1))
    return output.to(query.dtype)


def kda_reference(query, key, value, raw_gate, raw_beta, initial, a_log, dt_bias,
                  *, lower_bound=-5.):
    """因果 KDA 参考计算；状态为 [H,V,K]，仅返回临时状态，不写 initial。

    这不是“双向 KDA”的定义，不能把它称为普通 SDPA 的换 mask 版本。
    """
    q = F.normalize(query.float(), dim=-1, eps=1e-6) * query.shape[-1]**-.5
    k = F.normalize(key.float(), dim=-1, eps=1e-6)
    gate = lower_bound * torch.sigmoid(a_log.float().reshape(1, -1, 1).exp()
                                      * (raw_gate.float() + dt_bias.float().reshape(1, *q.shape[1:])))
    state = initial.float().clone()
    outputs, states = [], []
    for pos in range(query.shape[0]):
        state = state * gate[pos].exp().unsqueeze(-2)
        prediction = torch.einsum("hvk,hk->hv", state, k[pos])
        delta = (value[pos].float() - prediction) * raw_beta[pos].float().sigmoid().unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k[pos].unsqueeze(-2)
        outputs.append(torch.einsum("hvk,hk->hv", state, q[pos]))
        states.append(state.clone())
    return torch.stack(outputs).to(value.dtype), torch.stack(states)


def kda_native(query, key, value, raw_gate, raw_beta, initial, a_log, dt_bias,
               *, lower_bound=-5.):
    """调用已有 AscendC recurrent KDA，仅使用自有临时状态槽。"""
    length, heads, dim = query.shape
    if not 1 <= length <= 8 or dim != 128:
        raise ValueError("current native KDA supports 1..8 rows and 128-dim heads")
    states = initial.new_empty((length, heads, value.shape[-1], dim))
    states[0].copy_(initial)
    output = torch.ops._C_ascend.recurrent_kda(
        query[None], key[None], value[None], raw_gate[None], raw_beta[None], states,
        torch.tensor([0, length], dtype=torch.int32, device=query.device),
        torch.arange(length, dtype=torch.int32, device=query.device)[None],
        a_log.reshape(-1).contiguous(), dt_bias.contiguous(),
        scale=dim**-.5, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
        safe_gate=True, lower_bound=lower_bound,
    )
    return output[0], states


def finite_error(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("non-finite comparison")
    delta = (actual.float() - expected.float()).abs()
    return dict(max_abs=float(delta.max()), rms=float(delta.square().mean().sqrt()),
                relative_rms=float(delta.square().mean().sqrt()
                                   / expected.float().square().mean().sqrt().clamp_min(math.ulp(1.))))
