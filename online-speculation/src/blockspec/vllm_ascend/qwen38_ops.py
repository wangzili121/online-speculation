"""Existing NPU primitives with differentiable FP32-intermediate fallbacks."""

from functools import partial
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def swiglu(packed):
    if packed.device.type == "npu":
        import torch_npu
        return torch_npu.npu_swiglu(packed)
    gate, up = packed.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(packed.dtype)


def l2norm(value):
    if value.device.type == "npu" and not torch.is_grad_enabled() and value.shape[-1] == 128:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        return l2norm_fwd(value)
    full = value.float()
    return (full * torch.rsqrt(full.square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype)


def causal_conv(x, weight, bias=None, activation="silu", seq_idx=None):
    if seq_idx is not None or activation not in {"silu", "swish"}:
        raise ValueError("unpacked SiLU causal convolution required")
    if (x.device.type != "npu" or x.dtype != torch.bfloat16 or torch.is_grad_enabled()
            or bias is not None or weight.shape[-1] != 4 or x.shape[0] != 1):
        if torch.is_inference(weight) and torch.is_grad_enabled() and x.requires_grad:
            weight = weight.clone()
        out = F.conv1d(x.float(), weight[:, None].float(), None if bias is None else bias.float(),
                       padding=weight.shape[-1] - 1, groups=weight.shape[0])[..., :x.shape[-1]]
        return F.silu(out).to(x.dtype)
    # 处理完整输入（可能带有 HF 拼接的历史）；临时卷积状态绝不指向主模型。
    raw = x[0].T.contiguous()
    state = x.new_zeros((1, 34, x.shape[1]))
    tail = x.new_zeros((3, x.shape[1]))
    result = []
    for start in range(0, raw.shape[0], 32):
        part = raw[start:start + 32]
        state[0, :3] = tail
        out = torch.empty_like(part)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out, part, weight.T, conv_state=state, bias_opt=None,
            query_start_loc_opt=torch.tensor([0, part.shape[0]], dtype=torch.int32, device=x.device),
            cache_indices_opt=torch.zeros((1, part.shape[0]), dtype=torch.int32, device=x.device),
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=torch.ones(1, dtype=torch.int32, device=x.device),
            activation_mode=1, pad_slot_id=-1, run_mode=1)
        result.append(out)
        tail = torch.cat((tail, part), dim=0)[-3:]
    return torch.cat(result).T[None]


def causal_conv_update(hidden, state, weight, bias=None, activation="silu"):
    updated = torch.cat((state[..., 1:], hidden.reshape(*state.shape[:2], 1)), dim=-1)
    output = causal_conv(updated, weight, bias, activation)[..., -1:]
    state.copy_(updated)
    return output


def delta_rule(query, key, value, *, fallback, groups, **kwargs):
    normalize = kwargs.pop("use_qk_l2norm_in_kernel", False)
    if normalize:
        query, key = l2norm(query), l2norm(key)
    native = (query.device.type == "npu" and query.dtype == torch.bfloat16
              and query.shape[0] == 1 and 1 <= query.shape[1] <= 32
              and query.shape[-1] == value.shape[-1] == 128 and not torch.is_grad_enabled()
              and kwargs.get("cu_seqlens") is None
              and not set(kwargs) - {"g", "beta", "initial_state", "output_final_state", "cu_seqlens"})
    if not native:
        return fallback(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
    from .qwen38_gdn_backend import recurrent_slices
    length = query.shape[1]
    state = torch.zeros((length, value.shape[2], 128, 128), device=query.device, dtype=torch.float32)
    initial = kwargs.get("initial_state")
    if initial is not None:
        state[:1] = initial.transpose(-1, -2).float()
    # HF 已经展开了 GQA 头；原生递推可直接接收少量 Q/K 头。
    operation = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    operation = getattr(operation, "__wrapped__", operation)
    out = recurrent_slices(operation, query=query[0, :, ::groups].contiguous(),
        key=key[0, :, ::groups].contiguous(), value=value[0].contiguous(),
        g=kwargs["g"][0].float().contiguous(), beta=kwargs["beta"][0].contiguous(), state=state,
        actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device=query.device),
        ssm_state_indices=torch.arange(length, dtype=torch.int32, device=query.device),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=query.device), scale=128 ** -.5)
    final = state[-1:].transpose(-1, -2).contiguous() if kwargs.get("output_final_state", False) else None
    return out[None], final


class GatedRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.variance_epsilon = weight, eps

    def forward(self, value, gate):
        weight = self.weight
        if value.device.type == "npu" and not torch.is_grad_enabled():
            from vllm_ascend.ops.layernorm import layer_norm_fwd_npu
            return layer_norm_fwd_npu(value, weight, None, self.variance_epsilon, z=gate,
                group_size=None, norm_before_gate=True, is_rms_norm=True, activation="swish")[0]
        if torch.is_inference(weight) and torch.is_grad_enabled() and (value.requires_grad or gate.requires_grad):
            weight = weight.clone()
        full = value.float()
        normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (normalized * weight.float() * F.silu(gate.float())).to(value.dtype)


def configure_gdn(mixer):
    mixer.norm = GatedRMSNorm(mixer.norm.weight, mixer.layer_norm_epsilon)
    mixer.causal_conv1d_fn = causal_conv
    mixer.causal_conv1d_update = causal_conv_update
    groups = mixer.num_v_heads // mixer.num_k_heads
    mixer.chunk_gated_delta_rule = partial(delta_rule, fallback=mixer.chunk_gated_delta_rule, groups=groups)
    mixer.recurrent_gated_delta_rule = partial(delta_rule, fallback=mixer.recurrent_gated_delta_rule, groups=groups)


class SharedRotary:
    def __init__(self, original):
        self.cache = original.cos_sin_cache
        self.sections = tuple(original.mrope_section)
        self.interleaved = original.mrope_interleaved
        self.dim = original.rotary_dim

    def __deepcopy__(self, memo):
        return self

    def positions(self, start, length, device):
        return torch.arange(start, start + length, device=device)

    def coefficients(self, positions, dtype):
        cos, sin = self.cache.index_select(0, positions).chunk(2, -1)
        return tuple(torch.cat((value, value), dim=-1)[None, None].to(dtype) for value in (cos, sin))


def full_attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    if (hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] != 1 or kwargs
            or self.config._attn_implementation != "sdpa"):
        return self._reference_forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
    import torch_npu

    batch, length, _ = hidden_states.shape
    shape = (batch, length, -1, self.head_dim)
    projected_q, projected_k, projected_v = (projection(hidden_states)
        for projection in (self.q_proj, self.k_proj, self.v_proj))
    query, gate = projected_q.reshape(batch, length, -1, 2 * self.head_dim).chunk(2, -1)
    query, key = query.reshape(shape).transpose(1, 2), projected_k.reshape(shape).transpose(1, 2)
    value = projected_v.reshape(shape).transpose(1, 2)
    native_rope = self._native_rope
    if native_rope is not None:
        row = None if past_key_values is None else past_key_values.layers[self.layer_idx].keys
        past = 0 if row is None else row.shape[2]
        positions = native_rope.positions(past, length, hidden_states.device)
        cos, sin = native_rope.coefficients(positions, query.dtype)
    else:
        cos, sin = (component[:, None].contiguous() for component in position_embeddings)
    rotary_dim = cos.shape[-1]

    if native_rope is not None and not torch.is_grad_enabled():
        packed = torch.cat((projected_q, projected_k, projected_v), dim=-1).reshape(length, -1)
        cos_sin = native_rope.cache.index_select(0, positions).to(packed.dtype)[None].expand(3, -1, -1).contiguous()
        q, k, v, raw_gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=packed, q_weight=1 + self.q_norm.weight, k_weight=1 + self.k_norm.weight,
            cos_sin=cos_sin, num_q_heads=query.shape[1], num_kv_heads=key.shape[1], head_size=self.head_dim,
            eps=self.config.rms_norm_eps, mrope_section=list(native_rope.sections),
            is_interleaved=native_rope.interleaved, rope_dim=native_rope.dim, has_gate=True)
        query = q.reshape(shape).transpose(1, 2)
        key, value = (part.reshape(shape).transpose(1, 2) for part in (k, v))
        gate = raw_gate.reshape(batch, length, -1)
    else:
        def normalize_rotate(value, weight):
            full = value.float()
            # 原生融合分支先形成 BF16 gamma，但 norm 与 RoPE 之间不舍入。
            normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
            normalized = normalized * (1 + weight).float()
            rotated = torch_npu.npu_rotary_mul(normalized[..., :rotary_dim].contiguous(), cos.float(), sin.float())
            return torch.cat((rotated, normalized[..., rotary_dim:]), dim=-1).to(value.dtype).contiguous()

        query = normalize_rotate(query, self.q_norm.weight)
        key = normalize_rotate(key, self.k_norm.weight)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)
    copies = self.num_key_value_groups
    key, value = key.repeat_interleave(copies, dim=1), value.repeat_interleave(copies, dim=1)
    # PyTorch 加性 mask 中的 -inf 对应 Ascend FAS 的 True（禁止读取）。
    blocked = (None if attention_mask is None else (~attention_mask if attention_mask.dtype == torch.bool
               else torch.isneginf(attention_mask)).contiguous())
    out = torch_npu.npu_fusion_attention(query, key.contiguous(), value.contiguous(),
        query.shape[1], input_layout="BNSD", atten_mask=blocked, scale=self.scaling,
        keep_prob=1., sparse_mode=0)[0]
    out = out.transpose(1, 2).reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
    return self.o_proj(out), None


def configure_attention(mixer, native_rope=None):
    mixer._reference_forward = mixer.forward
    mixer._native_rope = None if native_rope is None else SharedRotary(native_rope)
    mixer.forward = MethodType(full_attention, mixer)
