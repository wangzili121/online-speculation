"""Native vLLM BF16 TP weights exposed to the Orthrus functional executor.

No checkpoint is loaded here. Shared parameters alias the target; only draft
mixers are copied. Collectives follow column/row-parallel autograd semantics.
"""

import copy
import hashlib
import json

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..parallel.online import SuffixLearner
from ..parallel.qwen38 import Qwen38DualView


class TensorParallel:
    def __init__(self, group=None, ranks=(0,)):
        self.group, self.ranks = group, tuple(ranks)
        self.size = len(self.ranks)
        self.rank = 0 if self.size == 1 else dist.get_rank(group)

    def __deepcopy__(self, memo):
        return self

    def reduce(self, value):
        if self.size > 1:
            dist.all_reduce(value, group=self.group)
        return value

    def gather(self, value):
        if self.size == 1:
            return value
        pieces = [torch.empty_like(value) for _ in self.ranks]
        dist.all_gather(pieces, value.contiguous(), group=self.group)
        return torch.cat(pieces, dim=-1)

    def broadcast(self, value):
        if self.size > 1:
            dist.broadcast(value, src=self.ranks[0], group=self.group)
        return value


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, tp):
        ctx.tp = tp
        return value

    @staticmethod
    def backward(ctx, gradient):
        return ctx.tp.reduce(gradient.contiguous().clone()), None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, tp):
        return tp.reduce(value.clone())

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, tp):
        ctx.tp = tp
        return tp.gather(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.chunk(ctx.tp.size, dim=-1)[ctx.tp.rank].contiguous(), None


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight):
        # 原生推理权重不能 save_for_backward；仅保留只读引用，反向只算输入梯度。
        ctx.weight = weight
        return F.linear(value, weight)

    @staticmethod
    def backward(ctx, gradient):
        return gradient @ ctx.weight, None


class ParallelLinear(nn.Module):
    def __init__(self, weight, tp, mode):
        super().__init__()
        self.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight, requires_grad=False)
        self.tp, self.mode = tp, mode

    def forward(self, value):
        if self.mode in {"column", "vocab"}:
            value = _Copy.apply(value, self.tp)
        if torch.is_inference(self.weight) and torch.is_grad_enabled() and value.requires_grad:
            if self.weight.requires_grad:
                raise RuntimeError("target inference weights must remain frozen")
            out = _FrozenLinear.apply(value, self.weight)
        else:
            out = F.linear(value, self.weight)
        if self.mode == "row":
            out = _Reduce.apply(out, self.tp)
        elif self.mode == "vocab":
            out = _Gather.apply(out, self.tp)
        return out


class SharedRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.eps = weight, eps

    def forward(self, value):
        weight = self.weight
        if torch.is_inference(weight) and torch.is_grad_enabled() and value.requires_grad:
            weight = weight.clone()
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + weight.float())).to(value.dtype)


class SharedEmbedding(nn.Module):
    def __init__(self, original, tp, vocab_size):
        super().__init__()
        self.weight = original.weight
        indices = original.shard_indices
        self.start, self.end = indices.org_vocab_start_index, indices.org_vocab_end_index
        self.tp, self.vocab_size = tp, vocab_size
        if getattr(original, "num_added_embeddings", 0):
            raise ValueError("added vocabulary/LoRA embeddings are not supported")

    def forward(self, tokens):
        outside = (tokens < self.start) | (tokens >= self.end)
        local = (tokens - self.start).masked_fill(outside, 0)
        out = F.embedding(local, self.weight).masked_fill(outside[..., None], 0)
        return self.tp.reduce(out)


class SharedHead(ParallelLinear):
    def __init__(self, weight, tp, vocab_size):
        super().__init__(weight, tp, "vocab")
        self.vocab_size = vocab_size

    def forward(self, value):
        return super().forward(value)[..., :self.vocab_size]


def _weight(module):
    weight = module.weight
    if weight.dtype not in {torch.bfloat16, torch.float32} or weight.ndim != 2:
        raise ValueError("unquantized, unpacked BF16/FP32 native linear weights required")
    return weight


def from_native_model(target, config, *, tp, block_size, mask_token_id):
    """Build a TP-sharded HF-shaped view without a second backbone allocation."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    language = getattr(target, "language_model", target)
    native = language.model
    if (config.model_type != "qwen3_5_text" or getattr(config, "layer_scale", False)
            or not getattr(config, "attn_output_gate", True) or getattr(config, "attention_bias", False)):
        raise ValueError("dense Qwen3.5/3.8 text architecture with gated attention and no layer scaling required")
    config = Qwen3_5TextConfig(**config.to_dict())
    local = copy.deepcopy(config)
    for field in ("num_attention_heads", "num_key_value_heads", "intermediate_size",
                  "linear_num_key_heads", "linear_num_value_heads"):
        value = getattr(local, field)
        if value % tp.size:
            raise ValueError(f"{field}={value} is not divisible by TP={tp.size}")
        setattr(local, field, value // tp.size)
    with torch.device("meta"):
        shaped = Qwen3_5ForCausalLM(local)
    text = shaped.model
    text.embed_tokens = SharedEmbedding(native.embed_tokens, tp, config.vocab_size)
    text.norm = SharedRMSNorm(native.norm.weight, config.rms_norm_eps)
    shaped.lm_head = SharedHead(_weight(language.lm_head), tp, config.vocab_size)
    # 只新建很小的 RoPE 缓冲；backbone 参数均引用当前 TP rank 的原生权重。
    text.rotary_emb = type(text.rotary_emb)(local).to(native.norm.weight.device)
    for index, (layer, source) in enumerate(zip(text.layers, native.layers, strict=True)):
        layer.input_layernorm = SharedRMSNorm(source.input_layernorm.weight, config.rms_norm_eps)
        layer.post_attention_layernorm = SharedRMSNorm(source.post_attention_layernorm.weight, config.rms_norm_eps)
        gate, up = _weight(source.mlp.gate_up_proj).chunk(2)
        layer.mlp.gate_proj = ParallelLinear(gate, tp, "column")
        layer.mlp.up_proj = ParallelLinear(up, tp, "column")
        layer.mlp.down_proj = ParallelLinear(_weight(source.mlp.down_proj), tp, "row")
        if config.layer_types[index] == "full_attention":
            mixer, ar = layer.self_attn, source.self_attn
            q = local.num_attention_heads * local.head_dim * 2
            kv = local.num_key_value_heads * local.head_dim
            parts = _weight(ar.qkv_proj).split((q, kv, kv))
            for name, weight in zip(("q_proj", "k_proj", "v_proj"), parts, strict=True):
                setattr(mixer, name, ParallelLinear(weight, tp, "column"))
            mixer.o_proj = ParallelLinear(_weight(ar.o_proj), tp, "row")
            mixer.q_norm = SharedRMSNorm(ar.q_norm.weight, config.rms_norm_eps)
            mixer.k_norm = SharedRMSNorm(ar.k_norm.weight, config.rms_norm_eps)
        else:
            mixer, ar = layer.linear_attn, source.linear_attn
            if getattr(ar, "gqa_interleaved_layout", False) or getattr(ar, "disable_tp_for_ba_proj", False):
                raise ValueError("Qwen3.5 non-interleaved, TP-sharded GDN weights required")
            kd = local.linear_num_key_heads * local.linear_key_head_dim
            vd = local.linear_num_value_heads * local.linear_value_head_dim
            packed = _weight(ar.in_proj_qkvz)
            mixer.in_proj_qkv = ParallelLinear(packed[:2 * kd + vd], tp, "column")
            mixer.in_proj_z = ParallelLinear(packed[2 * kd + vd:], tp, "column")
            b, a = _weight(ar.in_proj_ba).chunk(2)
            mixer.in_proj_b = ParallelLinear(b, tp, "column")
            mixer.in_proj_a = ParallelLinear(a, tp, "column")
            mixer.out_proj = ParallelLinear(_weight(ar.out_proj), tp, "row")
            mixer.conv1d.weight = ar.conv1d.weight
            mixer.A_log, mixer.dt_bias = ar.A_log, ar.dt_bias
            mixer.norm.weight = ar.norm.weight
    model = Qwen38DualView(shaped, block_size=block_size, mask_token_id=mask_token_id,
                           hybrid_policy="orthrus_qwen35_reference")
    # 映射不完整就报错，不能用随机参数掩盖缺失的权重。
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("native weight mapping left unmapped meta parameters")
    model.source = {"model": str(getattr(target, "prefix", "native-vllm")),
                    "tp_size": tp.size, "tp_rank": tp.rank, "history_backend": "replay_native_weights"}
    return model


class NativeSuffixLearner(SuffixLearner):
    """TP-aware online replay against an immutable native vLLM target."""
    def __init__(self, model, config, tp):
        self.tp = tp
        super().__init__(model, config)
        self.replicated = {name for name in self.master if ".q_norm." in name or ".k_norm." in name
                           or ".norm." in name}

    @staticmethod
    def parameter_stamp(parameter):
        version = None if torch.is_inference(parameter) else parameter._version
        return (parameter.data_ptr(), tuple(parameter.shape), parameter.device, parameter.dtype, version)

    def fingerprint(self):
        # No 50GB host copy at worker startup. This identity is process-local:
        # target hot reload and cross-process optimizer restore are prohibited.
        identity = [(name, str(self.parameter_stamp(p))) for name, p in self.model.named_parameters()
                    if name not in self.execution]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def backward(self):
        loss = super().backward()
        for name in sorted(self.replicated):
            if self.master[name].grad is not None:
                self.tp.reduce(self.master[name].grad)
        return loss

    def clip_gradients(self):
        total = torch.zeros((), device=self.parameters[0].device, dtype=torch.float32)
        for name, value in self.master.items():
            if value.grad is not None and (name not in self.replicated or self.tp.rank == 0):
                total += value.grad.float().square().sum()
        norm = self.tp.reduce(total).sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite TP draft gradient norm")
        scale = (self.config.clip_norm / (norm + 1e-6)).clamp(max=1.)
        for value in self.parameters:
            if value.grad is not None:
                value.grad.mul_(scale)
        return norm
