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
from .qwen38_ops import configure_attention, configure_gdn, swiglu


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

    def _project(self, value):
        if self.mode in {"column", "vocab"}:
            value = _Copy.apply(value, self.tp)
        if torch.is_inference(self.weight) and torch.is_grad_enabled() and value.requires_grad:
            if self.weight.requires_grad:
                raise RuntimeError("target inference weights must remain frozen")
            out = _FrozenLinear.apply(value, self.weight)
        else:
            out = F.linear(value, self.weight)
        return out

    def forward(self, value):
        out = self._project(value)
        if self.mode == "row":
            out = _Reduce.apply(out, self.tp)
        elif self.mode == "vocab":
            out = _Gather.apply(out, self.tp)
        return out


class SharedRMSNorm(nn.Module):
    def __init__(self, weight, eps, *, native_residual=False):
        super().__init__()
        self.weight, self.eps = weight, eps
        self.native_residual = native_residual
        with torch.inference_mode(False):
            self.register_buffer("_unit_weight", torch.ones_like(weight) if native_residual else None,
                                 persistent=False)

    def forward(self, value):
        weight = self.weight
        if torch.is_inference(weight) and torch.is_grad_enabled() and value.requires_grad:
            weight = weight.clone()
        if self.native_residual and value.dtype == torch.bfloat16:
            # 对齐原生残差 norm：归一化结果和 1+weight 各先舍入一次；首层及 Q/K norm 不走此分支。
            if value.device.type == "npu":
                import torch_npu
                normalized = torch_npu.npu_rms_norm(value, self._unit_weight, self.eps)[0]
            else:
                normalized = (value.float() * torch.rsqrt(
                    value.float().square().mean(-1, keepdim=True) + self.eps)).to(value.dtype)
            return normalized * (1 + weight)
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + weight.float())).to(value.dtype)


class SharedMLP(nn.Module):
    def __init__(self, gate_up, down, tp):
        super().__init__()
        self.gate_up_proj = ParallelLinear(gate_up, tp, "column")
        self.down_proj = ParallelLinear(down, tp, "row")

    def forward(self, value):
        return self.down_proj(swiglu(self.gate_up_proj(value)))


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

    @property
    def vocab_start(self):
        return min(self.tp.rank * self.weight.shape[0], self.vocab_size)

    def local_logits(self, value):
        width = max(0, min(self.weight.shape[0], self.vocab_size - self.vocab_start))
        return self._project(value)[..., :width]


def _weight(module):
    weight = module.weight
    if weight.dtype not in {torch.bfloat16, torch.float32} or weight.ndim != 2:
        raise ValueError("unquantized, unpacked BF16/FP32 native linear weights required")
    return weight


def from_native_model(target, config, *, tp, block_size, mask_token_id, training_gdn_backend="torch"):
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
    text.norm = SharedRMSNorm(native.norm.weight, config.rms_norm_eps, native_residual=True)
    shaped.lm_head = SharedHead(_weight(language.lm_head), tp, config.vocab_size)
    # 只新建很小的 RoPE 缓冲；backbone 参数均引用当前 TP rank 的原生权重。
    text.rotary_emb = type(text.rotary_emb)(local).to(native.norm.weight.device)
    for index, (layer, source) in enumerate(zip(text.layers, native.layers, strict=True)):
        layer.input_layernorm = SharedRMSNorm(source.input_layernorm.weight, config.rms_norm_eps,
                                             native_residual=index > 0)
        layer.post_attention_layernorm = SharedRMSNorm(source.post_attention_layernorm.weight, config.rms_norm_eps,
                                                      native_residual=True)
        layer.mlp = SharedMLP(_weight(source.mlp.gate_up_proj), _weight(source.mlp.down_proj), tp)
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
            configure_attention(mixer, getattr(ar, "rotary_emb", None))
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
            configure_gdn(mixer, training_backend=training_gdn_backend)
    model = Qwen38DualView(shaped, block_size=block_size, mask_token_id=mask_token_id,
                           hybrid_policy="orthrus_qwen35_reference")
    # 映射不完整就报错，不能用随机参数掩盖缺失的权重。
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("native weight mapping left unmapped meta parameters")
    model.source = {"model": str(getattr(target, "prefix", "native-vllm")),
                    "tp_size": tp.size, "tp_rank": tp.rank, "history_backend": "unbound"}
    return model


class NativeSuffixLearner(SuffixLearner):
    """TP-aware online replay against an immutable native vLLM target."""
    def __init__(self, model, config, tp, *, vocab_parallel_loss=False):
        self.tp = tp
        self.vocab_parallel_loss = vocab_parallel_loss and tp.size > 1
        super().__init__(model, config)
        if self.parameters[0].device.type == "cpu":
            self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate,
                                               weight_decay=0, fused=True)
        self.replicated = {name for name in self.master if ".q_norm." in name or ".k_norm." in name
                           or ".norm." in name}

    def replay_logits(self, item):
        if not self.vocab_parallel_loss:
            return super().replay_logits(item)
        return self.model.forward_suffix(item.boundary, cache=item.cache, logit_range=(0, item.valid),
            draft_weights=self.master, checkpoint_layers=self.config.checkpoint_layers,
            logit_head=self.model.head.local_logits).logits[0]

    def replay_divergence(self, student, teacher):
        if not self.vocab_parallel_loss:
            return super().replay_divergence(student, teacher)
        from .qwen38_loss import vocab_divergence
        return vocab_divergence(student, teacher, start=self.model.head.vocab_start,
                                 tp=self.tp, kind=self.config.loss)

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
        names = [name for name in sorted(self.replicated) if self.master[name].grad is not None]
        gradients = [self.master[name].grad for name in names]
        if gradients and self.tp.size > 1:
            # 所有复制的 norm 参数只做一次小通信；CPU offload 仍仅搬运这一小包。
            device = next(iter(self.execution.values())).device
            packed = torch.cat([gradient.reshape(-1) for gradient in gradients]).to(device)
            packed = self.tp.reduce(packed).to(gradients[0].device)
            for name, gradient, reduced in zip(names, gradients, packed.split([g.numel() for g in gradients]),
                                                strict=True):
                self.master[name].grad = reduced.reshape_as(gradient)
        return loss

    def clip_gradients(self):
        total = torch.zeros((), device=self.parameters[0].device, dtype=torch.float32)
        gradients = [value.grad.float() for name, value in self.master.items()
                     if value.grad is not None and (name not in self.replicated or self.tp.rank == 0)]
        if gradients and total.device.type == "npu":
            # 现成多张量归约，避免逐个大矩阵创建平方副本。
            total = torch.stack(torch._foreach_norm(gradients, 2)).square().sum()
        else:
            for gradient in gradients:
                total += (torch.linalg.vector_norm(gradient, dim=-1).square().sum()
                          if gradient.device.type == "cpu" else gradient.square().sum())
        collective_device = next(iter(self.execution.values())).device
        norm = self.tp.reduce(total.to(collective_device)).sqrt().to(total.device)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite TP draft gradient norm")
        scale = (self.config.clip_norm / (norm + 1e-6)).clamp(max=1.)
        for value in self.parameters:
            if value.grad is not None:
                value.grad.mul_(scale)
        return norm
