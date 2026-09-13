"""A small, explicit causal Transformer with token-gated low-rank projections."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import ATTENTION_BACKENDS, GROUPED_QUERY_LIMIT, grouped_attention
from blockspec.state import Cache as Cache, PackedCache as PackedCache, cache_length as cache_length, trim_cache as trim_cache


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class DraftBoundary:
    """Input to a suffix of layers, captured during the ordinary draft pass."""

    hidden: Tensor
    positions: Tensor
    allowed: Tensor
    adapter_mask: Tensor | None
    start_layer: int

    def detached(self):
        return DraftBoundary(self.hidden.detach().clone(), self.positions.detach().clone(),
                             self.allowed.detach().clone(),
                             None if self.adapter_mask is None else self.adapter_mask.detach().clone(),
                             self.start_layer)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rms_norm_eps: float = 1e-6
    norm_groups: int = 1
    norm_weight_in_fp32: bool = False
    query_key_norm: bool = False
    attention_bias: bool = False
    attention_output_bias: bool = False
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    rope_original_length: int = 8192
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_attention_factor: float = 1.0
    rope_truncate: bool = True
    adapter_rank: int = 8
    adapter_alpha: float = 8.0
    adapter_targets: tuple[str, ...] = PROJECTIONS

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "norm_groups"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("even rotary head dimension and integral query/KV ratio required")
        if self.hidden_size % self.norm_groups or self.adapter_rank < 0:
            raise ValueError("invalid norm groups or adapter rank")
        if not set(self.adapter_targets).issubset(PROJECTIONS):
            raise ValueError("unknown adapter projection")
        if len(set(self.adapter_targets)) != len(self.adapter_targets):
            raise ValueError("duplicate adapter projection")
        if not all(math.isfinite(v) and v > 0 for v in (
            self.rms_norm_eps, self.rope_theta, self.rope_factor, self.rope_attention_factor,
            self.rope_beta_fast, self.rope_beta_slow, self.adapter_alpha,
        )):
            raise ValueError("normalization, rotation, and adapter scales must be finite and positive")
        if self.rope_theta <= 1 or self.rope_factor < 1:
            raise ValueError("rotary base > 1 and extension factor >= 1 required")
        if self.rope_beta_fast <= self.rope_beta_slow or self.rope_original_length < 1:
            raise ValueError("invalid rotary correction interval")

    def to_dict(self):
        result = asdict(self)
        result["adapter_targets"] = list(self.adapter_targets)
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "adapter_targets" in value:
            value["adapter_targets"] = tuple(value["adapter_targets"])
        return cls(**value)


class GatedLinear(nn.Module):
    """W x + m (alpha/r) B A x. A mask of None does not execute the adapter."""

    def __init__(self, inputs, outputs, rank, alpha, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs)) if bias else None
        self.rank = rank
        self.scale = alpha / rank if rank else 0.0
        if rank:
            self.lora_A = nn.Parameter(torch.randn(rank, inputs) / math.sqrt(inputs))
            self.lora_B = nn.Parameter(torch.zeros(outputs, rank))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, mask=None):
        out = F.linear(x, self.weight, self.bias)
        if self.rank and mask is not None:
            low = F.linear(x, self.lora_A.to(x.dtype))
            if not torch.is_grad_enabled() and mask.dtype == torch.bool and x.dtype in (torch.float32, torch.float64):
                # The scalar row gate commutes with the B projection. Apply it
                # in rank space, then fuse B, scaling and accumulation in GEMM.
                gated = low * mask[..., None]
                return torch.addmm(out.reshape(-1, out.shape[-1]),
                                   gated.reshape(-1, self.rank), self.lora_B.to(x.dtype).t(),
                                   beta=1, alpha=self.scale).reshape_as(out)
            delta = F.linear(low, self.lora_B.to(x.dtype))
            if self.scale == 1 and mask.dtype == torch.bool and x.dtype in (torch.float32, torch.float64):
                # Binary multiplication is exact; combine gating and residual
                # addition while loading the Boolean mask directly in the kernel.
                out = torch.addcmul(out, mask[..., None], delta)
            else:
                out = out + mask[..., None].to(x.dtype) * (delta * self.scale)
        return out


class GroupedRMSNorm(nn.Module):
    def __init__(self, size, eps, groups=1, weight_in_fp32=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps, self.groups = eps, groups
        self.weight_in_fp32 = weight_in_fp32

    def forward(self, x):
        dtype = x.dtype
        work = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        grouped = work.reshape(*x.shape[:-1], self.groups, -1)
        unit = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        if self.weight_in_fp32:
            return (unit.reshape_as(x) * self.weight.to(work.dtype)).to(dtype)
        return unit.reshape_as(x).to(dtype) * self.weight


def rotary_frequencies(config: ModelConfig, *, device=None):
    """Ordinary rotary frequencies, or the fixed YaRN interpolation ramp."""
    d = config.head_dim
    indices = torch.arange(d // 2, dtype=torch.float32, device=device)
    frequency = config.rope_theta ** (-2 * indices / d)
    if config.rope_factor == 1:
        return frequency
    def boundary(rotations):
        return d * math.log(config.rope_original_length / (2 * math.pi * rotations)) / (
            2 * math.log(config.rope_theta)
        )
    lo, hi = boundary(config.rope_beta_fast), boundary(config.rope_beta_slow)
    if config.rope_truncate:
        lo, hi = math.floor(lo), math.ceil(hi)
    lo, hi = max(0, lo), min(d - 1, hi)
    ramp = ((indices - lo) / max(hi - lo, 1e-3)).clamp(0, 1)
    return frequency * (1 - ramp + ramp / config.rope_factor)


def rotary_embeddings(positions, frequencies, dtype, attention_factor=1.0):
    """The same position-only trigonometric values are shared by every layer."""
    angles = positions.float()[..., None] * frequencies
    phase = torch.cat((angles, angles), dim=-1)[:, None, :, :]
    return (phase.cos() * attention_factor).to(dtype), (phase.sin() * attention_factor).to(dtype)


def apply_rotary(x, embeddings):
    # A projection may be autocast even when the embedding/normalization was
    # FP32. Never let a shared FP32 position table promote that projection.
    cosine, sine = (value.to(x.dtype) for value in embeddings)
    half = x.shape[-1] // 2
    perpendicular = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + perpendicular * sine


def rotate(x, positions, frequencies, attention_factor=1.0):
    """Standalone reference operation; full Decoder calls share embeddings."""
    return apply_rotary(x, rotary_embeddings(positions, frequencies, x.dtype, attention_factor))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backend = "sdpa"
        for name, outputs in (("q_proj", config.num_attention_heads * config.head_dim),
                              ("k_proj", config.num_key_value_heads * config.head_dim),
                              ("v_proj", config.num_key_value_heads * config.head_dim)):
            setattr(self, name, GatedLinear(config.hidden_size, outputs,
                    config.adapter_rank if name in config.adapter_targets else 0,
                    config.adapter_alpha, config.attention_bias))
        self.o_proj = GatedLinear(config.num_attention_heads * config.head_dim,
                                 config.hidden_size,
                                 config.adapter_rank if "o_proj" in config.adapter_targets else 0,
                                 config.adapter_alpha, config.attention_output_bias)
        if config.query_key_norm:
            self.q_norm = GroupedRMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = GroupedRMSNorm(config.head_dim, config.rms_norm_eps)
        self.register_buffer("frequencies", rotary_frequencies(config), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # model.to(BF16) must not quantize the rotary frequency table.
        self.frequencies = rotary_frequencies(self.config, device=self.q_proj.weight.device)
        return self

    def forward(self, x, positions, allowed, mask, past, rotary=None):
        b, length, _ = x.shape
        c = self.config
        q = self.q_proj(x, mask).reshape(b, length, c.num_attention_heads, c.head_dim)
        k = self.k_proj(x, mask).reshape(b, length, c.num_key_value_heads, c.head_dim)
        v = self.v_proj(x, mask).reshape(b, length, c.num_key_value_heads, c.head_dim)
        if c.query_key_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rotary is None:
            q = rotate(q.transpose(1, 2), positions, self.frequencies, c.rope_attention_factor)
            k = rotate(k.transpose(1, 2), positions, self.frequencies, c.rope_attention_factor)
        else:
            q, k = apply_rotary(q.transpose(1, 2), rotary), apply_rotary(k.transpose(1, 2), rotary)
        v = v.transpose(1, 2)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        if (self.backend == "grouped" and length <= GROUPED_QUERY_LIMIT
                and q.dtype in (torch.float32, torch.float64)):
            attended = grouped_attention(q, k, v, allowed)
        else:
            repeat = c.num_attention_heads // c.num_key_value_heads
            attended = F.scaled_dot_product_attention(
                q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
                attn_mask=allowed, dropout_p=0.0,
            )
        out = attended.transpose(1, 2).reshape(b, length, -1)
        return self.o_proj(out, mask), (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name, inputs, outputs in (
            ("gate_proj", config.hidden_size, config.intermediate_size),
            ("up_proj", config.hidden_size, config.intermediate_size),
            ("down_proj", config.intermediate_size, config.hidden_size),
        ):
            setattr(self, name, GatedLinear(inputs, outputs,
                    config.adapter_rank if name in config.adapter_targets else 0,
                    config.adapter_alpha))

    def forward(self, x, mask):
        return self.down_proj(F.silu(self.gate_proj(x, mask)) * self.up_proj(x, mask), mask)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                             config.norm_groups, config.norm_weight_in_fp32)
        self.post_attention_layernorm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                                      config.norm_groups, config.norm_weight_in_fp32)
        self.self_attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, positions, allowed, mask, past, rotary=None):
        attention, cache = self.self_attn(self.input_layernorm(x), positions, allowed, mask, past, rotary)
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x), mask), cache


class Body(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                   config.norm_groups, config.norm_weight_in_fp32)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Body(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_attention_backend(self, backend):
        """Choose execution before preparing graphs or retaining online feedback."""
        if backend not in ATTENTION_BACKENDS:
            raise ValueError("unknown attention backend")
        for layer in self.model.layers:
            layer.self_attn.backend = backend
        return self

    def attention_signature(self):
        return tuple(layer.self_attn.backend for layer in self.model.layers)

    def set_base_dtype(self, dtype):
        """Convert execution base weights after checkpoint validation, before graph/learner setup.

        Adapter tensors and FP32 rotary frequencies retain their exact stored values.
        """
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("base execution dtype must be FP32 or BF16")
        for name, parameter in self.named_parameters():
            if not is_adapter(name):
                parameter.data = parameter.data.to(dtype)
        return self

    def _rotary(self, positions, dtype):
        # Supported architectures use one fixed RoPE scheme across ALL layers.
        # This is a per-forward value, not a cache tied to the adapter version.
        return rotary_embeddings(positions, self.model.layers[0].self_attn.frequencies,
                                  dtype, self.config.rope_attention_factor)

    def _project(self, hidden, logit_range):
        if logit_range is not None:
            if (not isinstance(logit_range, tuple) or len(logit_range) != 2
                    or any(type(i) is not int for i in logit_range)
                    or not 0 <= logit_range[0] < logit_range[1] <= hidden.shape[1]):
                raise ValueError("logit range must be a nonempty start/end interval within the input")
            hidden = hidden[:, logit_range[0]:logit_range[1]]
        return self.lm_head(self.model.norm(hidden))

    def forward(self, tokens, *, positions=None, allowed=None, adapter_mask=None,
                cache: Cache | None = None, return_cache=False, capture_layer=None, logit_range=None):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape [batch, nonempty sequence]")
        b, length = tokens.shape
        prefix = cache_length(cache)
        if cache is not None and len(cache) != self.config.num_hidden_layers:
            raise ValueError("cache layer count differs from model")
        if capture_layer is not None and (type(capture_layer) is not int or not return_cache
                                           or not 0 <= capture_layer <= self.config.num_hidden_layers):
            raise ValueError("boundary capture needs return_cache and a valid layer")
        if positions is None:
            positions = torch.arange(prefix, prefix + length, device=tokens.device)[None].expand(b, -1)
        if positions.shape != tokens.shape:
            raise ValueError("position ids must match tokens")
        if adapter_mask is not None and adapter_mask.shape != tokens.shape:
            raise ValueError("adapter mask must match tokens")
        if allowed is None:
            keys = torch.arange(prefix + length, device=tokens.device)
            queries = prefix + torch.arange(length, device=tokens.device)
            allowed = (keys[None, :] <= queries[:, None])[None, None]
        if allowed.dtype != torch.bool or allowed.shape[-2:] != (length, prefix + length):
            raise ValueError("attention mask must be boolean with query/key dimensions")
        hidden = self.model.embed_tokens(tokens)
        rotary = self._rotary(positions, hidden.dtype)
        new_cache = []
        for i, layer in enumerate(self.model.layers):
            if i == capture_layer:
                boundary = DraftBoundary(hidden, positions, allowed, adapter_mask, i)
            hidden, kv = layer(hidden, positions, allowed, adapter_mask,
                               None if cache is None else cache[i], rotary)
            if return_cache:
                new_cache.append(kv)
        if capture_layer == self.config.num_hidden_layers:
            # The terminal boundary exposes final residual features for light heads.
            boundary = DraftBoundary(hidden, positions, allowed, adapter_mask, capture_layer)
        logits = self._project(hidden, logit_range)
        if capture_layer is not None:
            return logits, tuple(new_cache), boundary
        return (logits, tuple(new_cache)) if return_cache else logits

    def forward_suffix(self, boundary: DraftBoundary, *, cache: Cache | None = None, logit_range=None):
        """Exact suffix recomputation, given a still-valid frozen-prefix boundary.

        cache contains only this suffix's BASE prefix KV. The caller must freeze
        all layers before start_layer for the lifetime of the captured boundary.
        """
        start = boundary.start_layer
        if type(start) is not int or not 0 <= start < self.config.num_hidden_layers:
            raise ValueError("boundary start must be an existing integer layer")
        remaining = self.config.num_hidden_layers - start
        if cache is not None and len(cache) != remaining:
            raise ValueError("boundary/cache does not match the suffix layer count")
        hidden = boundary.hidden
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or not hidden.shape[1]:
            raise ValueError("invalid boundary hidden state")
        length = hidden.shape[1]
        if (boundary.positions.shape != hidden.shape[:2] or boundary.allowed.dtype != torch.bool
                or boundary.allowed.shape[-2:] != (length, cache_length(cache) + length)
                or (boundary.adapter_mask is not None and boundary.adapter_mask.shape != hidden.shape[:2])):
            raise ValueError("boundary layout does not match replay dimensions")
        rotary = self._rotary(boundary.positions, hidden.dtype)
        for i in range(start, self.config.num_hidden_layers):
            hidden, _ = self.model.layers[i](hidden, boundary.positions, boundary.allowed,
                                            boundary.adapter_mask, None if cache is None else cache[i - start], rotary)
        return self._project(hidden, logit_range)

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if is_adapter(n)]

    def train_adapters_only(self):
        for name, parameter in self.named_parameters():
            parameter.grad = None
            parameter.requires_grad_(is_adapter(name))
            if is_adapter(name) and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.float()
        return self

    def train_base_only(self):
        for name, parameter in self.named_parameters():
            parameter.grad = None
            parameter.requires_grad_(not is_adapter(name))
        return self


def is_adapter(name):
    return name.endswith(".lora_A") or name.endswith(".lora_B")
