"""Independent Qwen3 dual-view backbone. Persistent KV always belongs to AR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..state import Cache, cache_length


@dataclass(frozen=True)
class DualViewConfig:
    vocab_size: int = 32
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 4
    mask_token_id: int = 1
    eos_token_id: int | None = None

    def __post_init__(self):
        dimensions = (self.vocab_size, self.hidden_size, self.intermediate_size,
                      self.num_hidden_layers, self.num_attention_heads,
                      self.num_key_value_heads, self.head_dim)
        if any(v < 1 for v in dimensions):
            raise ValueError("positive model dimensions required")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("even head width and integral query/KV ratio required")
        if self.block_size < 2 or not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("invalid mask token or block size")
        if self.eos_token_id is not None and not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("invalid EOS token")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("positive finite normalization epsilon required")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("finite rotary base greater than one required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_public(cls, data):
        if data.get("model_type") not in ("qwen3", "orthrus"):
            raise ValueError("the dual-view importer requires a Qwen3 checkpoint")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("SwiGLU activation required")
        if data.get("use_sliding_window", False) or any(
                v != "full_attention" for v in data.get("layer_types", [])):
            raise ValueError("this backbone implements full attention")
        rope = data.get("rope_parameters") or data.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("the Qwen3 reproduction uses ordinary RoPE")
        names = cls.__dataclass_fields__
        values = {key: data[key] for key in names if key in data}
        if "rope_theta" in rope:
            values["rope_theta"] = rope["rope_theta"]
        required = set(names) - {"attention_bias", "tie_word_embeddings", "eos_token_id"}
        if required - values.keys():
            raise ValueError(f"missing architecture fields: {sorted(required - values.keys())}")
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, hidden):
        work = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.epsilon)
        return normalized.to(hidden.dtype) * self.weight


class AttentionView(nn.Module):
    """One set of projections and per-head Q/K normalization."""

    def __init__(self, config):
        super().__init__()
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q = nn.Linear(config.hidden_size, q_width, bias=config.attention_bias)
        self.k = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)


def _rotate(x, cosine, sine):
    left, right = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-right, left), dim=-1) * sine


class DualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ar = AttentionView(config)
        self.draft = AttentionView(config)

    def forward(self, hidden, view, cache, rotary, allowed, backend):
        config = self.config
        b, length, _ = hidden.shape
        projection = getattr(self, view)
        q = projection.q_norm(projection.q(hidden).view(b, length, -1, config.head_dim)).transpose(1, 2)
        k = projection.k_norm(projection.k(hidden).view(b, length, -1, config.head_dim)).transpose(1, 2)
        v = projection.v(hidden).view(b, length, -1, config.head_dim).transpose(1, 2)
        q, k = _rotate(q, *rotary), _rotate(k, *rotary)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        if backend == "flash_attention_2":
            if allowed is not None:
                raise ValueError("explicit training masks use eager or SDPA attention")
            from flash_attn import flash_attn_func
            attended = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                      causal=view == "ar", softmax_scale=config.head_dim ** -0.5)
        elif backend == "sdpa":
            # CUDA SDPA uses its fused GQA path for unmasked attention. Explicit
            # masks use matched Q/K/V head counts to keep the fused mask path.
            repeats = config.num_attention_heads // config.num_key_value_heads
            k_attn, v_attn = k, v
            if allowed is not None and repeats > 1:
                k_attn, v_attn = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=allowed, enable_gqa=allowed is None,
                is_causal=view == "ar" and allowed is None and length > 1,
                scale=config.head_dim ** -0.5).transpose(1, 2)
        else:
            repeats = config.num_attention_heads // config.num_key_value_heads
            k_full, v_full = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            scores = (q @ k_full.transpose(-2, -1)) * config.head_dim ** -0.5
            if allowed is not None:
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            masses = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            attended = (masses @ v_full).transpose(1, 2)
        return projection.o(attended.reshape(b, length, -1).contiguous()), (k, v)


class DualLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DualAttention(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, view, cache, rotary, allowed, backend):
        attention, new_cache = self.attention(self.input_norm(hidden), view, cache, rotary, allowed, backend)
        hidden = hidden + attention
        normalized = self.post_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), new_cache


@dataclass
class DraftBoundary:
    hidden: Tensor
    rotary: tuple[Tensor, Tensor]
    allowed: Tensor | None
    start_layer: int
    history_length: int
    backend: str

    def detached(self):
        return DraftBoundary(self.hidden.detach().clone(), tuple(x.detach().clone() for x in self.rotary),
                             None if self.allowed is None else self.allowed.detach().clone(),
                             self.start_layer, self.history_length, self.backend)


@dataclass
class BackboneOutput:
    hidden: Tensor
    cache: Cache | None
    logits: Tensor | None
    boundary: DraftBoundary | None = None


class DualViewDecoder(nn.Module):
    """Two attention views over shared embeddings, MLPs, norms and output head."""

    def __init__(self, config=DualViewConfig()):
        super().__init__()
        self.config = config
        self.backend = "sdpa"
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DualLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.register_buffer("frequencies", self._frequencies(self.embedding.weight.device), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.tie_weights()
        self.initialize_draft_from_ar()

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.head.weight = self.embedding.weight

    def _frequencies(self, device):
        # Form the canonical FP32 table on the host before device transfer.
        # CPU and CUDA power kernels can round the same frequency differently.
        indices = torch.arange(0, self.config.head_dim, 2, device="cpu", dtype=torch.float32)
        return (1.0 / (self.config.rope_theta ** (indices / self.config.head_dim))).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.frequencies = self._frequencies(self.embedding.weight.device)
        return self

    @torch.no_grad()
    def initialize_draft_from_ar(self):
        for layer in self.layers:
            layer.attention.draft.load_state_dict(layer.attention.ar.state_dict())

    def train_draft_only(self):
        self.requires_grad_(False)
        for layer in self.layers:
            layer.attention.draft.requires_grad_(True)
        return self.train()

    def set_backend(self, backend):
        if backend not in ("eager", "sdpa", "flash_attention_2"):
            raise ValueError("unknown attention backend")
        self.backend = backend
        return self

    def forward(self, tokens, *, view="ar", cache=None, positions=None, allowed=None,
                logits_to_keep=0, compute_logits=True, capture_layer=None):
        if view not in ("ar", "draft"):
            raise ValueError("view must be ar or draft")
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("nonempty tokens[batch,time] required")
        if capture_layer is not None and (type(capture_layer) is not int or view != "draft"
                                         or not 0 <= capture_layer < len(self.layers)):
            raise ValueError("draft capture requires a valid layer boundary")
        batch, length = tokens.shape
        past = cache_length(cache)
        if cache is not None:
            if len(cache) != len(self.layers):
                raise ValueError("cache layer count mismatch")
            expected = (batch, self.config.num_key_value_heads, past, self.config.head_dim)
            if any(k.shape != expected or v.shape != expected for k, v in cache):
                raise ValueError("cache shape mismatch")
        if positions is None:
            positions = torch.arange(past, past + length, device=tokens.device)[None, :]
        if positions.shape not in ((1, length), (batch, length)):
            raise ValueError("positions must have one row or one row per batch")
        if allowed is not None:
            if allowed.dtype != torch.bool or allowed.shape not in (
                    (1, 1, length, past + length), (batch, 1, length, past + length)):
                raise ValueError("allowed must be Boolean [batch,1,query,key]")
        elif view == "ar" and (self.backend == "eager" or (
                self.backend == "sdpa" and past > 0 and length > 1)):
            query = torch.arange(past, past + length, device=tokens.device)
            keys = torch.arange(past + length, device=tokens.device)
            allowed = (keys[None, :] <= query[:, None])[None, None]
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative row count")
        hidden = self.embedding(tokens)
        # RoPE frequencies retain FP32 precision under model.to(BF16).
        phase = positions.float()[..., None] * self.frequencies
        phase = torch.cat((phase, phase), -1)[:, None]
        rotary = phase.cos().to(hidden.dtype), phase.sin().to(hidden.dtype)
        new_cache, boundary = [], None
        for index, layer in enumerate(self.layers):
            if index == capture_layer:
                boundary = DraftBoundary(hidden, rotary, allowed, index, past, self.backend)
            hidden, kv = layer(hidden, view, None if cache is None else cache[index],
                               rotary, allowed, self.backend)
            if view == "ar":
                new_cache.append(kv)
        hidden = self.norm(hidden)
        rows = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        logits = self.head(rows) if compute_logits else None
        return BackboneOutput(hidden, tuple(new_cache) if view == "ar" else cache, logits, boundary)

    def forward_suffix(self, boundary, *, cache=None, logit_range=None, draft_weights=None):
        """Replay a frozen-prefix boundary; optional draft weights carry master gradients."""
        if (not isinstance(boundary, DraftBoundary) or not 0 <= boundary.start_layer < len(self.layers)
                or boundary.backend != self.backend or cache_length(cache) != boundary.history_length):
            raise ValueError("matching draft boundary, attention backend and historical cache required")
        hidden = boundary.hidden
        batch, length, width = hidden.shape
        count = len(self.layers) - boundary.start_layer
        expected = (batch, self.config.num_key_value_heads, boundary.history_length, self.config.head_dim)
        if (width != self.config.hidden_size or (cache is not None and (
                len(cache) != count or any(k.shape != expected or v.shape != expected for k, v in cache)))):
            raise ValueError("suffix cache and boundary shapes must agree")
        if logit_range is not None and (len(logit_range) != 2 or not 0 <= logit_range[0] < logit_range[1] <= length):
            raise ValueError("logit range must select nonempty boundary rows")
        overrides = {} if draft_weights is None else draft_weights
        allowed_keys = {f"layers.{i}.attention.draft.{name}"
                        for i in range(boundary.start_layer, len(self.layers))
                        for name, _ in self.layers[i].attention.draft.named_parameters()}
        if overrides.keys() - allowed_keys:
            raise ValueError("suffix overrides are restricted to its draft attention parameters")
        for offset, layer in enumerate(self.layers[boundary.start_layer:]):
            prefix = f"layers.{boundary.start_layer + offset}."
            weights = {name[len(prefix):]: value for name, value in overrides.items() if name.startswith(prefix)}
            args = (hidden, "draft", None if cache is None else cache[offset],
                    boundary.rotary, boundary.allowed, self.backend)
            hidden, _ = (torch.func.functional_call(layer, weights, args) if weights else layer(*args))
        hidden = self.norm(hidden)
        rows = hidden if logit_range is None else hidden[:, logit_range[0]:logit_range[1]]
        return BackboneOutput(hidden, cache, self.head(rows))
