# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：公共模型输出与边界类型，以及普通 Transformer 参考 backbone；Qwen 主要复用 BackboneOutput / DraftBoundary。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Independent Qwen3 dual-view backbone. Persistent KV always belongs to AR."""

# 依赖：从 __future__ 导入 annotations；供本模块调用。
from __future__ import annotations

# 依赖：从 dataclasses 导入 asdict, dataclass；供本模块调用。
from dataclasses import asdict, dataclass
# 依赖：导入 math。
import math

# 依赖：导入 torch。
import torch
# 依赖：从 torch 导入 Tensor, nn；供本模块调用。
from torch import Tensor, nn
# 依赖：从 torch.nn 导入 functional；供本模块调用。
from torch.nn import functional as F

# 依赖：从 ..state 导入 Cache, cache_length；供本模块调用。
from ..state import Cache, cache_length


@dataclass(frozen=True)
# ── DualViewConfig ──
# 职责：类型 `DualViewConfig`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class DualViewConfig:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    vocab_size: int = 32
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    hidden_size: int = 64
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    intermediate_size: int = 128
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    num_hidden_layers: int = 2
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    num_attention_heads: int = 4
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    num_key_value_heads: int = 2
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    head_dim: int = 16
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    rms_norm_eps: float = 1e-6
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    rope_theta: float = 1000000.0
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    attention_bias: bool = False
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    tie_word_embeddings: bool = True
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    block_size: int = 4
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    mask_token_id: int = 1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    eos_token_id: int | None = None

    # ── DualViewConfig.__post_init__ ──
    # 职责：数据类构造后校验配置，拒绝无法支持的参数组合。
    def __post_init__(self):
        dimensions = (self.vocab_size, self.hidden_size, self.intermediate_size,
                      self.num_hidden_layers, self.num_attention_heads,
                      self.num_key_value_heads, self.head_dim)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive model dimensions required。
        if any(v < 1 for v in dimensions):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive model dimensions required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：even head width and integral query/KV ratio required。
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("even head width and integral query/KV ratio required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid mask token or block size。
        if self.block_size < 2 or not 0 <= self.mask_token_id < self.vocab_size:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid mask token or block size")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid EOS token。
        if self.eos_token_id is not None and not 0 <= self.eos_token_id < self.vocab_size:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid EOS token")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive finite normalization epsilon required。
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive finite normalization epsilon required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：finite rotary base greater than one required。
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("finite rotary base greater than one required")

    # ── DualViewConfig.to_dict ──
    # 职责：把参考模型配置转为可保存的字典。
    def to_dict(self):
        return asdict(self)

    @classmethod
    # ── DualViewConfig.from_public ──
    # 职责：从外部模型配置抽取普通参考 backbone 所需的配置字段。
    def from_public(cls, data):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：the dual-view importer requires a Qwen3 checkpoint。
        if data.get("model_type") not in ("qwen3", "orthrus"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("the dual-view importer requires a Qwen3 checkpoint")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：SwiGLU activation required。
        if data.get("hidden_act", "silu") != "silu":
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("SwiGLU activation required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：this backbone implements full attention。
        if data.get("use_sliding_window", False) or any(
                v != "full_attention" for v in data.get("layer_types", [])):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("this backbone implements full attention")
        rope = data.get("rope_parameters") or data.get("rope_scaling") or {}
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：the Qwen3 reproduction uses ordinary RoPE。
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("the Qwen3 reproduction uses ordinary RoPE")
        names = cls.__dataclass_fields__
        values = {key: data[key] for key in names if key in data}
        if "rope_theta" in rope:
            values["rope_theta"] = rope["rope_theta"]
        required = set(names) - {"attention_bias", "tie_word_embeddings", "eos_token_id"}
        if required - values.keys():
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError(f"missing architecture fields: {sorted(required - values.keys())}")
        return cls(**values)


# ── RMSNorm ──
# 职责：类型 `RMSNorm`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class RMSNorm(nn.Module):
    # ── RMSNorm.__init__ ──
    # 职责：初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。
    # 输入提示：width：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    # ── RMSNorm.forward ──
    # 职责：普通参考 backbone 的 RMSNorm；Qwen native 使用 qwen38_weights.SharedRMSNorm。
    # 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
    def forward(self, hidden):
        # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
        work = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.epsilon)
        return normalized.to(hidden.dtype) * self.weight


# ── AttentionView ──
# 职责：类型 `AttentionView`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class AttentionView(nn.Module):
    """One set of projections and per-head Q/K normalization."""

    # ── AttentionView.__init__ ──
    # 职责：初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。
    def __init__(self, config):
        super().__init__()
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        self.q = nn.Linear(config.hidden_size, q_width, bias=config.attention_bias)
        self.k = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)


# ── _rotate ──
# 职责：执行 RoPE 所需的向量分半旋转；不改变 token 的序列顺序。
def _rotate(x, cosine, sine):
    left, right = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-right, left), dim=-1) * sine


# ── DualAttention ──
# 职责：类型 `DualAttention`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class DualAttention(nn.Module):
    # ── DualAttention.__init__ ──
    # 职责：初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ar = AttentionView(config)
        self.draft = AttentionView(config)

    # ── DualAttention.forward ──
    # 职责：普通全 Attention 参考模型的 AR/draft QKV 与可见性计算；不是 Qwen GDN。
    # 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID；allowed：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
    def forward(self, hidden, view, cache, rotary, allowed, backend):
        config = self.config
        b, length, _ = hidden.shape
        projection = getattr(self, view)
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。 此处交换张量维度以适配 head/token 或 K/V 布局。
        q = projection.q_norm(projection.q(hidden).view(b, length, -1, config.head_dim)).transpose(1, 2)
        # 计算：交换张量维度以适配 head/token 或 K/V 布局。
        k = projection.k_norm(projection.k(hidden).view(b, length, -1, config.head_dim)).transpose(1, 2)
        # 计算：交换张量维度以适配 head/token 或 K/V 布局。
        v = projection.v(hidden).view(b, length, -1, config.head_dim).transpose(1, 2)
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q, k = _rotate(q, *rotary), _rotate(k, *rotary)
        if cache is not None:
            # 计算：沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        if backend == "flash_attention_2":
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：explicit training masks use eager or SDPA attention。
            if allowed is not None:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("explicit training masks use eager or SDPA attention")
            # 依赖：从 flash_attn 导入 flash_attn_func；供本模块调用。
            from flash_attn import flash_attn_func
            # 计算：交换张量维度以适配 head/token 或 K/V 布局。
            attended = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                      causal=view == "ar", softmax_scale=config.head_dim ** -0.5)
        elif backend == "sdpa":
            # CUDA SDPA uses its fused GQA path for unmasked attention. Explicit
            # masks use matched Q/K/V head counts to keep the fused mask path.
            repeats = config.num_attention_heads // config.num_key_value_heads
            k_attn, v_attn = k, v
            if allowed is not None and repeats > 1:
                k_attn, v_attn = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            # 计算：交换张量维度以适配 head/token 或 K/V 布局。
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=allowed, enable_gqa=allowed is None,
                is_causal=view == "ar" and allowed is None and length > 1,
                scale=config.head_dim ** -0.5).transpose(1, 2)
        else:
            repeats = config.num_attention_heads // config.num_key_value_heads
            k_full, v_full = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            # 计算：交换张量维度以适配 head/token 或 K/V 布局。
            scores = (q @ k_full.transpose(-2, -1)) * config.head_dim ** -0.5
            if allowed is not None:
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
            masses = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            # 计算：交换张量维度以适配 head/token 或 K/V 布局。
            attended = (masses @ v_full).transpose(1, 2)
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return projection.o(attended.reshape(b, length, -1).contiguous()), (k, v)


# ── DualLayer ──
# 职责：类型 `DualLayer`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class DualLayer(nn.Module):
    # ── DualLayer.__init__ ──
    # 职责：初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。
    def __init__(self, config):
        super().__init__()
        self.attention = DualAttention(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # 数据：门控特征；Attention 输出乘 sigmoid(gate)，GDN norm 使用相应 SiLU gate。
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    # ── DualLayer.forward ──
    # 职责：普通参考 block 的 norm、Attention、MLP 与残差计算。
    # 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID；allowed：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
    def forward(self, hidden, view, cache, rotary, allowed, backend):
        attention, new_cache = self.attention(self.input_norm(hidden), view, cache, rotary, allowed, backend)
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = hidden + attention
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
        normalized = self.post_norm(hidden)
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), new_cache


@dataclass
# ── DraftBoundary ──
# 职责：类型 `DraftBoundary`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class DraftBoundary:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    hidden: Tensor
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    rotary: tuple[Tensor, Tensor]
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    allowed: Tensor | None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    start_layer: int
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    history_length: int
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    backend: str

    # ── DraftBoundary.detached ──
    # 职责：断开旧计算图，并按需要保留 cache 后缀和复制反馈张量；之后训练会新建后缀计算图。
    def detached(self):
        return DraftBoundary(self.hidden.detach().clone(), tuple(x.detach().clone() for x in self.rotary),
                             None if self.allowed is None else self.allowed.detach().clone(),
                             self.start_layer, self.history_length, self.backend)


@dataclass
# ── BackboneOutput ──
# 职责：类型 `BackboneOutput`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class BackboneOutput:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    hidden: Tensor
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    cache: Cache | None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    logits: Tensor | None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    boundary: DraftBoundary | None = None


# ── DualViewDecoder ──
# 职责：类型 `DualViewDecoder`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class DualViewDecoder(nn.Module):
    """Two attention views over shared embeddings, MLPs, norms and output head."""

    # ── DualViewDecoder.__init__ ──
    # 职责：初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。
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

    # ── DualViewDecoder.tie_weights ──
    # 职责：普通参考模型中绑定输入 embedding 与输出词表 head 的权重。
    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.head.weight = self.embedding.weight

    # ── DualViewDecoder._frequencies ──
    # 职责：为普通参考模型构造 RoPE 所需频率/位置旋转信息。
    def _frequencies(self, device):
        # Form the canonical FP32 table on the host before device transfer.
        # CPU and CUDA power kernels can round the same frequency differently.
        indices = torch.arange(0, self.config.head_dim, 2, device="cpu", dtype=torch.float32)
        return (1.0 / (self.config.rope_theta ** (indices / self.config.head_dim))).to(device)

    # ── DualViewDecoder._apply ──
    # 职责：适配 nn.Module 设备/精度变换，同时维护共享对象与源模型引用。
    def _apply(self, fn, recurse=True):
        # 调用：适配 nn.Module 设备/精度变换，同时维护共享对象与源模型引用。
        super()._apply(fn, recurse=recurse)
        self.frequencies = self._frequencies(self.embedding.weight.device)
        return self

    @torch.no_grad()
    # ── DualViewDecoder.initialize_draft_from_ar ──
    # 职责：把普通参考模型的 AR Attention 参数复制到 draft 作为初始化；不代表已经蒸馏训练。
    def initialize_draft_from_ar(self):
        for layer in self.layers:
            # 调用：校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。
            layer.attention.draft.load_state_dict(layer.attention.ar.state_dict())

    # ── DualViewDecoder.train_draft_only ──
    # 职责：普通参考模型设置只训练 draft 的参数开关；native learner 则使用自己的 master 参数。
    def train_draft_only(self):
        self.requires_grad_(False)
        for layer in self.layers:
            layer.attention.draft.requires_grad_(True)
        return self.train()

    # ── DualViewDecoder.set_backend ──
    # 职责：配置双视图模型的 Attention 实现选择，保持显式后端约定。
    def set_backend(self, backend):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unknown attention backend。
        if backend not in ("eager", "sdpa", "flash_attention_2"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("unknown attention backend")
        self.backend = backend
        return self

    # ── DualViewDecoder.forward ──
    # 职责：普通参考 backbone 的整网前向；Qwen 使用 Qwen38DualView，二者不要混淆。
    # 输入提示：positions：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE；allowed：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
    def forward(self, tokens, *, view="ar", cache=None, positions=None, allowed=None,
                logits_to_keep=0, compute_logits=True, capture_layer=None):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：view must be ar or draft。
        if view not in ("ar", "draft"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("view must be ar or draft")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonempty tokens[batch,time] required。
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("nonempty tokens[batch,time] required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：draft capture requires a valid layer boundary。
        if capture_layer is not None and (type(capture_layer) is not int or view != "draft"
                                         or not 0 <= capture_layer < len(self.layers)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("draft capture requires a valid layer boundary")
        batch, length = tokens.shape
        # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
        past = cache_length(cache)
        if cache is not None:
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：cache layer count mismatch。
            if len(cache) != len(self.layers):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("cache layer count mismatch")
            expected = (batch, self.config.num_key_value_heads, past, self.config.head_dim)
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：cache shape mismatch。
            if any(k.shape != expected or v.shape != expected for k, v in cache):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("cache shape mismatch")
        if positions is None:
            # 数据：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE。
            positions = torch.arange(past, past + length, device=tokens.device)[None, :]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positions must have one row or one row per batch。
        if positions.shape not in ((1, length), (batch, length)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positions must have one row or one row per batch")
        if allowed is not None:
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：allowed must be Boolean [batch,1,query,key]。
            if allowed.dtype != torch.bool or allowed.shape not in (
                    (1, 1, length, past + length), (batch, 1, length, past + length)):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("allowed must be Boolean [batch,1,query,key]")
        # 视图选路：AR 使用因果教师/历史提交语义，draft 使用独立 mixer 和临时块语义。
        elif view == "ar" and (self.backend == "eager" or (
                self.backend == "sdpa" and past > 0 and length > 1)):
            # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局。
            query = torch.arange(past, past + length, device=tokens.device)
            keys = torch.arange(past + length, device=tokens.device)
            # 数据：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
            allowed = (keys[None, :] <= query[:, None])[None, None]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：logits_to_keep must be a nonnegative row count。
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("logits_to_keep must be a nonnegative row count")
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = self.embedding(tokens)
        # RoPE frequencies retain FP32 precision under model.to(BF16).
        # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
        phase = positions.float()[..., None] * self.frequencies
        # 计算：沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
        phase = torch.cat((phase, phase), -1)[:, None]
        # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        rotary = phase.cos().to(hidden.dtype), phase.sin().to(hidden.dtype)
        # 数据：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
        new_cache, boundary = [], None
        for index, layer in enumerate(self.layers):
            if index == capture_layer:
                # 数据：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
                boundary = DraftBoundary(hidden, rotary, allowed, index, past, self.backend)
            # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
            hidden, kv = layer(hidden, view, None if cache is None else cache[index],
                               rotary, allowed, self.backend)
            # 视图选路：AR 使用因果教师/历史提交语义，draft 使用独立 mixer 和临时块语义。
            if view == "ar":
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                new_cache.append(kv)
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = self.norm(hidden)
        rows = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        # 数据：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
        logits = self.head(rows) if compute_logits else None
        return BackboneOutput(hidden, tuple(new_cache) if view == "ar" else cache, logits, boundary)

    # ── DualViewDecoder.forward_suffix ──
    # 职责：从 boundary.hidden 开始执行待训练后缀；用函数式参数替换注入 master，可选逐层 checkpoint 和局部词表 head。
    # 输入提示：boundary：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
    def forward_suffix(self, boundary, *, cache=None, logit_range=None, draft_weights=None,
                       checkpoint_layers=False):
        """Replay a frozen-prefix boundary; optional draft weights carry master gradients."""
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：matching draft boundary, attention backend and historical cache required。
        if (not isinstance(boundary, DraftBoundary) or not 0 <= boundary.start_layer < len(self.layers)
                or boundary.backend != self.backend or cache_length(cache) != boundary.history_length):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("matching draft boundary, attention backend and historical cache required")
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = boundary.hidden
        # 数据：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
        batch, length, width = hidden.shape
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = len(self.layers) - boundary.start_layer
        expected = (batch, self.config.num_key_value_heads, boundary.history_length, self.config.head_dim)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：suffix cache and boundary shapes must agree。
        if (width != self.config.hidden_size or (cache is not None and (
                len(cache) != count or any(k.shape != expected or v.shape != expected for k, v in cache)))):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("suffix cache and boundary shapes must agree")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：logit range must select nonempty boundary rows。
        if logit_range is not None and (len(logit_range) != 2 or not 0 <= logit_range[0] < logit_range[1] <= length):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("logit range must select nonempty boundary rows")
        overrides = {} if draft_weights is None else draft_weights
        allowed_keys = {f"layers.{i}.attention.draft.{name}"
                        for i in range(boundary.start_layer, len(self.layers))
                        for name, _ in self.layers[i].attention.draft.named_parameters()}
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：suffix overrides are restricted to its draft attention parameters。
        if overrides.keys() - allowed_keys:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("suffix overrides are restricted to its draft attention parameters")
        for offset, layer in enumerate(self.layers[boundary.start_layer:]):
            prefix = f"layers.{boundary.start_layer + offset}."
            weights = {name[len(prefix):]: value for name, value in overrides.items() if name.startswith(prefix)}

            # ── DualViewDecoder.forward_suffix.execute ──
            # 职责：实现 DualViewDecoder.forward_suffix.execute；输入见下方签名，具体处理与返回值逐语句标注。
            # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；parameters：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数；history：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态。
            def execute(value, module=layer, parameters=weights, history=None if cache is None else cache[offset]):
                resident = dict(module.named_parameters())
                # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
                staged = {name: weight.to(device=resident[name].device, dtype=resident[name].dtype)
                          for name, weight in parameters.items()}
                args = (value, "draft", history, boundary.rotary, boundary.allowed, self.backend)
                return (torch.func.functional_call(module, staged, args) if staged else module(*args))[0]

            # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
            if checkpoint_layers and torch.is_grad_enabled():
                # 依赖：从 torch.utils.checkpoint 导入 checkpoint；供本模块调用。
                from torch.utils.checkpoint import checkpoint
                # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。 此处前向少存激活，反向时重算，换取显存节省。
                hidden = checkpoint(execute, hidden, use_reentrant=False)
            else:
                # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
                hidden = execute(hidden)
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = self.norm(hidden)
        rows = hidden if logit_range is None else hidden[:, logit_range[0]:logit_range[1]]
        return BackboneOutput(hidden, cache, self.head(rows))
