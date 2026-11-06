# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：权重与 TP 适配层：引用原生 backbone 参数，复制 draft mixer，并为冻结权重输入梯度、TP collective 和 learner 提供适配。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Native vLLM BF16 TP weights exposed to the Orthrus functional executor.

No checkpoint is loaded here. Shared parameters alias the target; only draft
mixers are copied. Collectives follow column/row-parallel autograd semantics.
"""

# 依赖：导入 copy。
import copy
# 依赖：导入 hashlib。
import hashlib
# 依赖：导入 json。
import json

# 依赖：导入 torch。
import torch
# 依赖：导入 torch.distributed。
import torch.distributed as dist
# 依赖：从 torch 导入 nn；供本模块调用。
from torch import nn
# 依赖：从 torch.nn 导入 functional；供本模块调用。
from torch.nn import functional as F

# 依赖：从 ..parallel.online 导入 SuffixLearner；供本模块调用。
from ..parallel.online import SuffixLearner
# 依赖：从 ..parallel.qwen38 导入 Qwen38DualView；供本模块调用。
from ..parallel.qwen38 import Qwen38DualView
# 依赖：从 .qwen38_ops 导入 configure_attention, configure_gdn, swiglu；供本模块调用。
from .qwen38_ops import configure_attention, configure_gdn, swiglu


# ── TensorParallel ──
# 职责：类型 `TensorParallel`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class TensorParallel:
    # ── TensorParallel.__init__ ──
    # 职责：记录设备通信组、全局 rank 列表、组大小和本 rank 的组内编号。
    def __init__(self, group=None, ranks=(0,)):
        self.group, self.ranks = group, tuple(ranks)
        self.size = len(self.ranks)
        self.rank = 0 if self.size == 1 else dist.get_rank(group)

    # ── TensorParallel.__deepcopy__ ──
    # 职责：控制深复制的共享语义；返回 self 表示对象继续共享，而非创建新的通信组/缓存。
    def __deepcopy__(self, memo):
        return self

    # ── TensorParallel.reduce ──
    # 职责：TP all-reduce 求和；形状不变，部分结果变成各 rank 一致的完整结果。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def reduce(self, value):
        if self.size > 1:
            # 执行：跨 TP rank 原地求和。
            dist.all_reduce(value, group=self.group)
        # 输出：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        return value

    # ── TensorParallel.gather ──
    # 职责：把各 rank 最后一维的分片 all-gather 后拼接；常用于重建完整词表 logits。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def gather(self, value):
        if self.size == 1:
            # 输出：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
            return value
        pieces = [torch.empty_like(value) for _ in self.ranks]
        # 执行：收集各 TP 分片；使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        dist.all_gather(pieces, value.contiguous(), group=self.group)
        return torch.cat(pieces, dim=-1)

    # ── TensorParallel.broadcast ──
    # 职责：用首个 TP rank 的内容同步其他 rank，确保候选 token 一致。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def broadcast(self, value):
        if self.size > 1:
            # 调用：用首个 TP rank 的内容同步其他 rank，确保候选 token 一致。
            dist.broadcast(value, src=self.ranks[0], group=self.group)
        # 输出：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        return value


# ── _Copy ──
# 职责：类型 `_Copy`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _Copy(torch.autograd.Function):
    @staticmethod
    # ── _Copy.forward ──
    # 职责：column/vocab 并行输入在前向保持不变，记录 TP 组供反向归并。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def forward(ctx, value, tp):
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        ctx.tp = tp
        # 输出：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        return value

    @staticmethod
    # ── _Copy.backward ──
    # 职责：把不同列分片对同一输入的梯度 all-reduce；第二返回值 None 对应无梯度的 tp 对象。
    def backward(ctx, gradient):
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return ctx.tp.reduce(gradient.contiguous().clone()), None


# ── _Reduce ──
# 职责：类型 `_Reduce`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _Reduce(torch.autograd.Function):
    @staticmethod
    # ── _Reduce.forward ──
    # 职责：row 并行输出在前向 all-reduce 相加；先 clone，避免原地改动输入。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def forward(ctx, value, tp):
        return tp.reduce(value.clone())

    @staticmethod
    # ── _Reduce.backward ──
    # 职责：把共同输出的梯度直接传回本 rank 的局部投影输入，不再重复求和。
    def backward(ctx, gradient):
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return gradient, None


# ── _Gather ──
# 职责：类型 `_Gather`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _Gather(torch.autograd.Function):
    @staticmethod
    # ── _Gather.forward ──
    # 职责：在词表最后一维 all-gather 局部分数，形成完整词表 logits。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def forward(ctx, value, tp):
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        ctx.tp = tp
        return tp.gather(value)

    @staticmethod
    # ── _Gather.backward ──
    # 职责：取出全局梯度属于本 rank 词表片段的部分，不把其他 rank 的分片用于本地反向。
    def backward(ctx, gradient):
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return gradient.chunk(ctx.tp.size, dim=-1)[ctx.tp.rank].contiguous(), None


# ── _FrozenLinear ──
# 职责：类型 `_FrozenLinear`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    # ── _FrozenLinear.forward ──
    # 职责：计算 y=xWᵀ，只保存冻结权重的只读引用，避免 inference tensor 的 save_for_backward 限制。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def forward(ctx, value, weight):
        # 原生推理权重不能 save_for_backward；仅保留只读引用，反向只算输入梯度。
        ctx.weight = weight
        return F.linear(value, weight)

    @staticmethod
    # ── _FrozenLinear.backward ──
    # 职责：返回输入梯度 dL/dx=dL/dy·W，权重梯度返回 None；冻结不等于阻断输入梯度。
    def backward(ctx, gradient):
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return gradient @ ctx.weight, None


# ── ParallelLinear ──
# 职责：类型 `ParallelLinear`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class ParallelLinear(nn.Module):
    # ── ParallelLinear.__init__ ──
    # 职责：包装已经存在的 TP 权重引用，并记住 column/row/vocab 并行模式，不重新从磁盘加载矩阵。
    # 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def __init__(self, weight, tp, mode):
        super().__init__()
        self.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight, requires_grad=False)
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        self.tp, self.mode = tp, mode

    # ── ParallelLinear._project ──
    # 职责：执行本 rank 的线性投影；处理 column/vocab 输入梯度归并与冻结 inference tensor 的输入反向。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def _project(self, value):
        if self.mode in {"column", "vocab"}:
            # 数据：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
            value = _Copy.apply(value, self.tp)
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if torch.is_inference(self.weight) and torch.is_grad_enabled() and value.requires_grad:
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：target inference weights must remain frozen。
            if self.weight.requires_grad:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise RuntimeError("target inference weights must remain frozen")
            # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
            out = _FrozenLinear.apply(value, self.weight)
        else:
            # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
            out = F.linear(value, self.weight)
        # 输出：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
        return out

    # ── ParallelLinear.forward ──
    # 职责：本地矩阵乘后按 row/vocab 模式求和或拼接，column 模式保持局部输出。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def forward(self, value):
        # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
        out = self._project(value)
        if self.mode == "row":
            # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
            out = _Reduce.apply(out, self.tp)
        elif self.mode == "vocab":
            # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
            out = _Gather.apply(out, self.tp)
        # 输出：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
        return out


# ── SharedRMSNorm ──
# 职责：类型 `SharedRMSNorm`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SharedRMSNorm(nn.Module):
    # ── SharedRMSNorm.__init__ ──
    # 职责：引用冻结 norm 权重，保存 epsilon 与 residual 舍入模式；必要时建立全 1 的辅助 buffer。
    def __init__(self, weight, eps, *, native_residual=False):
        super().__init__()
        self.weight, self.eps = weight, eps
        self.native_residual = native_residual
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False):
            self.register_buffer("_unit_weight", torch.ones_like(weight) if native_residual else None,
                                 persistent=False)

    # ── SharedRMSNorm.forward ──
    # 职责：用共享冻结 norm 参数做归一化，匹配原生 residual RMSNorm 的 BF16 舍入，训练保留输入梯度。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def forward(self, value):
        weight = self.weight
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if torch.is_inference(weight) and torch.is_grad_enabled() and value.requires_grad:
            # 计算：新建独立张量存储；与只建立别名不同。
            weight = weight.clone()
        if self.native_residual and value.dtype == torch.bfloat16:
            # 对齐原生残差 norm：归一化结果和 1+weight 各先舍入一次；首层及 Q/K norm 不走此分支。
            # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
            if value.device.type == "npu":
                # 依赖：导入 torch_npu。
                import torch_npu
                # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
                normalized = torch_npu.npu_rms_norm(value, self._unit_weight, self.eps)[0]
            else:
                # 数据：归一化后的浮点特征，送入 mixer 或 MLP。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
                normalized = (value.float() * torch.rsqrt(
                    value.float().square().mean(-1, keepdim=True) + self.eps)).to(value.dtype)
            return normalized * (1 + weight)
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。 此处转为 FP32 做分数/归一化/梯度等中间计算。
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + weight.float())).to(value.dtype)


# ── SharedMLP ──
# 职责：类型 `SharedMLP`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SharedMLP(nn.Module):
    # ── SharedMLP.__init__ ──
    # 职责：用共享 gate/up 与 down 矩阵构造列并行/行并行线性层。
    # 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def __init__(self, gate_up, down, tp):
        super().__init__()
        self.gate_up_proj = ParallelLinear(gate_up, tp, "column")
        self.down_proj = ParallelLinear(down, tp, "row")

    # ── SharedMLP.forward ──
    # 职责：共享列并行 gate/up 投影→SwiGLU→行并行 down 投影；共享权重仍执行计算。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def forward(self, value):
        return self.down_proj(swiglu(self.gate_up_proj(value)))


# ── SharedEmbedding ──
# 职责：类型 `SharedEmbedding`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SharedEmbedding(nn.Module):
    # ── SharedEmbedding.__init__ ──
    # 职责：引用原生词表分片权重，保存该 rank 的 token 起止 ID；当前不接受新增 LoRA 词表。
    # 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def __init__(self, original, tp, vocab_size):
        super().__init__()
        self.weight = original.weight
        indices = original.shard_indices
        self.start, self.end = indices.org_vocab_start_index, indices.org_vocab_end_index
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        self.tp, self.vocab_size = tp, vocab_size
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：added vocabulary/LoRA embeddings are not supported。
        if getattr(original, "num_added_embeddings", 0):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("added vocabulary/LoRA embeddings are not supported")

    # ── SharedEmbedding.forward ──
    # 职责：全局 token ID 映射到本 rank 的词表分片；非本地 token 贡献置零，再求和得到 embedding。
    def forward(self, tokens):
        outside = (tokens < self.start) | (tokens >= self.end)
        # 数据：当前 TP rank 对应的配置或局部 token 索引，需结合右侧表达式判断。
        local = (tokens - self.start).masked_fill(outside, 0)
        # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
        out = F.embedding(local, self.weight).masked_fill(outside[..., None], 0)
        return self.tp.reduce(out)


# ── SharedHead ──
# 职责：类型 `SharedHead`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SharedHead(ParallelLinear):
    # ── SharedHead.__init__ ──
    # 职责：建立共享词表并行 LM head，并记录真实词表宽度以裁掉 padding。
    # 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def __init__(self, weight, tp, vocab_size):
        super().__init__(weight, tp, "vocab")
        self.vocab_size = vocab_size

    # ── SharedHead.forward ──
    # 职责：调用词表并行线性层，拼接后裁掉 padding，输出完整有效词表分数。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def forward(self, value):
        return super().forward(value)[..., :self.vocab_size]

    @property
    # ── SharedHead.vocab_start ──
    # 职责：返回当前 rank 分片对应的全局词表起点，用于 teacher 切片与全局 token 对齐。
    def vocab_start(self):
        return min(self.tp.rank * self.weight.shape[0], self.vocab_size)

    # ── SharedHead.local_logits ──
    # 职责：只计算本 rank 的有效词表分片，去除 padding；供词表并行 loss 避免完整 logits gather。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def local_logits(self, value):
        # 数据：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
        width = max(0, min(self.weight.shape[0], self.vocab_size - self.vocab_start))
        return self._project(value)[..., :width]


# ── _weight ──
# 职责：确认当前原生线性权重是可直接使用的二维 BF16/FP32；量化 packed 权重不能走此映射。
def _weight(module):
    weight = module.weight
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unquantized, unpacked BF16/FP32 native linear weights required。
    if weight.dtype not in {torch.bfloat16, torch.float32} or weight.ndim != 2:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("unquantized, unpacked BF16/FP32 native linear weights required")
    return weight


# ── from_native_model ──
# 职责：用 meta 模型骨架承接 vLLM 原生 TP 参数；共享 backbone 后构造 Qwen38DualView，检查没有遗漏 meta 权重。
# 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
# 输入：已经驻留设备的原生 target、完整文本 config、TP 组、K 和 mask token。
# 输出：本 rank 的 Qwen38DualView；共享矩阵是原生分片，只有 draft mixer 复制。
# 特别注意：Transformers 类名使用 Qwen3_5，这是所适配架构的实现类名，不是把 27B 模型替换成更小模型。
def from_native_model(target, config, *, tp, block_size, mask_token_id, training_gdn_backend="torch"):
    """Build a TP-sharded HF-shaped view without a second backbone allocation."""
    # 依赖：从 transformers 导入 Qwen3_5TextConfig, Qwen3_5ForCausalLM；供本模块调用。
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    language = getattr(target, "language_model", target)
    # 数据：原生模型主体，后续共享参数/缓存来自这里。
    native = language.model
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：dense Qwen3.5/3.8 text architecture with gated attention and no layer scaling required。
    if (config.model_type != "qwen3_5_text" or getattr(config, "layer_scale", False)
            or not getattr(config, "attn_output_gate", True) or getattr(config, "attention_bias", False)):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("dense Qwen3.5/3.8 text architecture with gated attention and no layer scaling required")
    config = Qwen3_5TextConfig(**config.to_dict())
    # 数据：当前 TP rank 对应的配置或局部 token 索引，需结合右侧表达式判断。
    local = copy.deepcopy(config)
    for field in ("num_attention_heads", "num_key_value_heads", "intermediate_size",
                  "linear_num_key_heads", "linear_num_value_heads"):
        # 数据：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        value = getattr(local, field)
        if value % tp.size:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError(f"{field}={value} is not divisible by TP={tp.size}")
        setattr(local, field, value // tp.size)
    # meta 设备只构造形状骨架，不分配真实大矩阵；后面必须映射全部原生参数。
    with torch.device("meta"):
        shaped = Qwen3_5ForCausalLM(local)
    text = shaped.model
    text.embed_tokens = SharedEmbedding(native.embed_tokens, tp, config.vocab_size)
    text.norm = SharedRMSNorm(native.norm.weight, config.rms_norm_eps, native_residual=True)
    shaped.lm_head = SharedHead(_weight(language.lm_head), tp, config.vocab_size)
    # 只新建很小的 RoPE 缓冲；backbone 参数均引用当前 TP rank 的原生权重。
    # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
    text.rotary_emb = type(text.rotary_emb)(local).to(native.norm.weight.device)
    for index, (layer, source) in enumerate(zip(text.layers, native.layers, strict=True)):
        layer.input_layernorm = SharedRMSNorm(source.input_layernorm.weight, config.rms_norm_eps,
                                             native_residual=index > 0)
        layer.post_attention_layernorm = SharedRMSNorm(source.post_attention_layernorm.weight, config.rms_norm_eps,
                                                      native_residual=True)
        layer.mlp = SharedMLP(_weight(source.mlp.gate_up_proj), _weight(source.mlp.down_proj), tp)
        if config.layer_types[index] == "full_attention":
            mixer, ar = layer.self_attn, source.self_attn
            # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
            q = local.num_attention_heads * local.head_dim * 2
            kv = local.num_key_value_heads * local.head_dim
            # 数据：从原生合并 QKV 矩阵切出的 Q/K/V 视图，切片本身不等于独立 clone。
            parts = _weight(ar.qkv_proj).split((q, kv, kv))
            for name, weight in zip(("q_proj", "k_proj", "v_proj"), parts, strict=True):
                setattr(mixer, name, ParallelLinear(weight, tp, "column"))
            mixer.o_proj = ParallelLinear(_weight(ar.o_proj), tp, "row")
            mixer.q_norm = SharedRMSNorm(ar.q_norm.weight, config.rms_norm_eps)
            mixer.k_norm = SharedRMSNorm(ar.k_norm.weight, config.rms_norm_eps)
            # 调用：保留 reference forward，并把已有 NPU 算子包装绑定为该 mixer 的 forward。
            configure_attention(mixer, getattr(ar, "rotary_emb", None))
        else:
            mixer, ar = layer.linear_attn, source.linear_attn
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：Qwen3.5 non-interleaved, TP-sharded GDN weights required。
            if getattr(ar, "gqa_interleaved_layout", False) or getattr(ar, "disable_tp_for_ba_proj", False):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
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
            # 调用：给 mixer 装配归一化、卷积和 delta rule 实现；这里选择后端，并不执行一次完整推理。
            configure_gdn(mixer, training_backend=training_gdn_backend)
    model = Qwen38DualView(shaped, block_size=block_size, mask_token_id=mask_token_id,
                           hybrid_policy="orthrus_qwen35_reference")
    # 映射不完整就报错，不能用随机参数掩盖缺失的权重。
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native weight mapping left unmapped meta parameters。
    if any(p.is_meta for p in model.parameters()):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("native weight mapping left unmapped meta parameters")
    model.source = {"model": str(getattr(target, "prefix", "native-vllm")),
                    "tp_size": tp.size, "tp_rank": tp.rank, "history_backend": "unbound"}
    return model


# ── NativeSuffixLearner ──
# 职责：类型 `NativeSuffixLearner`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class NativeSuffixLearner(SuffixLearner):
    """TP-aware online replay against an immutable native vLLM target."""
    # ── NativeSuffixLearner.__init__ ──
    # 职责：在通用 learner 上增加 TP 组、可选词表分片 loss、复制 norm 参数清单，以及 CPU 分支的融合优化器。
    # 输入提示：tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def __init__(self, model, config, tp, *, vocab_parallel_loss=False):
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        self.tp = tp
        self.vocab_parallel_loss = vocab_parallel_loss and tp.size > 1
        super().__init__(model, config)
        # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
        if self.parameters[0].device.type == "cpu":
            self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate,
                                               weight_decay=0, fused=True)
        # 数据：TP 之间复制而非分片的 norm 参数名；梯度要归并且范数只计一次。
        self.replicated = {name for name in self.master if ".q_norm." in name or ".k_norm." in name
                           or ".norm." in name}

    # ── NativeSuffixLearner.replay_logits ──
    # 职责：用保存的 boundary/history 重算学生后缀，选取有效预测行；native 可覆盖为词表分片实现。
    def replay_logits(self, item):
        if not self.vocab_parallel_loss:
            return super().replay_logits(item)
        return self.model.forward_suffix(item.boundary, cache=item.cache, logit_range=(0, item.valid),
            draft_weights=self.master, checkpoint_layers=self.config.checkpoint_layers,
            logit_head=self.model.head.local_logits).logits[0]

    # ── NativeSuffixLearner.replay_divergence ──
    # 职责：按配置比较 student 与 teacher 概率；不是与 token ID 做均方差。
    # 输入提示：student：需要梯度的 draft logits（可为本 rank 的词表分片）；teacher：冻结主模型的监督 logits；它不承担参数更新。
    def replay_divergence(self, student, teacher):
        if not self.vocab_parallel_loss:
            return super().replay_divergence(student, teacher)
        # 依赖：从 .qwen38_loss 导入 vocab_divergence；供本模块调用。
        from .qwen38_loss import vocab_divergence
        return vocab_divergence(student, teacher, start=self.model.head.vocab_start,
                                 tp=self.tp, kind=self.config.loss)

    @staticmethod
    # ── NativeSuffixLearner.parameter_stamp ──
    # 职责：记录参数版本/身份供冻结保护；NativeSuffixLearner 覆盖以兼容 inference tensor。
    def parameter_stamp(parameter):
        version = None if torch.is_inference(parameter) else parameter._version
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return (parameter.data_ptr(), tuple(parameter.shape), parameter.device, parameter.dtype, version)

    # ── NativeSuffixLearner.fingerprint ──
    # 职责：建立冻结部分身份；native 版本使用进程内参数身份，验证脚本的完整内容哈希是另一种更强的审计。
    def fingerprint(self):
        # No 50GB host copy at worker startup. This identity is process-local:
        # target hot reload and cross-process optimizer restore are prohibited.
        identity = [(name, str(self.parameter_stamp(p))) for name, p in self.model.named_parameters()
                    if name not in self.execution]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    # ── NativeSuffixLearner.backward ──
    # 职责：先调用通用后缀反向，再把 TP 中复制的 norm 参数梯度打包 all-reduce，保持各 rank 更新一致。
    def backward(self):
        # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。 此处从 loss 建立的图反向累积梯度。
        loss = super().backward()
        names = [name for name in sorted(self.replicated) if self.master[name].grad is not None]
        gradients = [self.master[name].grad for name in names]
        if gradients and self.tp.size > 1:
            # 所有复制的 norm 参数只做一次小通信；CPU offload 仍仅搬运这一小包。
            device = next(iter(self.execution.values())).device
            # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入；沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
            packed = torch.cat([gradient.reshape(-1) for gradient in gradients]).to(device)
            # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
            packed = self.tp.reduce(packed).to(gradients[0].device)
            for name, gradient, reduced in zip(names, gradients, packed.split([g.numel() for g in gradients]),
                                                strict=True):
                # 数据：用于训练的 FP32 参数副本；优化器更新它，再 publish 到执行参数。
                self.master[name].grad = reduced.reshape_as(gradient)
        # 输出：学生与教师概率分布的散度；必须有限才允许反向更新。
        return loss

    # ── NativeSuffixLearner.clip_gradients ──
    # 职责：计算全局梯度范数并按阈值缩放；TP 版本排除复制参数的重复计数后跨 rank 归并。
    def clip_gradients(self):
        total = torch.zeros((), device=self.parameters[0].device, dtype=torch.float32)
        # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
        gradients = [value.grad.float() for name, value in self.master.items()
                     if value.grad is not None and (name not in self.replicated or self.tp.rank == 0)]
        # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
        if gradients and total.device.type == "npu":
            # 现成多张量归约，避免逐个大矩阵创建平方副本。
            total = torch.stack(torch._foreach_norm(gradients, 2)).square().sum()
        else:
            for gradient in gradients:
                # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
                total += (torch.linalg.vector_norm(gradient, dim=-1).square().sum()
                          if gradient.device.type == "cpu" else gradient.square().sum())
        collective_device = next(iter(self.execution.values())).device
        # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        norm = self.tp.reduce(total.to(collective_device)).sqrt().to(total.device)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonfinite TP draft gradient norm。
        if not torch.isfinite(norm):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise FloatingPointError("nonfinite TP draft gradient norm")
        # 数据：缩放系数：注意力中用于点积缩放，梯度裁剪中用于限制范数。
        scale = (self.config.clip_norm / (norm + 1e-6)).clamp(max=1.)
        for value in self.parameters:
            if value.grad is not None:
                value.grad.mul_(scale)
        return norm
