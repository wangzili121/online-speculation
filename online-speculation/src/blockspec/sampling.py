# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：概率与校正数学工具：温度/top-k/top-p、抽样、greedy 和随机校正参考实现。vLLM 主路径复用的是原生 rejection sampler。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Categorical sampling and the exact positive-residual rejection correction."""

# 依赖：从 dataclasses 导入 dataclass；供本模块调用。
from dataclasses import dataclass
# 依赖：导入 math。
import math

# 依赖：导入 torch。
import torch
# 依赖：从 torch 导入 Tensor；供本模块调用。
from torch import Tensor


@dataclass(frozen=True)
# ── SamplingConfig ──
# 职责：类型 `SamplingConfig`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SamplingConfig:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    temperature: float = 0.0
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    top_k: int = 0
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    top_p: float = 1.0

    # ── SamplingConfig.__post_init__ ──
    # 职责：数据类构造后校验配置，拒绝无法支持的参数组合。
    def __post_init__(self):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：temperature must be finite and nonnegative。
        if not math.isfinite(self.temperature) or self.temperature < 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("temperature must be finite and nonnegative")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid top-k or top-p。
        if self.top_k < 0 or not 0 < self.top_p <= 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid top-k or top-p")


# ── probabilities ──
# 职责：验证采样配置/输入 logits 后计算 temperature/top-k/top-p 对应的完整概率分布。
# 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
def probabilities(logits: Tensor, config: SamplingConfig) -> Tensor:
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：logits must be finite。
    if not torch.isfinite(logits).all():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("logits must be finite")
    return _probabilities_unchecked(logits, config)


# ── _probabilities_unchecked ──
# 职责：对 logits 做温度缩放、候选截断和 softmax；内部路径假定输入已由上层检查。
# 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
def _probabilities_unchecked(logits: Tensor, config: SamplingConfig) -> Tensor:
    """Pure tensor transform; graph callers check finiteness at the boundary."""
    # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
    work = logits if logits.dtype == torch.float64 else logits.float()
    if config.temperature == 0:
        return torch.zeros_like(work).scatter_(-1, work.argmax(-1, keepdim=True), 1)
    work = work / config.temperature
    if config.top_k:
        # 数据：卷积后续递推必须保留的历史输入数，通常为 kernel_width−1。
        keep = min(config.top_k, work.shape[-1])
        indices = work.topk(keep, dim=-1).indices
        selected = torch.zeros_like(work, dtype=torch.bool).scatter_(-1, indices, True)
        work = work.masked_fill(~selected, -torch.inf)
    if config.top_p < 1:
        ordered, indices = work.sort(dim=-1, descending=True)
        masses = ordered.softmax(-1)
        # Retain the crossing token; probability before this token is < top_p.
        remove = (masses.cumsum(-1) - masses) >= config.top_p
        remove[..., 0] = False
        removed = torch.zeros_like(remove).scatter_(-1, indices, remove)
        work = work.masked_fill(removed, -torch.inf)
    return work.softmax(-1)


# ── greedy_tokens ──
# 职责：沿词表维取 argmax，得到每个预测位置的 token ID。
# 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
def greedy_tokens(logits):
    """Exact greedy specialization: no vocabulary-sized one-hot allocation."""
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonempty finite logits required。
    if logits.ndim < 1 or logits.shape[-1] < 1 or not torch.isfinite(logits).all():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("nonempty finite logits required")
    return logits.argmax(-1)


# ── sample_logits ──
# 职责：把一行 logits 按配置转成概率并抽取一个 token；不是独立执行模型。
# 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行；generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
def sample_logits(logits, config=SamplingConfig(), generator=None):
    if config.temperature == 0:
        return greedy_tokens(logits)
    return draw(probabilities(logits, config), generator)


# ── validate_distribution ──
# 职责：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
def validate_distribution(p):
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：probabilities must be finite and nonnegative。
    if p.ndim < 1 or p.shape[-1] < 1 or not torch.isfinite(p).all() or (p < 0).any():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("probabilities must be finite and nonnegative")
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：probabilities must sum to one。
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-6, rtol=1e-6):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("probabilities must sum to one")


# ── draw ──
# 职责：按每行概率分布抽样，用传入 Generator 控制随机序列。
# 输入提示：generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
def draw(p: Tensor, generator=None) -> Tensor:
    # 调用：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
    validate_distribution(p)
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).reshape(p.shape[:-1])


# ── residual ──
# 职责：构造首次拒绝后的残差分布：对 p−q 取非负部分再归一化。
# 输入提示：q：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
def residual(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    # 调用：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
    validate_distribution(p)
    # 调用：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
    validate_distribution(q)
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：target and proposal shapes differ。
    if p.shape != q.shape:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("target and proposal shapes differ")
    positive = (p - q).clamp_min(0)
    mass = positive.sum(-1, keepdim=True)
    # When mass=0 rejection is impossible; the returned fallback is never used
    # by a correct rejection decision. Keeping it normalized helps diagnostics.
    corrected = torch.where(mass > 0, positive / mass.clamp_min(torch.finfo(p.dtype).tiny), p)
    # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
    return corrected, mass.squeeze(-1)


@dataclass
# ── Verification ──
# 职责：类型 `Verification`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Verification:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    tokens: list[int]
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    accepted: int
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    rejected_at: int | None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    supervised: int


# ── verify_greedy ──
# 职责：参考 greedy 校正：找候选与 AR argmax 的第一个不一致处，保留匹配前缀并补目标 token。
def verify_greedy(proposed, target_ids):
    """The p/q correction for two one-hot laws reduces to equality + replacement."""
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：integer proposals[n] and greedy targets[n+1] required。
    if (proposed.ndim != 1 or target_ids.shape != (proposed.numel() + 1,)
            or proposed.dtype not in (torch.int32, torch.int64)
            or target_ids.dtype not in (torch.int32, torch.int64)):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("integer proposals[n] and greedy targets[n+1] required")
    # 数据：待主模型验证的 token ID，一维长度最多 K−1。
    candidates, targets = proposed.tolist(), target_ids.tolist()
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：negative token id。
    if any(t < 0 for t in candidates + targets):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("negative token id")
    for i, token in enumerate(candidates):
        if token != targets[i]:
            return Verification(candidates[:i] + [targets[i]], i, i, i + 1)
    return Verification(candidates + [targets[-1]], len(candidates), None, len(candidates))


# ── verify_linear ──
# 职责：参考随机校正：按 min(1,p/q) 接受候选，首次拒绝后从残差分布采样；vLLM 服务路径仍用原生 sampler。
# 输入提示：q：本轮真实起草概率分布；随机校正需要它，greedy 可为 None；generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
def verify_linear(proposed: Tensor, q: Tensor, p: Tensor, *, generator=None,
                  acceptance_uniforms: Tensor | None = None) -> Verification:
    """Verify n proposals; p has n+1 rows, the last for the all-accept bonus.

    Returned tokens do not include the clean root generated in the draft pass.
    The first rejected row is a valid training row; later rows are not on-policy.
    """
    n = proposed.numel()
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：expected proposals[n], q[n,V], p[n+1,V]。
    if proposed.ndim != 1 or q.ndim != 2 or p.shape != (n + 1, q.shape[-1]) or q.shape[0] != n:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("expected proposals[n], q[n,V], p[n+1,V]")
    # 调用：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
    validate_distribution(p)
    # 调用：检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。
    validate_distribution(q)
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：proposal token outside vocabulary。
    if ((proposed < 0) | (proposed >= q.shape[-1])).any():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("proposal token outside vocabulary")
    if acceptance_uniforms is None:
        acceptance_uniforms = torch.rand(n, device=p.device, generator=generator)
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：acceptance draws must lie in [0,1)。
    if (acceptance_uniforms.shape != (n,) or not torch.isfinite(acceptance_uniforms).all()
            or ((acceptance_uniforms < 0) | (acceptance_uniforms >= 1)).any()):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("acceptance draws must lie in [0,1)")
    # 数据：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
    output = []
    for i in range(n):
        token = int(proposed[i])
        denominator = q[i, token]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：a token cannot have been sampled from a zero proposal probability。
        if denominator <= 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("a token cannot have been sampled from a zero proposal probability")
        if acceptance_uniforms[i] < torch.minimum(p[i, token] / denominator, denominator.new_tensor(1)):
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            output.append(token)
        else:
            correction, mass = residual(p[i], q[i])
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：rejection with zero residual mass indicates numerical inconsistency。
            if mass <= 0:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise RuntimeError("rejection with zero residual mass indicates numerical inconsistency")
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            output.append(int(draw(correction, generator)))
            return Verification(output, i, i, i + 1)
    # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
    output.append(int(draw(p[-1], generator)))
    return Verification(output, n, None, n)
