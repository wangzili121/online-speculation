# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：训练反馈数据结构：把 MASK 输入、历史、教师 logits、有效位置与冻结边界打包，供 learner 重放。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Owned reached-prefix feedback and frozen-boundary replay protocol."""

# 依赖：从 __future__ 导入 annotations；供本模块调用。
from __future__ import annotations

# 依赖：从 dataclasses 导入 dataclass；供本模块调用。
from dataclasses import dataclass
# 依赖：从 typing 导入 Protocol；供本模块调用。
from typing import Protocol

# 依赖：从 torch 导入 Tensor；供本模块调用。
from torch import Tensor

# 依赖：从 .state 导入 Cache；供本模块调用。
from .state import Cache


# ── ReplayBoundary ──
# 职责：类型 `ReplayBoundary`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class ReplayBoundary(Protocol):
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    start_layer: int

    # ── ReplayBoundary.detached ──
    # 职责：断开旧计算图，并按需要保留 cache 后缀和复制反馈张量；之后训练会新建后缀计算图。
    def detached(self) -> ReplayBoundary: ...


@dataclass
# ── Feedback ──
# 职责：类型 `Feedback`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Feedback:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    inputs: Tensor
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    cache: Cache | None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    teacher_logits: Tensor
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    valid: int
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    boundary: ReplayBoundary | None = None
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    fully_covered: bool = False

    # ── Feedback.detached ──
    # 职责：克隆输入/教师 logits 并 detach cache 与 boundary，保留从 cache_start 起的层历史，供后缀重放。
    def detached(self, *, cache_start=0):
        # 计算：断开旧 autograd 计算图，不自动复制底层存储。
        cache = None if self.cache is None else tuple((k.detach(), v.detach()) for k, v in self.cache[cache_start:])
        return Feedback(self.inputs.detach().clone(), cache, self.teacher_logits.detach().clone(), self.valid,
                        None if self.boundary is None else self.boundary.detached(), self.fully_covered)
