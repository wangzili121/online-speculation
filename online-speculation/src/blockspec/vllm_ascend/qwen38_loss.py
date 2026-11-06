# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：可选词表并行损失：每卡只保留自己的 student 词表分片，通过 collective 得到全局归一化与散度。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Exact vocabulary-parallel distillation using existing tensor collectives."""

# 依赖：导入 torch。
import torch


# ── _VocabDivergence ──
# 职责：类型 `_VocabDivergence`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _VocabDivergence(torch.autograd.Function):
    @staticmethod
    # ── _VocabDivergence.forward ──
    # 职责：从局部 student logits 和全局 teacher 分布计算 KL/TV；跨 rank 合并归一化量，并保存局部解析梯度。
    # 输入提示：student：需要梯度的 draft logits（可为本 rank 的词表分片）；teacher：冻结主模型的监督 logits；它不承担参数更新；tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
    def forward(ctx, student, teacher, start, tp, kind):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：aligned local student and complete teacher vocabulary required。
        if (kind not in {"forward_kl", "tv"} or student.shape[:-1] != teacher.shape[:-1]
                or start < 0 or start + student.shape[-1] > teacher.shape[-1]):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("aligned local student and complete teacher vocabulary required")
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        # 数据：当前 TP rank 对应的配置或局部 token 索引，需结合右侧表达式判断。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        local = (student.to(dtype), teacher[..., start:start + student.shape[-1]].detach().to(dtype))
        # 只汇总每个位置的两项局部 LSE，再稳定合并为完整词表的归一化量。
        normalizer = tp.gather(torch.stack([value.logsumexp(-1) for value in local], dim=-1))
        normalizer = normalizer.unflatten(-1, (tp.size, 2)).logsumexp(-2)
        log_q, log_p = [value - normalizer[..., i, None] for i, value in enumerate(local)]
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q, p = log_q.exp(), log_p.exp()
        if kind == "forward_kl":
            # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。
            loss = tp.reduce((p * (log_p - log_q)).sum(-1))
            derivative = q - p
        else:
            sign = (q - p).sign()
            combined = tp.reduce(torch.stack([.5 * (q - p).abs().sum(-1), (q * sign).sum(-1)]))
            # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。
            loss = combined[0]
            derivative = .5 * q * (sign - combined[1, ..., None])
        # 每个 rank 持有相同 loss；只返回本地 logits 梯度，LM head 再归并 hidden 梯度。
        ctx.save_for_backward(derivative)
        ctx.dtype = student.dtype
        # 输出：学生与教师概率分布的散度；必须有限才允许反向更新。
        return loss

    @staticmethod
    # ── _VocabDivergence.backward ──
    # 职责：将保存的局部 student 梯度乘上游梯度；教师、分片起点、TP 对象和散度类型不更新。
    def backward(ctx, gradient):
        (derivative,) = ctx.saved_tensors
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return (derivative * gradient[..., None]).to(ctx.dtype), None, None, None, None


# ── vocab_divergence ──
# 职责：计算词表分片 KL/TV；输入 student 是局部分片，teacher 与全局词表/分片起点共同决定监督。
# 输入提示：student：需要梯度的 draft logits（可为本 rank 的词表分片）；teacher：冻结主模型的监督 logits；它不承担参数更新；tp：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
def vocab_divergence(student, teacher, *, start, tp, kind):
    return _VocabDivergence.apply(student, teacher, start, tp, kind)
