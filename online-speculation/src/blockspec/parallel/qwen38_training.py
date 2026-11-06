# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：离线监督构造：对干净文本执行冻结 AR 教师前向，构造 anchor+MASK，保存待训练后缀入口。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Clean-text random-anchor distillation for the hybrid Qwen draft."""

# 依赖：导入 torch。
import torch

# 依赖：从 ..feedback 导入 Feedback；供本模块调用。
from ..feedback import Feedback


@torch.no_grad()
# ── offline_feedback ──
# 职责：教师强制 AR 前向生成完整 K−1 行监督；构造 MASK 输入与冻结前缀边界，返回 Feedback。
# 输入提示：anchor：锚点：起草时指已知 token ID，离线训练函数的 anchor 参数则是文本中的位置索引，注意所在函数。
# 输入：干净 token 序列 [1,T]，anchor 是其中一个位置，要求容纳完整 K-token 块。
# 输出：Feedback，教师有完整 K−1 行，区别于在线实际到达的 valid 行。
# 装饰器 no_grad 表示教师与边界构造不记录旧图；之后 learner 会开启梯度重放后缀。
def offline_feedback(model, tokens, anchor):
    """Teacher-forced AR supplies all K-1 targets, unlike reached-only online feedback."""
    # 数据：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
    width = model.config.block_size
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：one clean token sequence and an anchor with a complete block required。
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long
            or type(anchor) is not int or not 0 <= anchor <= tokens.shape[1] - width
            or bool(((tokens < 0) | (tokens >= model.config.vocab_size)).any())):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("one clean token sequence and an anchor with a complete block required")
    # 数据：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态。
    history = model(tokens[:, :anchor], logits_to_keep=1).cache if anchor else None
    clean = tokens[:, anchor:anchor + width]
    # 数据：冻结主模型的监督 logits；它不承担参数更新。 此处断开旧 autograd 计算图，不自动复制底层存储。
    teacher = model(clean, cache=history).logits[0, :-1].detach()
    noisy = torch.full_like(clean, model.config.mask_token_id)
    noisy[:, 0] = clean[:, 0]
    # 数据：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
    boundary = model.draft_boundary(noisy, history)
    return Feedback(noisy, history, teacher, width - 1, boundary)
