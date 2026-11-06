# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：采样策略包装：从 draft logits 取候选与真实 q；其独立 verify 方法不等于 vLLM 在线路径的原生 sampler。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Candidate sampling and exact correction over the same probability policy."""

# 依赖：从 ..sampling 导入 SamplingConfig, draw, greedy_tokens, probabilities, sample_logits, verify_greedy, verify_linear；供本模块调用。
from ..sampling import (SamplingConfig, draw, greedy_tokens, probabilities, sample_logits,
                        verify_greedy, verify_linear)


# ── ProposalSampler ──
# 职责：类型 `ProposalSampler`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class ProposalSampler:
    # ── ProposalSampler.__init__ ──
    # 职责：保存采样配置与可选 executor；无 executor 时直接使用概率数学工具。
    def __init__(self, config=SamplingConfig(), *, executor=None):
        self.config, self.executor = config, executor

    # ── ProposalSampler.sample_ar ──
    # 职责：独立参考路径的 AR token 选择；可交给外部 executor 执行。
    # 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行；generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
    def sample_ar(self, logits, generator):
        if self.executor is not None:
            return self.executor.sample_ar(logits, generator)
        return int(sample_logits(logits, self.config, generator))

    # ── ProposalSampler.propose ──
    # 职责：greedy 取最大分数 token 且 q=None；随机模式计算真实温度/截断后 q 并从中抽样。
    # 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行；generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
    def propose(self, logits, generator):
        if self.executor is not None:
            return self.executor.draft(logits, generator)
        if self.config.temperature == 0:
            # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
            return greedy_tokens(logits), None, None
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q = probabilities(logits, self.config)
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return draw(q, generator), q, None

    # ── ProposalSampler.verify ──
    # 职责：在 runtime 中包装原生采样；在独立 ProposalSampler 中则是参考校正。以所在类/模块为准，不是两套同时调用。
    # 输入提示：generator：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
    def verify(self, proposal, target_logits, generator):
        if self.executor is not None:
            return self.executor.verify(proposal.candidates, proposal.proposal, target_logits, generator)
        if self.config.temperature == 0:
            # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
            return verify_greedy(proposal.candidates, greedy_tokens(target_logits)), None
        target = probabilities(target_logits, self.config)
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return verify_linear(proposal.candidates, proposal.proposal, target, generator=generator), target
