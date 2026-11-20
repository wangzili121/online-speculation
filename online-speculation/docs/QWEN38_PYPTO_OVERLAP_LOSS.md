# Qwen3.8 在线更新的分布重合损失

TP2 在线更新增加显式选项 `--objective log_overlap`，对应 TTS [作者代码](https://github.com/avinkumarUT/test-time-speculation/blob/main/python/sglang/srt/speculative/tts_loss.py)中的分布重合负对数。默认仍为 TV，KL 也保留。该选项使用已验证的真实前缀教师分布，不修改 target、候选校正或历史 cache。

## 计算

对第 $i$ 个已监督位置，$p_i$ 是固定的教师分布，$q_i=\operatorname{softmax}(z_i)$ 是 draft 分布，$\mathcal V$ 为完整词表。定义

$$
\alpha_i=\sum_{v\in\mathcal V}\min\{p_i(v),q_i(v)\},\qquad
\mathcal L=-\frac1Z\sum_i w_i\log\max(\alpha_i,10^{-8}),
\qquad Z=\sum_i w_i.
$$

$w_i$ 为有效位置的权重，未监督位置为零；多个反馈块使用共同的 $Z$。令 $m_i(v)=\mathbf1[q_i(v)<p_i(v)]+\frac12\mathbf1[q_i(v)=p_i(v)]$。当 $\alpha_i\ge10^{-8}$ 时，显式反向为

$$
\frac{\partial\mathcal L}{\partial z_i(v)}
=\frac{w_i q_i(v)}{Z\alpha_i}
\left(\sum_u q_i(u)m_i(u)-m_i(v)\right).
$$

低于截断值时梯度为零；相等处使用半梯度，与 `torch.minimum` 对齐。该计算复用完整词表损失 kernel 的 softmax 与分块归约，在 PyPTO 中直接产生 loss 和 logits 梯度，再进入已有手写反向及 Adam。

$\alpha_i=1-\mathrm{TV}(p_i,q_i)$，在标准随机拒绝采样中等于单位置接受概率。greedy 下没有这一等式保证，需要与 TV/KL 做完整请求流对照；梯度验证通过不代表吞吐已经改善。

## 验证与使用

468 项 CPU 回归通过，包括 last-1/last-4 的独立 autograd、动态位置权重、相等概率和截断边界。六个独立损失程序及真实形状的 last-1/last-4 训练入口已通过编译。核内条件表达式及分支变量作用域的两次编译失败、修正后的完整记录分别保存在 `overlap198/`、`overlap199/`、`overlap200/`。

真实 A2 上完成 31 组完整词表检查，词表大小为 248,320。新损失的梯度相对 L2 最大为 0.000344；未监督行梯度为零，零重合概率的 loss 有限、梯度为零。原 TV 的加权与非加权输出均与保留版本逐位一致。该项为独立计算验证，整模型服务及吞吐单独评测。

在配套环境中运行：

```bash
python scripts/validate_qwen38_pypto_position_loss.py \
  --device 0 --vocabulary 248320 --log-overlap \
  --reference-source /path/to/original/training_programs.py \
  --output /data/disk/results/overlap-loss.json
```

TP2 benchmark 选择 `--objective log_overlap --teacher-targets distribution`；原有更新范围、频率、发布周期及资源设置保持显式配置。当前探索比较同两张 A2、同起始 head 的 AR、固定起草、全程持续更新、最终发布 head 原题重跑和独立开发集，不自动使用预留确认题。
