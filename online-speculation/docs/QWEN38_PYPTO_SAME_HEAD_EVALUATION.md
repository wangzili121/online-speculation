# 同一份训练 head 的真实推理对照

这项检查把固定批次训练得到的同一份 head 接回完整 Qwen3.8-27B，区分“能够拟合训练块”和“能够提高真实投机接受率”。更新前后共享相同主模型、冻结前缀、历史读取方式和推理程序，评估期间不训练。

## 实测结果

两张 A2 上的完整对照通过。此 head 从 AR 参数复制初始化，只更新最后一层，在两道题的 8 个固定块上训练 128 步，块间存在重叠。有效 K=8，物理缓冲为 K32。

| 场景 | 初始化 head：候选接受率 / 每轮接受数 | 训练后同一份 head |
| --- | --- | --- |
| 6 个训练过的回答 anchor，42 个候选 | `16.67% / 1.17` | `100% / 7.00` |
| 2 条训练题的前 32 个输出 token | `17.30% / 1.10` | `18.23% / 1.18` |
| 2 条 dev 的前 32 个输出 token | `18.08% / 1.14` | `17.88% / 1.10` |
| 8 条 test 的前 32 个输出 token | `16.69% / 1.05` | `18.48% / 1.17` |

六组执行参数与 master 转换结果逐值一致；16 次 anchor 检查中，冻结前缀输入、推理与采集路径 logits、teacher logits 的最大差值均为零。六个已训练回答块在真实 verifier 中各接受 7/7，验证输出与逐步 AR 相同。两份 head 的全部 24 个完整请求均对齐配对 AR；连同 12 个 AR 基线请求，36 个请求零错误，结束时队列与 KV 占用归零。

test 的单次诊断吞吐从 `4.61` 提高到 `4.87 tok/s`，主模型单独运行为 `6.78 tok/s`，训练后仍只有其 `0.72×`，未获得净加速。这是一次小规模诊断，不是多轮稳定性压测。

结果说明这份 head 确实学会了训练过的回答块，并能在真实推理中生效；新题只有有限提升，dev 未改善。少量固定块的拟合尚未变成足够的自由起草能力。完整训练题的统计也不等于精确训练 anchor 的统计：自由生成会走到不同的块起点，32-token 输出预算还未覆盖原 48-token 回答中的全部已训练位置。

两个 prompt anchor 虽然教师强制匹配率达到 100%，真实验证仍各只接受一个候选。原因是教师强制输入原问题的后续文本，而真实 greedy 会生成不同的后续 token，之后的条件分布已经改变。因此，prompt 与模型生成回答的训练位置应分别分析；是否限制到回答区域更有效，需要保持训练预算和问题集合不变再作对照。

原始结果、命令和参数哈希见[验证记录](validation/pypto-qwen38/same-head-20261111/README.md)。

## 检查方法

输入为已完成 128 步固定批次训练的 last-1 初始与最终 FP32 master。将其转换为执行参数并发布到推理卡，读回六组参数逐值核对。评估不执行 Adam 更新，也不将权重快照当作可续训检查点。

先沿采集时相同的分段前缀，重放 8 个训练 anchor。每个位置分别执行正常融合推理与训练采集路径，对照冻结前缀输入、logits、teacher 分布和历史状态，再由真实 verifier 判断候选的接受前缀，并检查输出与逐步 AR 一致。

其中 6 个 anchor 位于模型生成的回答内；另 2 个位于 prompt。prompt 的真实后续文本不一定等于主模型自己的 greedy 续写，因此两组分别统计，不能把教师强制匹配率直接当成实际接受率。

随后对 2 条训练题、2 条 dev 和 8 条 test，各生成最多 32 个 token；初始与最终 head 使用相同题目、K8 和缓存策略。记录真实候选接受率、每轮接受长度、逐 token AR 一致性与吞吐。训练 anchor 的接受情况用于检查训推衔接，新题用于观察泛化。

## 运行

在固定 OneServing/PyPTO 环境中执行，推理与参数准备使用两个独立 NPU worker：

```bash
PYTHONPATH=src python scripts/evaluate_qwen38_pypto_fixed_head.py \
  --source /path/to/OneServing --model /models/Qwen3.8-27B \
  --data /data/gsm8k --device 0 --learner-device 1 \
  --capture-report /results/fixed-last1.tensors/manifest.json \
  --tokens 32 --test-count 8 --output /results/same-head.json
```

输入目录需保留真实反馈快照、`initial.master.safetensors` 和 `final.master.safetensors`。本入口只恢复模型执行参数，不能用来恢复该快照未保存的优化器状态。
