# Orthrus-GLM5.3 / Ascend

分支：`feature/orthrus-glm53`。针对 `Eco-Tech/GLM-5.3-Flash-w8a8` 的 Orthrus 推理适配；训练与在线更新暂不推进。

- 模型：冻结并共享主模型 backbone；draft 适配 MLA 与 KDA，包含因果 KDA 和显式实验性双向 KDA 两条路径，不能将两者当作同一个算法配置。
- 实现：原生 vLLM-Ascend 负责主模型和验证；插件负责 draft、只读历史与候选分布，使用已有算子，不含自研 kernel。
- 已验：算子 demo、缓存只读和裁剪真实权重模型的部分功能；不等于完整 45 层模型已经验收。
- 后续验证：在八卡环境运行完整 45 层模型，覆盖因果/双向 KDA、head 重载与采样。
- head：没有训练 head 时只验证初始化接入，不承诺接受率或速度收益。

运行入口及镜像要求见 [GLM 推理说明](online-speculation/docs/ORTHRUS_GLM53_FLASH.md)。算子需求与实现见 [PyPTO 报告](online-speculation/docs/ORTHRUS_GLM53_PYPTO_REPORT.md)，KDA 策略见 [方向定义](online-speculation/docs/GLM53_KDA_DIRECTION.md)，资源与验收状态见 [验证队列](online-speculation/docs/VALIDATION_QUEUE.md)。

历史中保留公共代码和其他模型；后续 GLM 改动仅在此分支迭代。

现成算子的选择、局部比较与测试范围见 [推理算子复核](online-speculation/docs/GLM53_INFERENCE_OPS_AUDIT.md)。
