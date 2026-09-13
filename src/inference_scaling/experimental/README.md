# 非默认研究实现

本目录保存通过实现测试的可选算法。调用方使用 `inference_scaling.experimental` 下的完整模块路径显式导入。

当前包括：

- 动态候选、逐轮 i-SIR、初始估计与最终估计分离的 IS、SMC 多树搜索和流式 IS；
- 有界提前停止、随机化 QMC rollout 设计和逐轮 SIR 公共算子；
- Qwen2.5-0.5B 草稿模型的精确推测解码实验实现。

算法原理、使用约束和非默认方案的简要记录集中在[算法文档](../../../docs/methods/ALGORITHMS.md#alg-nondefault-notes)。
默认组件由 `experiments/shared/components.py` 定义；运行结果保存在 Git 忽略的 `results/` 目录。
