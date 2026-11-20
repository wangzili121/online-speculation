# Qwen3.8 PyPTO 低秩在线更新

该实验保留 Orthrus 的 K32 起草、历史共享、验证及 cache 管理，只改变在线更新的参数化方式。普通全量 draft 参数更新仍为默认；低秩更新使用显式的 `lora_merged` 选项。

## 计算与执行

对所选后段每个 draft attention 的投影矩阵使用 $W=W_0+BA$，当前 rank 为 64、缩放为 1。$W_0$ 固定，$A$ 按输入维度初始化，$B=0$；因此初始推理 head 与固定起草完全相同。norm、卷积、GDN 的 gate 等非矩阵参数保留原来的完整更新。last-1 的可训练参数从 104858112 减至 2621952，减少优化器状态；计算耗时仍需实测。

前向、输入梯度和服务推理读取合并后的完整 $W$，没有在每轮起草中额外执行 LoRA 分支。当前反向复用已有 $dW$，计算 $dA=B^T dW$、$dB=dW A^T$。该路径仍计算 dense $dW$，不能仅由参数减少推断反向加速。投影梯度与合并遵守既有 BF16 运算边界。

每次 Adam 更新先渲染候选适配器，复用已完成反向的 $dW$ 工作区计算 $BA$，与只读 $W_0$ 相加，写入非活跃完整权重视图。通过有限性检查后才切换优化器状态；即使本次没有到服务发布周期，也要更新 learner 的完整权重。到达发布周期时沿用现有 TP2 打包、DMA 和安全边界，两个 rank 一起切换。推理 head 导出为普通合并权重，不要求调用方支持 LoRA。

显存预检额外计入一份 $W_0$、两个完整执行视图和统计缓冲；小适配器执行缓冲不能借给完整 head 打包。当前仅支持 fresh Adam 和单一起始 head，不接受 dense optimizer checkpoint 或中途更换 $W_0$。

## 验证状态

421 项 CPU 检查通过，包括 last-1/last-4 的独立 autograd、零增量初始输出、权重合并、发布间隔和非有限候选的原子拒绝。64 个独立程序及完整 last-1/last-4 服务图已通过编译。真实 A2 上 14 组投影/VJP/权重合并检查通过，梯度相对 L2 最大约 $9.22\times10^{-5}$，合并权重相对 L2 最大约 $7.54\times10^{-6}$；七个零增量用例的初始权重逐位相同，原 dense 梯度、八种 BF16 element 和四组 FP32 统计回归也逐位一致，私有 NaN/Inf 候选均被检测。真实 27B、同两张 A2 的普通更新回归、LoRA last-1 和 last-4 短检查均已通过，各完成六次更新并逐次发布：greedy 与 AR 相同，发布权重精确导出；普通更新六步轨迹及最终权重与原实现完全一致。last-1/last-4 的回放 TV 最大分别为 $4.08\times10^{-7}$ / $1.26\times10^{-6}$，相对 L2 为 0.002963 / 0.005511。两种 LoRA 范围的两个 rank 权重均发生变化。该短检查不报告吞吐收益；每四步发布的完整探索流另行验证。

编译阶段曾因同类型 FP32 cast 失败；改为在构造程序时选择精度分支后通过。失败日志、源码哈希、CPU 记录和后续设备结果保留在 `validation/pypto-qwen38/online-datasets-20261119/lora166/`。


## 入口

沿用 `scripts/benchmark_qwen38_oneserving_tp2_online.py` 的 TP2 参数和数据配置，增加 `--parameterization lora_merged --adapter-rank 64 --adapter-seed 743`。选择 `--last-layers 1` 或 `4`，并使用 `--feedback-mode delayed`。未指定 `parameterization` 时仍执行原来的 `dense` 更新。硬标签教师继续要求 greedy 生成及 KL；其他教师分布配置保留。
