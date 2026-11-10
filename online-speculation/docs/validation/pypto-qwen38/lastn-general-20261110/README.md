# 跨四层组的 last-N 验证

环境与 [last-1～4 记录](../lastn-20261110/README.md) 相同。后段按真实层序列构造：每四层为三层因果 GDN 和一层块内双向 Full Attention。中间 Full Attention 不重复执行 LM head 或损失；每层历史独立复制且只读，参数整组发布。

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| last-5 后段 | 39 组参数，部分块和非零历史、完整块和零历史；梯度、两次 Adam 更新与 checkpoint 精确恢复通过 | `suffix-last5-adam-r28.json` |
| last-8 后段 | 66 组参数，跨两个四层组；相同检查通过 | `suffix-last8-adam-r28.json` |
| 大 N 的特殊参数缓冲 | 11264 个元素的 FP32/BF16 缓冲各两次完整字节清零通过 | `flat-zero-special11264-r28.json` |
| CPU 回归 | 486 项；包含 last-5/8 的 FP32/BF16 独立 autograd、64 层图结构、跨组截断点及显存拒绝检查 | `source-lastn-r30-cpu-output.json`、`source-lastn-r30-cpu-output.txt` |

TV 在教师概率等于 draft 概率处不可微。两个设备用例分别有 1/2 个比较位置因 BF16 前向微差改变符号；逐节点使用相同输入核对，再以相同 TV 上游梯度驱动独立 autograd，全部参数通过，最大相对 L2 分别约 1.18%/1.93%。损失公式和原有梯度阈值保持不变。

优化器分两步核对：FP32 master 与 Torch Adam 使用原数值阈值比较；BF16/特殊 FP32 执行参数则必须与设备实际 master 的转换结果精确一致。这样不会把两个已在 FP32 容差内、恰好位于 BF16 舍入边界两侧的 master 当成格式转换错误。

真实 27B 的 last-5 更新、冷恢复、导出和独立服务检查正在进行。last-8 此处是独立后段验证，不能视为真实模型两卡部署已通过。运行范围与资源要求见 [last-N 使用说明](../../../QWEN38_PYPTO_ONLINE_LASTN.md)。
