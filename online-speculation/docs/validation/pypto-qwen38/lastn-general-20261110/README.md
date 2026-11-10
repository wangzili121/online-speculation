# 跨四层组的 last-N 验证

环境与 [last-1～4 记录](../lastn-20261110/README.md) 相同。后段按真实层序列构造：每四层为三层因果 GDN 和一层块内双向 Full Attention。中间 Full Attention 不重复执行 LM head 或损失；每层历史独立复制且只读，参数整组发布。

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| last-5 后段 | 39 组参数，部分块和非零历史、完整块和零历史；梯度、两次 Adam 更新与 checkpoint 精确恢复通过 | `suffix-last5-adam-r28.json` |
| last-8 后段 | 66 组参数，跨两个四层组；相同检查通过 | `suffix-last8-adam-r28.json` |
| 大 N 的特殊参数缓冲 | 11264 个元素的 FP32/BF16 缓冲各两次完整字节清零通过 | `flat-zero-special11264-r28.json` |
| 真实 27B last-5 | 两块 K32 反馈联合更新，版本 0→4；下一轮换权、目标 cache 不变、greedy、scheduler、热恢复和 HTTP 更新通过 | `online-last5-real-r30.json` |
| 冷启动恢复 | 新进程从版本 1 恢复并更新到版本 5，greedy、scheduler、热恢复和 HTTP 更新通过 | `online-last5-resume-r30.json` |
| 恢复连续性 | 下一步完整参数和两组 Adam 状态逐字节一致，读取并核验约 6.7 GB 完整 payload | `last5-continuity-r30.json` |
| head 导出 | 39 组执行参数与 checkpoint 精确一致，未更新区域和源 head 保持原字节，SHA256 通过 | `last5-export-check-r31.json` |
| 独立服务 | 导出 head 无学习器启动，completion、chat、SSE 各一条请求成功；无错误，cache 已回收，服务正常关闭 | `last5-export-http-result.json` |
| CPU 回归 | 488 项；包含 last-5/8 的 FP32/BF16 独立 autograd、64 层图结构、跨组截断点及显存拒绝检查 | `source-lastn-r31-cpu-output.json`、`source-lastn-r31-cpu-output.txt` |

TV 在教师概率等于 draft 概率处不可微。两个设备用例分别有 1/2 个比较位置因 BF16 前向微差改变符号；逐节点使用相同输入核对，再以相同 TV 上游梯度驱动独立 autograd，全部参数通过，最大相对 L2 分别约 1.18%/1.93%。损失公式和原有梯度阈值保持不变。

优化器分两步核对：FP32 master 与 Torch Adam 以 `rtol=2e-5`、`atol=1e-7` 对照；BF16/特殊 FP32 执行参数则必须与设备实际 master 的转换结果精确一致。这样不会把两个已在 FP32 容差内、恰好位于 BF16 舍入边界两侧的 master 当成格式转换错误。

真实 last-5 的整条更新和交付链路已通过，推理卡在请求结束后剩余约 439 MiB。当前布局下 last-8 比 last-5 还需额外约 1.32 GiB，本次仅完成小尺寸 NPU 后段验证，真实 27B last-8 待具备显存条件后验收。运行范围与资源要求见 [last-N 使用说明](../../../QWEN38_PYPTO_ONLINE_LASTN.md)。
