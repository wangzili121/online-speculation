# PyPTO last-1 在线更新验证

验证对象为真实 Qwen3.8-27B 和外部 TP4 测试 head；推理重排为 TP1，另用一张 A2 执行学习。固定环境见[部署说明](../../../QWEN38_PYPTO_PORT.md)。当前范围为 K32、512 个总位置、greedy、一个活动请求，仅更新第 63 层 Full Attention 的六组参数。

| 检查 | 证据 | 已验证内容 |
| --- | --- | --- |
| CPU 回归 | [cpu.json](cpu.json)、[输出](cpu-output.txt) | 445 项通过，覆盖参考梯度、反馈边界、head 格式、版本切换和异常恢复。 |
| 显式反向 | [中间结果对照](online-suffix-tiled.json)、[共享视图](online-suffix-views.json) | 三组历史/块长/监督行配置；全部 77 个中间结果与独立 Torch autograd 对照，最大相对 L2 约 `6.46e-5`。 |
| 真实尺寸 | [末层图](real-shape-suffix.json)、[优化器](real-shape-optimizer.json) | 真实 27B 参数尺寸运行，77 个输出有限、六组梯度非零；Adam 和裁剪另与独立 Torch/FP64 范数对照。真实尺寸末层图未进行完整 CPU 数值对照。 |
| 优化器与参数视图 | [带偏移的加载](online-optimizer-offset-views.json) | 从真实 Runtime Tensor 视图加载 FP32 master，包含非零字节偏移；四步 Adam、非法梯度不发布、恢复后下一步精确一致。 |
| 从 head 初始化 | [整模型闭环](online-real27b-fresh-r2.json) | 四轮 K32、两块反馈共同更新，新 head 在下一轮生效；AR 缓存不变，greedy 对齐；scheduler 更新及恢复后的 HTTP 更新通过。版本 `0 → 4`。 |
| 从 checkpoint 冷启动 | [整模型闭环](online-real27b-resume-r1.json) | 新进程恢复版本 1，再完成同样的起草、验证、更新和 HTTP 流程，最终版本 5。 |
| 导出与重新部署 | [逐层核对](online-export-roundtrip.json)、[HTTP](online-exported-http.json) | 前 63 层逐字节相同，末层六组执行参数与 checkpoint 一致，源 head 未变；导出 head 独立运行 completion/chat/SSE，三请求零错误并正常关闭。 |

整模型实验将更新间隔设为两轮，以便在短请求中覆盖多次更新；部署默认间隔为 16 轮。`update_seconds` 包含 checkpoint 保存和验证读回，不能用作纯反向计算计时。测试文本用于功能联调，不据此评价训练质量、接受率或加速收益。

复现整模型检查，在已配置的固定 PyPTO 环境中运行：

```bash
cd online-speculation
ASCEND_RT_VISIBLE_DEVICES=0,1 python scripts/validate_qwen38_pypto_online.py \
  --source /path/to/OneServing --model /models/Qwen3.8-27B \
  --head-dir /models/orthrus-pypto-head --learner-head-dir /models/orthrus-raw-head \
  --device 0 --learner-device 1 --replay-blocks 2 --output /data/check.json
```

使用新的输出路径；加 `--resume-online /data/online-heads/step-00000001` 验证冷启动，`--head-dir` 仍指向该 checkpoint 绑定的初始化 head。实际命令和 source 快照见 [初始化运行](online-real27b-fresh-r2.supervisor.json)、[恢复运行](online-real27b-resume-r1.supervisor.json)及 [component-metadata.json](component-metadata.json)。
