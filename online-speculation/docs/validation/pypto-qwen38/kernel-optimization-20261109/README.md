# K32 Attention 与 GDN 验证记录

固定 OneServing `7aa40ba`、PyPTO `9875278`、Simpler `c1f89ea`、PTO-ISA `418de18` 和 PTOAS 0.65，在 910B3 A2 上执行。完整版本与方法见上级的 `QWEN38_PYPTO_GDN_KERNEL_TASK.md` 和 `QWEN38_PYPTO_KERNEL_TASK.md`。

| 文件 | 范围 |
| --- | --- |
| `attention-release.json` | 52 项 Attention：分阶段、融合、分页 16/32 query tile，输入逐字节不变及无效行 NaN 检查 |
| `gdn-independent-final.json` | 36 项独立 Q/K/V 递推：宽度 1/7/16/17/31/32，零/非零初态，输出及最终矩阵对照 |
| `gdn-packed-core.json` | 48 项包含 packed QKV 的递推，原生与两种 Vector 入口对照 |
| `gdn-aux-release.json` | 18 项 gate、gated norm/O projection、K32 卷积检查，原生与新路径输出逐字节一致 |
| `gdn-release-layered.json`、`gdn-release-group4.json` | 同 worker、相同真实 27B/head 的六轮交错对照；K32、缓存隔离、verifier 和历史边界 |
| `gdn-release-hf.json` | 64 层、两种后端，共 128 项真实输入 HF 对照；固定 2% 相对 L2 门槛 |
| `gdn-release-service.json`、`gdn-release-native-attention.json`、`gdn-release-fused-attention.json` | scheduler 续写及 K32 审计，三个 Attention 后端均使用 Vector GDN |
| `gdn-release-http.json` | HTTP completion/chat/SSE、配置实际生效和正常退出 |
| `cpu-release.xml` | 383 项 CPU 回归 |

其余 JSON 保留分块、归一化合并、V reshape、前后处理及整层的消融对照。`attn-slots2.json`、`attn-skipzero.json`、`gdn-values128.json` 是编译失败的候选，不计入已通过的检查；相关配置没有接入服务。`gdn-output-only.json` 只检查候选输出，不要求候选返回最终矩阵；收益约 0.9%，未接入模型。

`experiment-sources*.zip` 保存各次运行的完整 Python 源码，目录名对应远端不可变快照；`device-profiles*.zip` 保存报告所引用的重复时间线。前后对照在同 worker 交错计时，profiling 在计时外执行。`device_task_window_us` 是最早任务开始到最晚任务结束的窗口，不将并行 block 相加；`phases.device_seconds` 是同步调用墙钟，含主机调度。

完整模型使用 `trained-v4` 测试 head，记录实际参数哈希；权重与逐层张量没有放入 Git。独立 kernel 的随机输入可按脚本种子重建，不依赖这个 head。完整服务的验收范围为 TP1、单活动请求、512 位置、greedy，未由这些记录推导生产吞吐或接受率收益。
