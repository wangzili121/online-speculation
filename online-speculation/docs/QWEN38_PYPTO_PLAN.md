# Orthrus-Qwen3.8 PyPTO 接入计划

目标是在原生 Qwen3.8 模型之外完成 Orthrus 起草、验证和 head 换入，并独立优化 Full Attention 与 GDN。训练和在线更新继续走 PyTorch 路径，PyPTO 读取导出的 head 并推理。未来原生优化版本的接口尚未确定，届时逐项评估布局和融合边界，当前工作不依赖其发布。

| 工作 | 已完成 | 后续接入 |
| --- | --- | --- |
| 分页双向 Full Attention | 独立 Torch 参考、融合和分阶段对照、直接分页至 L1、128/256/512 容量选择；52 项 A2 检查通过 | 随原生局部 heads 接口扩展 TP；更长上下文另行设计与验证 |
| GDN | 只读 K32 卷积、packed QKV 直接读取、Vector 归一化/递推、gate 与 gated norm；102 项独立 NPU 检查通过 | 继续根据整层时间线检查投影等准备阶段，不改变因果公式 |
| 完整 K32 draft | 共享 embedding、norm、MLP、LM head 和真实历史；Full 块内双向，GDN 沿用约定的因果递推；128 项真实输入逐层 HF 对照通过 | 完整 TP 需要原生局部 Attention/GDN/MLP 与归约接口 |
| 执行调度 | Full 层准备与计算合并，提交由 82 次降为 66 次；可选四层合并进一步降为 18 次，按组上传参数；输出与逐层执行逐字节相同 | 参数全部驻留后的整网性能需要单独验收 |
| head 换入和事务 | TP2/4 文件布局、同参数重分片及重启换入；cache 只读、六个历史边界、八组接受/拒绝/截断、scheduler 续写通过 | 多 rank 同步候选、接受长度、页表与 GDN 恢复 |
| 多卡准备 | 两卡真实 head 分片驻留、原生行并行投影及重复归约组件已验证 | 将组件接入完整多卡模型及 HTTP 服务，不能将组件验证当作 TP 服务通过 |

当前完整服务仍为 **TP1、一个活动请求、512 个总位置、greedy**。默认保留 `native` 对照；优化配置为 `--attention-backend paged --gdn-backend vector --draft-group-size 4 --head-transfer packed`。CPU 回归共 383 项通过。

独立核与整轮起草分别计时。最终 GDN 组合将首层设备窗口由 4.001 降至 2.271 ms；同 worker 四层编排的完整起草中位数由 1.401 降至 1.321 s，约减少 5.7%。实际服务收益仍受 head 上传和候选接受率影响。完整 TP 与 head 驻留需要对接完整模型执行接口。

[kernel 输入与调优边界](QWEN38_PYPTO_KERNEL_TASK.md)、[执行与验证记录](QWEN38_PYPTO_PAGED_VALIDATION.md)、[固定环境和启动方式](QWEN38_PYPTO_PORT.md)分别说明开发接口、原始证据和部署配置。

GDN 已补齐独立参考、驻留计时与设备时间线，完成 Vector 递推、归一化合并、gate 合并和直接 Z 读取的实验；接口、分段结果及真实 27B 接回范围见 [GDN kernel](QWEN38_PYPTO_GDN_KERNEL_TASK.md)。Attention 也完成 KV 分块、跨 head 复用和流水槽的对照，保留测到收益的配置；不会将“更多融合”直接等同于“更快”。
