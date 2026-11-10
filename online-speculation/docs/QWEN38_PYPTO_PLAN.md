# Orthrus-Qwen3.8 PyPTO 接入计划

目标是在原生 Qwen3.8 模型之外完成 Orthrus 起草、验证和 head 换入，并独立优化 Full Attention 与 GDN。完整训练继续走 PyTorch 路径；PyPTO 在线更新覆盖末尾 1–4 层，使用显式反向而非自动微分。未来原生优化版本的接口尚未确定，届时逐项评估布局和融合边界，当前工作不依赖其发布。

| 工作 | 已完成 | 后续接入 |
| --- | --- | --- |
| 分页双向 Full Attention | 独立 Torch 参考、融合和分阶段对照、直接分页至 L1、128/256/512 容量选择；52 项 A2 检查通过 | 随原生局部 heads 接口扩展 TP；更长上下文另行设计与验证 |
| GDN | 只读 K32 卷积、packed QKV、Vector 归一化/递推、gate、gated norm，以及 K32 投影和共享 MLP；独立检查与真实逐层 HF 对照通过 | 按本地 heads、历史 stride 和归约接口扩展 TP，不改变因果公式 |
| 完整 K32 draft | 共享 embedding、norm、MLP、LM head 和真实历史；Full 块内双向，GDN 沿用约定的因果递推；128 项真实输入逐层 HF 对照通过 | 完整 TP 需要原生局部 Attention/GDN/MLP 与归约接口 |
| 执行调度 | Full 层准备与计算合并，提交由 82 次降为 66 次；可选四层合并进一步降为 18 次，按组上传参数；输出与逐层执行逐字节相同 | 参数全部驻留后的整网性能需要单独验收 |
| 设备端收尾 | PyPTO 准备 MASK/RoPE/Attention mask，设备端检查并选择 token，设备内保存和恢复 GDN；保留主机对照路径 | 完整 TP 时按本地状态布局、候选同步与全词表选择扩展 |
| 权重加载 | PyPTO 整理 head 分片、卷积转置、补零、norm 和类型转换；主模型按层 DMA 到最终 bank；支持保存并重载转换后的 head | 文件读取、JSON 和文件哈希继续使用主机接口；完整 TP 服务按各 rank 布局加载 |
| last-1–4 在线更新 | 显式 Full Attention/GDN 反向、双块 TV 更新、FP32 Adam/裁剪、多层原子发布及 checkpoint；真实 27B last-2/4 的冷/热恢复与 HTTP 通过，冷恢复下一步参数和 Adam 状态逐字节一致；last-4 导出 head 后独立推理通过 | last-3 已做 CPU 梯度对照，整模型服务待单独实测；完整 TP 根据模型实际布局扩展 |
| head 换入和事务 | TP2/4 文件布局、同参数重分片及重启换入；cache 只读、六个历史边界、八组接受/拒绝/截断、scheduler 续写通过 | 多 rank 同步候选、接受长度、页表与 GDN 恢复 |
| 多卡准备 | 两卡真实 head 分片驻留、原生行并行投影及重复归约组件已验证 | 将组件接入完整多卡模型及 HTTP 服务，不能将组件验证当作 TP 服务通过 |

当前完整服务仍为 **TP1、一个活动请求、512 个总位置、greedy**。Attention/GDN 默认保留 `native` 对照；优化配置为 `--attention-backend paged --gdn-backend vector --draft-group-size 4 --head-transfer packed`。收尾默认 `--serving-backend device`，权重加载默认 `--weight-backend pypto`，均保留 `host` 对照。CPU 回归共 472 项通过；多层更新配置及实测范围见 [last-N 更新](QWEN38_PYPTO_ONLINE_LASTN.md)。

独立核、完整层和整轮起草分别计时。同 worker、相同 head 的四层编排中，K32 共享计算将同步计算调用由约 0.199 降至 0.136 s；完整起草由 1.501 降至 1.434 s。另在同一 NPU 上，仅将 head 主机内存从远端 NUMA 改为本地，完整起草由 1.740 降至 1.094 s，logits 逐字节相同。两组实验不合并计算加速比；实际服务收益仍取决于上传和候选接受率。

[kernel 输入与调优边界](QWEN38_PYPTO_KERNEL_TASK.md)、[执行与验证记录](QWEN38_PYPTO_PAGED_VALIDATION.md)、[固定环境和启动方式](QWEN38_PYPTO_PORT.md)分别说明开发接口、原始证据和部署配置。

GDN 已补齐独立参考、驻留计时与设备时间线，完成 Vector 递推、归一化合并、gate 合并和直接 Z 读取的实验；接口、分段结果及真实 27B 接回范围见 [GDN kernel](QWEN38_PYPTO_GDN_KERNEL_TASK.md)。Attention 也完成 KV 分块、跨 head 复用和流水槽的对照，保留测到收益的配置；不会将“更多融合”直接等同于“更快”。
