# Qwen3.8 原生 vLLM 验证记录

## 2026-10-31 同宽度数值复查

在 167 的 0、4 卡恢复真实 27B、TP2、K32、8K、eager 单请求验证。固定复制初始化的 head，不进行在线更新。
同模型宽度对照再次复现：0/1/7/15/16 个候选保持原输出，31 个候选在同一 JSON 位置分叉。
逐层记录表明，普通 decode 与 K32 的首行 embedding、第一处 input norm 输出相同；第 0 层 GDN 后的 post norm 首次出现可见差异，最大绝对误差为 0.000732421875。
这比单独比较最终 logits 更接近差异来源，但还不能据此把全部最终误差归因于某一个 kernel。

修正 AR 对照的输入宽度：原脚本只重放 anchor 和前 30 个候选，现包含全部 31 个候选，与原生 verifier 同为 32 行，再比较前 31 行 logits。
同宽度最大 TV=0.11111507，首行 TV=0.03195545；预设的 0.05 门槛仍未通过，没有用放宽阈值处理。
诊断还覆盖第 15、16、26 行，不再只观察首行。旧的 0.09108947 记录保留为不同输入宽度下的历史结果。

在相同的原生输入上分别控制 gamma 和归一化中间结果的舍入，128 处 norm 中有 121 处与“两处均先舍入到 BF16”的表达式逐元素一致；第一处非残差 norm 则与 HF 表达式一致。
这说明原生 fused 分支与参考表达式确有数值路径区别，不代表直接把所有 norm 换成某个 fused 调用就能消除全网误差。

固定修复前 head 提出的候选块，四种归一化实验的原生 verifier logits 完全相同。只对齐残差 norm 的舍入时，AR 最大 TV 从 0.111115 降到 0.047429；
直接对所有 norm 使用 `npu_rms_norm(x, 1+w)` 则为 0.052653。这是同一候选块的受控实验，不是通用通过结论。
把候选全部改为另一组 token，原生前一位置的 320 处逐层记录仍逐元素相同；该用例未发现候选未来信息泄漏。

随后将共享残差 norm 接到现成的 `torch_npu.npu_rms_norm(x, ones)`，再乘 BF16 的 `1+w`；第一层输入及 Q/K norm 保留原路径。
1/16/32 行、隐藏维度 5120 的单算子前向与目标舍入表达式逐元素一致，输入梯度相对 L2 误差为 0.00266--0.00311，在预设容差内。
只增加非持久的单位 gamma buffer，不改变共享主模型参数、head tensor 格式、Attention、cache 或 rejection sampler；未开发新 kernel。

**扩大验证后，AR 数值门槛仍未通过。** 新执行路径会改变 draft 提出的候选，不能与前一段固定候选实验直接比较最大 TV。
复制初始化 head 的 JSON K32 对照最大 TV=0.298912；加载 version254 的扩展测试中，四个前缀的最大 TV 分别为 0.176190、0.030569、0.032436、0.290049。
不能把单个受控块的 0.047429 登记为整个执行器通过，也不能把所有误差都归因于 norm；后续仍需隔离 GDN 与 packed/split projection 的数值路径。

加载 version254 后，两个 rank 各完成 195 次最后四层更新、194 个完整验证块和 101 次缓存只读检查，计数及最终 loss 一致。
16 个 greedy 用例共 400 token，仍只有同一 JSON 用例与 target-only 不同；其他 15 个用例逐 token 一致。
第一次扩展运行的 JSON 在采样校准前落盘，因此其中计数为 163；195 次更新和最终失败门槛见原始日志，不把中间快照当作最终结果。
验收工具现会在最终断言前保存采样统计及权重哈希，另有回归确认失败文件仍包含这些字段。
修正保存逻辑后再次加载 version254，四个请求加 16 种子采样校准，两 rank 各完成 53 次更新、53 个完整验证块和 60 次缓存只读检查。
主模型两分片合计 53,793,351,168 字节的参数哈希不变，head 内容哈希改变；32 个投机输出位置的条件概率频数检查通过。
这四个请求中，实际到达行的最大 AR TV 为 0.028735；但全部候选行的最大 TV 仍为 0.176190，因此程序保持非零退出和 `completed=false`，没有用到达行结果替代原门槛。
相关回归为 225 passed、7 个 CUDA 专用测试 skipped；Ruff 与 `git diff --check` 通过。

复现入口（与部署环境和版本锁定文件配合使用）：

```bash
ORTHRUS_LAYERWISE_VARIANTS=1 ORTHRUS_LAYERWISE_OUTPUT=/work/evidence/layerwise.json \
python scripts/probe_qwen38_layerwise.py \
  --model /models/Qwen3.8-27B --mask-token-id 1 --tp 2 --block-size 32 \
  --max-model-len 8192 --kv-cache-memory-bytes 12884901888
```

[宽度复测](validation/qwen38/acceptance-1031-resume-width.json) · [修复前逐层记录](validation/qwen38/acceptance-1031-layerwise-aligned.json.gz) · [固定候选受控实验](validation/qwen38/acceptance-1031-layerwise-variants-v3.json.gz) · [现成 RMSNorm 前后向](validation/qwen38/acceptance-1031-norm-op.log.gz)

[新路径逐层记录](validation/qwen38/acceptance-1031-norm-layerwise.json.gz) · [195 次更新及失败门槛的原始日志](validation/qwen38/acceptance-1031-norm-last4.log.gz) · [最终回归](validation/qwen38/acceptance-1031-norm-release-cpu.xml) · [版本锁定](validation/qwen38/acceptance-1031-norm-runtime-lock.json)

[53 次更新、冻结哈希、采样及未通过门槛的完整结果](validation/qwen38/acceptance-1031-norm-final-last4.json)

本轮结束后已停止专用容器；最终使用的 167 的 4、5 卡实查没有 NPU 进程，只剩基础显存占用，模型及 version254 保留。
验证脚本因数值门槛返回 1；容器的 137 则来自随后主动停止 `sleep infinity`，`OOMKilled=false`，不将二者混为模型 OOM。
[资源释放记录](validation/qwen38/acceptance-1031-norm-resource-release.json)

## 2026-10-31 扩展验收

**扩展验收尚未全部通过，不能用下方较早的通过记录代替新用例的结果。**
真实 27B、TP2、8K 上下文的 K32 检查中，16 个 greedy 用例有 15 个逐 token 一致，JSON 提示在第二个输出 token 处分叉。
同一模型、同一提示、同一 draft head，只改变实际提交的候选数：0/1/7/15/16 个候选均与无投机一致，31 个候选出现分叉。
无投机该位置的前两名目标 logits 为 16.625 和 16.5；32 行验证时出现三个 16.625 的并列值，原生 argmax 选择发生变化。
进一步固定第一层原生投影的输入和权重，直接调用 `torch.nn.functional.linear`，仅改变 batch 行数，也复现了最大 0.03125 的输出变化。
实测各块长度进入第一层投影的输入按字节相同；原生调用与独立 linear 检查均出现该变化。
这确认了目标后端的 batch 形状相关数值变化是一个明确来源，不涉及 draft mask 或历史缓存读取；尚未逐层归因全部最终 logit 误差。

同一 K32 运行的 64 种子、128 个投机输出位置的采样统计通过；参考概率直接取自原生采样前 logits。
固定 head 的相同 seed 请求复现检查通过；两个 TP rank 的计数一致，每 rank 248 次缓存只读抽查通过。
两 rank 全量主模型参数运行前后 SHA256 相同，覆盖合计 53,793,351,168 字节。
映射 AR 视图与原生验证的额外数值对照未达到预设 TV<=0.05：31 行最大 TV=0.09108947，首行 TV=0.02189109。
这项诊断不能登记为通过，也不能仅凭阈值超出便认定参数映射有误；需与目标后端的块长度数值差异一同分析。

相关回归为 224 passed、16 个 CUDA 专用测试 skipped。
原始记录：[K32 严格对照](validation/qwen38/acceptance-1031-k32-fixed-strict.json)、[同模型块宽度对照](validation/qwen38/acceptance-1031-json-width.json)、[原生投影与独立 linear](validation/qwen38/acceptance-1031-json-projection.json)、[最新回归结果](validation/qwen38/acceptance-1031-final-v4-cpu.xml)。

另有不加载模型的单卡数值复现：`python scripts/probe_npu_linear_batch.py --device 0`。
本机随机矩阵的 batch 差异最大为 2.86e-6，见 [单算子记录](validation/qwen38/acceptance-1031-linear-synthetic.json)；真实模型投影的 0.03125 来自另一组权重和输入，不把二者当作同一误差指标。
新增回归覆盖固定反馈失败仍返回诊断结果；验证入口先保存 checkpoint，再执行最终门槛，保留诊断失败时的状态。

镜像内原生 sampler 的 31 个拒绝位置、全接受 bonus 和 32768 次分布抽样通过；另一个混合 batch 的零候选 RNG 检查失败，未登记为通过。
该失败由直接调用原生 sampler 复现，不经过本插件，且不属于当前 `max_num_seqs=1` 的声明范围；本轮未修改它。
`logprob_token_ids` 与投机组合、受限词表的 processed-logprobs 也未纳入已支持范围，镜像内实测出现过 bonus shape 错误或返回项不完整。
last-1、stride=16 的在线边界检查已完成：14 个 greedy 请求合计 215 token 与基线一致，另含 3 个 top-k/top-p 请求。
两个 rank 各验证 111 轮、更新 6 次，计数及最终 loss 一致；head 哈希改变，主模型全量权重哈希不变，每 rank 52 次缓存只读抽查通过。
五类请求边界各运行三轮，第二、三轮活跃内存的最大增长为 4,136,448 字节，低于 64 MiB 门槛；这是有界复测，不是长时间无泄漏证明。
两个分片及 manifest 已导出到 version6。见 [last-1 原始记录](validation/qwen38/acceptance-1031-last1-online.json)。
last-4 的第一次扩展运行在固定反馈验收失败：沿用已有 AdamW 状态重复更新 8 次，TV loss 从 0.51410162 升至 0.51602209，两 rank 分别有 25、26 个执行参数变化，缓存仍按字节不变。
这是未通过项，不将“参数能更新”当成“loss 已下降”；完整记录见 [失败摘要](validation/qwen38/acceptance-1031-replay8-failed.json)及 [原始日志](validation/qwen38/acceptance-1031-replay8-failed.log.gz)。
32 步复测完成：两 rank 各有 222 次请求触发更新和 32 次固定反馈更新，TV loss 从 0.51410162 降至 0.50858486；两 rank loss 序列一致，缓存不变。
8 步处的 0.51602209 在复测曲线中精确复现，后续 loss 仍有波动，不将单样本改善称为业务收敛。
该运行的主模型全量参数哈希不变，每 rank 83 次缓存只读抽查通过；24 个 greedy 请求中，除同一 JSON 数值边界外其余 23 个对齐。
见 [last-4 引擎执行记录](validation/qwen38/acceptance-1031-last4-online.json)及 [分项判定](validation/qwen38/acceptance-1031-functional-matrix.json)。引擎执行完成不代表外部数值或内存门槛全部通过。

此运行的重复请求内存门槛未过：最大增长 184,753,152 字节。定位到采样适配每轮定义局部类，其闭包让完整 q 表等待循环 GC。
改用模块级代理后，关闭循环 GC 的弱引用回归通过；临时换回旧实现可稳定复现失败，没有靠强制 GC 或放宽 64 MiB 门槛处理。
新进程加载 version254，两个 TP 分片内容哈希与导出时完全相同，再执行 85 次在线更新、五轮交替请求及采样校验；主模型哈希不变，每 rank 78 次缓存只读抽查通过。
最后两轮同类请求边界的活跃内存增长为 0；暖机后最大波动为 564,224 字节。
见 [修复后真机记录](validation/qwen38/acceptance-1031-memory-fixed-online.json)和 [内存判定](validation/qwen38/acceptance-1031-memory-fixed-check.json)。

受控全接受及拒绝分支均通过：全接受 4 轮、124 个候选，四次从接受 32 个输入的位置继续；连续拒绝 127 轮、接受 0 个候选，均从接受 1 个输入的位置继续。
两组各 128 个输出 token 均与无投机基线一致，见 [全接受](validation/qwen38/acceptance-1031-branch-oracle.json)、[连续拒绝](validation/qwen38/acceptance-1031-branch-reject.json)。这些候选是验收夹具，不是训练 head 的接受率证据。

原生 OpenAI API 的 15 项检查通过，包含 chat、普通 logprobs、SSE、两个客户端排队、1804-token 跨页输入及五轮取消/非法请求后恢复。
加载 version254 后使用 last-1、stride=16，两 rank 各实际更新 4 次，计数、loss、梯度范数一致。
最终服务的 JSON 数值探针仍与 target-only 不同，保留为未通过项；停服退出码为 0，关闭期出现原生异步 output-handler 错误和 resource-tracker 警告，不声称停服零报错。
本次专用服务及容器已停止，实查 167 的 0、4 卡没有运行进程；61、111 没有本次仍在运行的验收容器。临时 pre-replay version222 已清理，最终 version254 和 last-1 version6 保留。
见 [API 检查](validation/qwen38/acceptance-1031-api-167.json)、[服务在线更新](validation/qwen38/acceptance-1031-api-online-updates.json)、[最终 JSON 探针](validation/qwen38/acceptance-1031-api-json-diagnostic.json)及 [资源释放检查](validation/qwen38/acceptance-1031-resource-release.json)。
最终代码、框架 commit、镜像和 67 个源文件的 SHA256 见 [版本锁定](validation/qwen38/acceptance-1031-runtime-lock-final.json)；实际部署源码与本地逐文件一致，三个关键框架源文件哈希未改变。

以下为 2026-10-30 及之前已完成的验证范围。

日期：2026-10-30。对象为真实 Qwen3.8-27B BF16 checkpoint，不是缩小模型。

## 当前判断

**真实 27B 在 BF16、TP2、K32、eager、同步单请求调度下，原生 vLLM 起草、验证、拒绝恢复、在线更新和 head 保存重载已通过功能验证。**
不是将独立 Torch 执行器包装成 HTTP：主模型、正式缓存和 rejection sampler 均由原生 vLLM-Ascend 执行。
原生 OpenAI API 的 8 类接口检查也全部通过。本记录不代表生产性能验收。

## 扩展边界与修复

2026-10-30 的扩展检查使用 61 的 4、5、6、7 卡运行两组 TP2，不是 TP4。
新增 128-token 输出、4092-token 输入加 4-token 输出、停止后重新请求，以及固定反馈的重复更新检查。

修复后的完整 last-4 在线回归通过：7 个 greedy 用例共 220 个输出 token 与 target-only 逐 token 相同，另含 temperature=1 和 top-k/top-p 采样请求。
每 rank 起草及读取历史 120 次，验证 110 轮，其中 109 轮为完整 K32；实际请求触发 110 次更新，固定反馈再更新 8 次，共 118 次，两个 rank 的计数及 loss 一致。
每 rank 的 28 次抽查均确认主模型 cache 按字节不变；固定反馈更新前后的 cache 检查也通过。
同一反馈的 TV loss 从 0.5937114 降至 0.5093935，选中的 draft 参数发生变化；中间 loss 并不单调，这只是固定样本更新检查，不是业务收敛或吞吐收益证据。
更新后的两个 head 分片与 manifest 已保存为 `heads-final-61/version118`。
原始结果见 [完整边界基线](validation/qwen38/acceptance-soak-target-61.json)和 [last-4 在线回归](validation/qwen38/acceptance-final-last4-61.json)。

重启服务加载 version47，使用最后 1 层、默认 `stride=16`，两个 rank 各实际更新 3 次，计数、loss 和梯度范数一致。
[API 8 类检查](validation/qwen38/acceptance-api-final-61.json)全部通过：health/model、greedy、top-k/top-p、SSE、两个客户端排队、取消后恢复、1804-token 跨页输入及后续请求隔离。
两个客户端仍是在 `max_num_seqs=1` 下排队，不代表动态批处理已实现。

4K 边界检查发现并修复了 runner 适配问题：发生拒绝后，下一轮可能不再有候选，只有一个普通 decode token。
原适配把它与 prefill 一起标为 -1；原生 metadata 调用又因本轮没有候选而不传上轮接受位置，导致 GDN 从错误的状态槽继续。
修复后，prefill 保持 -1，无候选 decode 使用 0，并继续向 GDN builder 传递接受位置；没有修改 Attention、卷积或 GDN kernel。

固定 head、关闭在线更新时，修复前的边界输出为 `[3,3,3,3]`，修复后为 `[3,3,3,271]`，与 target-only 和原生 n-gram 对照一致。
最后一步目标 logits 的首选 token 也恢复为 271；因此这次差异不能归因于在线训练或普通舍入误差。
对应原始结果为 [修复前](validation/qwen38/acceptance-context-fixed-61.json)、[原生 n-gram 对照](validation/qwen38/acceptance-context-ngram-61.json)和 [修复后](validation/qwen38/acceptance-context-fixed-repaired-v2-61.json)。

一次服务启动的自动 KV 预算为 -5.14 GiB，尚未进入请求执行。该估值的根因未独立确认；扩展部署改用 vLLM 原有的显式 KV 预算，每 rank 12 GiB，仍执行 profiling forward，不修改预算算法。
此配置已完成固定 head 重载和服务启动。它仅适用于本次 TP2/4K/单请求配置，不能直接外推到其他规模。

修复后的 CPU 回归为 182 passed、7 个 CUDA 专用测试 skipped，Ruff 与 `git diff --check` 通过。
新增回归覆盖无候选 decode 的 GDN 标记及 metadata 传递、head 缺失/多余 tensor、形状、精度、非有限值和 TP 不匹配的拒绝，并确认加载失败前不改动任何 draft 权重。
部署边界仍为 eager、同步、`max_num_seqs=1`；未将有界测试描述为长时间稳定性或生产性能验收。

## 主模型缓存复用验证

原生入口现使用 `shared_target_snapshot`，直接读取 target 的分页 KV 和 GDN 接受前缀状态，不再调用 AR forward 重建历史。
现有 draft 执行器接收连续布局，因此分页 KV 仍需 gather，GDN 使用独立工作状态；这不是零拷贝分页 Attention。

本轮重新运行无投机基线，并在 61 的双卡上验证新路径，配置为 TP=2、K=32、eager、同步 batch-one。
固定 head 使用 max_model_len=2048，最后四层在线更新使用 1024；两者均与同后端基线对照。

| 检查 | 结果 |
| --- | --- |
| 固定 head、原生缓存 | 每 rank 55 次缓存读取与起草、50 轮完整 K32 验证、46 个接受候选；greedy 三个用例共 72 个输出 token 与基线完全一致 |
| 最后 4 层在线更新 | 每 rank 52 次读取与起草、47 轮完整 K32 验证、47 次更新、50 个接受候选；两个 rank 计数与 loss 一致，greedy 再次对齐 |
| 主模型缓存只读 | 在线测试每 rank 抽查 15 次，覆盖各请求的首次有效 prefill 起草、投机起草及在线更新；参与本轮的 KV 物理页、卷积行和候选递推状态前后按字节相同 |
| 原生 GDN 接受位置 | 单卡调用真实卷积及 recurrent 算子，连续三轮从上轮接受 1/17/32 个输入的位置继续；本轮全部 32 个接受位置均正确，合计 96 次选择，FP32 状态最大绝对误差 1.2e-7，读取前后原缓存不变 |
| 在线 head 导出与重载 | 两个 TP 分片和 manifest 保存到 `heads-shared-cache-61/version47`，重启服务成功加载，不复制 backbone |
| API 与最后 1 层在线更新 | 加载 version47 后每 rank 完成 52 次更新；8 项接口检查全部通过，包括 SSE、两个客户端排队、断开流后恢复、1804-token 跨页 prefill 及后续请求隔离 |

本轮最后四层测试的最终 TV loss 为 0.4607100487。这里只证明更新链路与冻结约束，不据此宣称收敛或加速。
API 测试使用 max_model_len=2048，最后一层测试的最终日志为 TV loss=0.5171158、梯度范数=0.9754444，两个 rank 一致。
CPU 回归为 154 passed、17 个 CUDA 专用测试 skipped；新增覆盖乱序页、跨页、接受边界、拒绝与全接受、缓存复用后的快照生命周期，以及只读审计能够检出故意写入。
proposer 单元测试明确禁止 AR forward，避免静默回退到历史重放。

一次在线启动在 profiling 阶段得到负 KV 预算，未进入请求执行；清理本次 worker 后重试通过，profiling 波动的根因未单独确认。

本轮记录为 `shared-cache-target-61.json`、`shared-cache-fixed-61.json`、`shared-cache-last4-61.json`、`shared-cache-api-61.json` 和 `shared-cache-gdn-61.log`。
[无投机基线](validation/qwen38/shared-cache-target-61.json)、[固定 head](validation/qwen38/shared-cache-fixed-61.json)、[共享缓存在线更新](validation/qwen38/shared-cache-last4-61.json)、[API](validation/qwen38/shared-cache-api-61.json) 的原始 JSON 随代码提供。
日志已保存在实验机和本地；结束后已停止本次 61、167 的专用容器，实查 61 的 2、3 卡无 NPU 进程，没有停止其他用户任务。安装目录的三个原生源码文件哈希与文末锁定值一致。
以下记录保留为缓存改造前的功能及底层算子证据，不将它们冒充新缓存路径的测试。

## 缓存改造前的验证

实验机 61 的两张 Ascend 910B3，TP=2、K=32、eager、同步调度、max_num_seqs=1。
离线 LLM 接口测试使用 max_model_len=1024，HTTP 服务使用 2048。
基线和 Orthrus 均使用相同 CANN GDN prefill 后端，KV 预算比例为 0.65。

| 检查 | 结果 |
| --- | --- |
| 无投机基线 | 完成 greedy、temperature=1、top-k/top-p、48-token 输出和 484-token 分块 prefill；逐层审计未发现非有限值 |
| Orthrus 更新最后 1 层 | 两 rank 各 47 个完整 K32 验证块、47 次在线更新、48 个接受候选；计数及 loss 一致 |
| greedy 对照 | 短文本、连续拒绝、分块 prefill 三个用例的 72 个输出 token 均与基线相同 |
| head 导出 | 两个 TP 分片及 manifest 成功保存到 version47；不包含第二份 backbone |
| head 重载并更新最后 4 层 | 重启加载 version47 后，两 rank 各完成 40 次更新，覆盖 3 个 GDN 和 1 个 Full Attention；greedy 再次逐 token 对齐 |
| 原生 OpenAI API | health/model、greedy、top-k/top-p、SSE、两个客户端排队、取消后恢复、1804-token 跨页 prefill、长请求后历史重置全部通过 |

两次在线测试的教师、loss、梯度均为有限值，TP 更新计数一致。最后 1 层结束时 TV loss 为 0.4872228503，最后 4 层为 0.3578712344。
这些数值来自不同请求和不同更新序列，不能用来宣称 loss 单调下降或最后 4 层收益更高。
HTTP 验收同样更新最后 4 层，两 rank 各完成 58 次更新，日志中的最终 TV loss 为 0.1591993，梯度范数为 3.5297961。
两个客户端是在 max_num_seqs=1 下排队，不是动态批处理；跨页用例验证可运行及后续历史隔离，不将它描述为长上下文的全量数值验收。

每 rank 的模型加载阶段内存约为 32.11 GiB（last-1）或 32.76 GiB（last-4），包含共享主模型、新增 draft mixer 和训练 master，不是完整运行峰值。
head 由 AR 参数复制初始化；此次没有完成离线 head 训练，也没有证明吞吐提升。

## 算子与采样验证

- GDN recurrent：17/31/32 行分段，以及上轮接受 27 行、本轮只算 8 行的恢复均在 NPU 通过。FP32 状态对 Torch 参考的最大绝对误差约 1.2e-7；输出约 1.82e-4。
- GDN prefill：1/8/32 行均有限；输出最大绝对误差分别为 6.10e-5、2.44e-4、2.44e-4。CANN 状态接口为 BF16，对 FP32 参考的状态最大误差为 0.00503409，含精度转换影响。
- 分页 FIA：覆盖 0/127/129/255 个历史位置、17/32 个 query、乱序物理页及跨页读取；KV 内容前后完全相同。对 FP32 SDPA 采用逐元素 rtol=0.025、atol=0.005，四组通过，整体最大绝对误差 0.00974393。
- 原生 rejection sampler：greedy/random/mixed 模式全部 31 个拒绝位置、全接受 bonus、固定种子批次独立性通过。32768 次抽样的目标概率为 [0.1,0.3,0.6]，观测为 [0.10229492,0.29669189,0.60101318]，通过六倍标准差阈值。

采样模式验证分布，不能要求开启与关闭投机时同种子输出序列相同。
对应脚本为 probe_qwen38_npu_gdn.py、probe_qwen38_npu_fia.py；采样检查直接调用镜像内的原生实现，没有重写采样或 Attention kernel。

## 回归与部署边界

缓存改造前 CPU 回归 130 passed、17 skipped，跳过项为 CUDA 专用测试，不是 NPU 通过记录。
覆盖权重共享、AR/HF 对照、原生参数复制、冻结权重输入反向、TP 前向与梯度、全局裁剪、last-1/last-4、到达行监督、请求切换、K32 元数据和 head 交付。
PIP_NO_INDEX=1 下 editable 安装通过；Ruff 检查通过。

当前不覆盖图模式、动态批处理、量化、PP/DP、超长上下文或生产 goodput。
当前插件复用 native cache 的内容，但仍有临时布局快照，不宣称已完成零拷贝或生产吞吐优化。
head 文件交付不含 AdamW 状态，重启不保证精确续接优化器轨迹。

本镜像原生 Triton prefill 在 target-only 中也出现过首层 GDN NaN，因而显式选择已有 CANN 后端；并非修改该算子的内核。
GDN 和分页 FIA 的 16-query 限制由现有算子的分段调用解决，K 仍为 32。forward 不依赖临时配置上下文，配置在 worker 初始化时绑定。
KV 预算需给 CANN 工作空间留余量，不能将原有 prefill profiling 数字直接当作新后端的完整峰值。

运行入口和限制见 [部署说明](QWEN38_VLLM.md)。
缓存改造前的测试结束时曾停止三台机器的专用容器，当时实查 61 八卡均无 NPU 进程；这不是当前共享机器的占用情况。

## 版本标识

```text
image: quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a
vLLM: 0.26.0+empty
vLLM-Ascend: 0.19.1rc2.dev1398+g85625bd77
Transformers: 5.14.1
PyTorch: 2.10.0+cpu + torch_npu

installed source SHA256:
model_runner_v1.py: c99d5172383c0bc3ef0f8b5722e87f558aa9e3d5f2e523b273bb1ff11e88a2e4
attention_v1.py: 4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076
ops/gdn.py: 4e0663d014503b49ce68e36b95a4dd728e33671983d4d8788db0dbfb9cea64f6

base config.json SHA256:
191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab
base model.safetensors.index.json SHA256:
77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df
```

完整运行日志在实验机 `/data/disk/wangzili/orthrus-qwen38-native-20261030/evidence/`，本地副本在工作区 `debug-20261030/qwen38-native-61/`。
关键结果文件为 target-cann-61.json、orthrus-cann-61.json、orthrus-reload-last4-61.json 和 api-61.json。
head 权重仅保留在实验机的 evidence/heads-native-61/version47，不随代码或本地日志副本发布。
权重索引哈希不等于全部权重内容哈希；正式部署仍需完整模型文件锁定。
