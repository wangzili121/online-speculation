# Qwen3.8 原生 vLLM 验证记录

## 2026-10-31 扩展验收

**扩展验收尚未全部通过，不能用下方较早的通过记录代替新用例的结果。**
真实 27B、TP2、8K 上下文的 K32 检查中，16 个 greedy 用例有 15 个逐 token 一致，JSON 提示在第二个输出 token 处分叉。
同一模型、同一提示、同一 draft head，只改变实际提交的候选数：0/1/7/15/16 个候选均与无投机一致，31 个候选出现分叉。
无投机该位置的前两名目标 logits 为 16.625 和 16.5；32 行验证时出现三个 16.625 的并列值，原生 argmax 选择发生变化。
目前证据说明差异出现在目标验证 logits 中，尚不据此断言具体 kernel 根因。

同一 K32 运行的 64 种子、128 个投机输出位置的采样统计通过；参考概率直接取自原生采样前 logits。
固定 head 的相同 seed 请求复现检查通过；两个 TP rank 的计数一致，每 rank 248 次缓存只读抽查通过。
两 rank 全量主模型参数运行前后 SHA256 相同，覆盖合计 53,793,351,168 字节。
映射 AR 视图与原生验证的额外数值对照未达到预设 TV<=0.05：31 行最大 TV=0.09108947，首行 TV=0.02189109。
这项诊断不能登记为通过，也不能仅凭阈值超出便认定参数映射有误；需与目标后端的块长度数值差异一同分析。

相关回归为 221 passed、16 个 CUDA 专用测试 skipped，另新增的两项诊断工具单测通过。
后续在线更新、保存重载和 API 的扩展结果将单独补齐；以下为 2026-10-30 及之前已完成的验证范围。

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
