# Qwen3.8 原生 vLLM 验证记录

日期：2026-10-30。对象为真实 Qwen3.8-27B BF16 checkpoint，不是缩小模型。

## 当前判断

**真实 27B 的原生 vLLM 起草、验证、拒绝恢复、在线更新和 head 保存重载已通过功能验证。**
不是将独立 Torch 执行器包装成 HTTP：主模型、正式缓存和 rejection sampler 均由原生 vLLM-Ascend 执行。
原生 OpenAI API 的 8 类接口检查也全部通过。本记录不代表生产性能验收。

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
