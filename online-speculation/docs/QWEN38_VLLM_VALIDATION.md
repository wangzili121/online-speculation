# Qwen3.8 原生 vLLM 验证记录

日期：2026-10-30。对象为真实 Qwen3.8-27B BF16 checkpoint，不是缩小模型。

## 当前判断

**真实 27B 的原生 vLLM 起草、验证、拒绝恢复、在线更新和 head 保存重载已通过功能验证。**
不是将独立 Torch 执行器包装成 HTTP：主模型、正式缓存和 rejection sampler 均由原生 vLLM-Ascend 执行。
原生 OpenAI API 的 8 类接口检查也全部通过。本记录不代表生产性能验收。

## 真实模型验证

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

CPU 回归 130 passed、17 skipped，跳过项为 CUDA 专用测试，不是 NPU 通过记录。
覆盖权重共享、AR/HF 对照、原生参数复制、冻结权重输入反向、TP 前向与梯度、全局裁剪、last-1/last-4、到达行监督、请求切换、K32 元数据和 head 交付。
PIP_NO_INDEX=1 下 editable 安装通过；Ruff 检查通过。

当前不覆盖图模式、动态批处理、量化、PP/DP、超长上下文或生产 goodput。
插件仍通过共享 AR 权重重放已提交历史，未直接读取 native paged KV/GDN cache；这是功能路径，不是最终高效共享缓存实现。
head 文件交付不含 AdamW 状态，重启不保证精确续接优化器轨迹。

本镜像原生 Triton prefill 在 target-only 中也出现过首层 GDN NaN，因而显式选择已有 CANN 后端；并非修改该算子的内核。
GDN 和分页 FIA 的 16-query 限制由现有算子的分段调用解决，K 仍为 32。forward 不依赖临时配置上下文，配置在 worker 初始化时绑定。
KV 预算需给 CANN 工作空间留余量，不能将原有 prefill profiling 数字直接当作新后端的完整峰值。

运行入口和限制见 [部署说明](QWEN38_VLLM.md)。
测试结束已停止本次三台机器的专用容器。释放后实查 61 八卡均无 NPU 进程，没有停止他人的任务。

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
