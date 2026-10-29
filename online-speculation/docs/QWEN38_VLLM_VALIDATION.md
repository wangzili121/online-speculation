# Qwen3.8 原生 vLLM 验证记录

日期：2026-10-30。对象为真实 Qwen3.8-27B BF16 checkpoint，不是缩小模型。

## 当前判断

**不能宣布“原生推理与在线更新已全部跑通”。** 接入代码、TP 权重映射和在线训练逻辑已有单元证据；真实 27B 的加载、K32 proposer 初始化与 native verifier 调用已执行。
完整服务、greedy 对照、采样正确性和原生 worker 内的成功在线更新仍需两卡复测。
此前独立 Torch 执行器的真实 27B 成功记录不替代这次 vLLM 验收。

## 已验证

CPU 回归：118 passed、17 skipped。跳过项均需 CUDA，不是已通过的 NPU 测试。
覆盖 native 参数别名、AR/HF 对照、vLLM 参数类型复制、inference tensor 的冻结输入反向、两进程 TP 前向/梯度/全局裁剪、last-1/last-4 后缀、仅实际到达教师行监督、请求切换和 head 交付格式。
原有 Qwen3/Qwen3.8 后缀训练、冻结参数及恢复逻辑同时回归。Ruff 和 Python 编译检查通过。

在 167 的 Ascend 910B3 上，真实 27B 已以原生 vLLM TP=2 完成加载。
共享 backbone 后，每 rank 加载及 draft/FP32 master 的内存记录约 32.11 GiB；没有第二份完整主模型。
K32 初始化越过了原生 Attention 的 16-query decode 限制，实际调度记录包含一次 32-token 模型验证；但当时教师状态已异常，不能作为完整数值正确性证据。

现有 CANN GDN primitive 独立探测结果：

| query 数 | 输出是否有限 | 对 Torch 参考输出最大绝对误差 | 最终状态最大绝对误差 |
| --- | --- | --- | --- |
| 1 | 是 | 0.0000610352 | 0.00224733 |
| 8 | 是 | 0.000244141 | 0.00411105 |
| 32 | 是 | 0.000244141 | 0.00503409 |

参考按相同 BF16 Q/K/V、GQA 分组和缩放计算；参考状态以 FP32 累积，CANN 接口状态为 BF16，因此状态误差含精度转换影响。
这只证明所测形状的 primitive 可用，不能据此宣称完整 27B 或长上下文已通过。

## 端到端阻塞

数值审计将 NaN 定位到主模型第 0 层 GDN 的输出归一化之前，QKV/门控投影输入输出仍为有限值。
**关闭 Orthrus、不加载任何 draft 的原生 target-only 对照也在同一位置出现异常。** 单纯“请求返回了 8 个 token”不能当作正确性通过；greedy 遇到 NaN 仍可能返回 token 0。
这与社区报告的 [910B GDN prefill 首次执行异常](https://github.com/vllm-project/vllm-ascend/issues/17009) 现象相似，但本次未证明二者底层根因完全相同。

当前增加的是显式 CANN prefill 后端选择，使用镜像已有的 `torch_npu.npu_chunk_gated_delta_rule`，不重写 kernel。
完整融合后端复测尚未生成成功结果：重新启动时 0、1 卡已由另一组 vLLM worker 占用，空余约 14.59 GiB/卡，低于本次启动要求。
没有停止其他人的任务。

## 下次验收

两卡窗口下，先以同一 CANN 后端跑 target-only，再运行 K32 Orthrus，最后启动原生 OpenAI API：

1. greedy 输出逐 token 对照，覆盖多轮拒绝后的 GDN/KV 状态。
2. temperature=1 跑通真实 q 分布校正，再补分布检验；不能要求与无投机同种子序列完全相同。
3. worker RPC 确认每个 TP rank 有真实 proposal/verification/update，更新计数及参数版本一致，教师、loss、梯度均有限。
4. 覆盖新请求、长一些的 prefill、请求结束/取消、head 保存重启与拒绝后恢复。
5. 再处理直接读取 native KV/GDN history、并发和性能优化；当前历史重放不是最终高效共享缓存实现。

脚本入口及参数见 [部署说明](QWEN38_VLLM.md)。当前脚本是复测入口，不是已经验收的一键生产部署包。

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

完整运行日志在实验机 `/data/disk/wangzili/orthrus-qwen38-native-20261030/evidence/`，本地副本在工作区 `debug-20261030/qwen38-native/`。
权重索引哈希不等于全部权重内容哈希；正式部署仍需完整模型文件锁定。
