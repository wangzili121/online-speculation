# Orthrus-DSV4.1 的 PyPTO 接入

公开 `pypto-serving` 已有 V4.1 executor、权重加载、缓存规划和 DSpark 调度。V4.1 的公开设备路径为 A5，加载 FP8/FP4 格式。A2/A3 的 V4.1 整模部署需要为 1017 W4A8 checkpoint 建立权重加载、MoE 和目标侧分页缓存基线。 [PyPTO Lib 平台矩阵](https://github.com/hw-native-sys/pypto-lib/blob/main/.github/ci_platforms.json) · [V4.1 executor](https://github.com/hw-native-sys/pypto-serving/blob/main/pypto_serving/model/deepseek_v41/npu_executor.py#L148-L154) · [权重格式](https://github.com/hw-native-sys/pypto-serving/blob/main/pypto_serving/model/deepseek_v41/weight_spec.py#L98-L114)

## K32 SWA

PyPTO 程序接收 RoPE 后的 32 行 query、128 行分页滑窗的只读视图、32 行临时 latent KV，以及每头的 Attention sink。输入页号经过宿主侧的 `swa_spans` 转为最多两段物理行。设备只搬读对应的历史行。所有 draft 位置读取同一 anchor 历史和整块临时 KV。两部分分数和 sink 使用同一个 FP32 softmax，输出 FP32 Attention 结果和 LSE。

验收输入包括历史为空、滑窗刚满、跨乱序物理页和长前缀。所有不可见缓存行用 NaN 填充，设备输出与现有 Torch FP32 golden 比较，并逐字节复读目标缓存、元数据和临时块。默认用 8 个本地 query head、512 维 latent，与实模 TP8 的单 rank 一致。 [PyPTO 源码](../src/blockspec/pypto_dsv41_swa.py) · [Torch golden](../src/blockspec/vllm_ascend/dsv41_attention.py)

```bash
PYTHONPATH=src python3 scripts/validate_dsv41_pypto_swa.py \
  --device 0 --output /tmp/dsv41-swa-a2.json
```

设备编号以 PyPTO 进程可见的设备为准。`--simulate` 使用 A2/A3 模拟器，`--compile-only` 仅验证代码生成与 PTOAS 编译。

模拟器与 Ascend 910B3 各通过九组用例，输出最大绝对误差分别为 `5.36e-6` 和 `5.84e-6`，LSE 最大误差均为 `9.54e-7`。PV 计算将 FP32 softmax 概率拆为 BF16 值和 BF16 残差，两次乘法均累加到 FP32，减小近零输出的误差。实机各组的独立进程调用中位数约 1.0 ms，含 PyPTO 调度和同步，整模接入后的耗时需单独测量。 [模拟器记录](validation/dsv41/pypto-swa-a2a3sim.json) · [910B3 记录](validation/dsv41/pypto-swa-910b3.json)

## 进入整模前的接口

1. **C2A/C1A 检索。** 将同一个 anchor 的已提交长度、源层的索引页和 diffusion query 交给检索。Full 计算本轮 Top-K，C1A Reindex 在本轮候选池内重排，Reuse 复用源层结果。先与现有 Torch golden 对照整数行号，再查能否复用 A2/A3 上的已有打分和 Top-K 内核。
2. **压缩历史 Attention。** 用选中行去只读源层 KV 页，对每个 query 计算压缩历史的输出和 LSE，再与 SWA 和临时块的结果合并。SWA 程序已经把 sink 计入其归一化，压缩历史一侧省去 sink。合并可用现有联合 softmax golden 验证。
3. **Serving。** 基模单跑通过后，把权重绑定、只读页表、K32 中间缓冲和 proposer/verifier 接入 executor。原生 DSpark 的候选提交、取消和回收接口可作为参考。用同一候选序列回放，独立核对验收输出与缓存状态，再测启用投机前后的吞吐。 [V4.1 DSpark 入口](https://github.com/hw-native-sys/pypto-serving/pull/324) · [vLLM 回放脚本](../scripts/probe_dsv41_matched_candidates.py)
