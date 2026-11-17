# Orthrus-DSV4.1 的 PyPTO 接入

DSV4.1 Flash 的 A3 路径使用 W8A8 权重和 DP4/TP4/EP16，Engram 大表在 CPU，按本轮 token 查表并传输查得的行。整模和请求缓存由同一个原生 session 管理。公开 A5 路径中的 DSpark 调度可作为 Orthrus 的接入参考。 [A3 Serving](https://github.com/Adelaide999/pypto-serving/tree/cd818945c3c4e5454b8c259306d4e6f9e865e05c/models/deepseek_v41_a3) · [V4.1 DSpark](https://github.com/hw-native-sys/pypto-serving/pull/324)

## A3 Serving 基线

Lib 基线为 `8dfbec1`，Serving 基线为 `cd81894`。原 Serving 固定了较早的 `cabe227`；新版 Lib 保留了原生优化，但需要移植请求池、按请求清理以及接收 Serving 所生成 Decode 元数据的入口。以下补丁将两边配对，并保持 Engram 的 HOST 加载路径。 [Lib 基线](https://github.com/Adelaide999/pypto-lib/commit/8dfbec1e11975cd3a8b2e7b2f8f45545de9bf360) · [Lib 补丁](../runtime/pypto_dsv41_a3/lib-serving.patch) · [Serving 补丁](../runtime/pypto_dsv41_a3/serving-pair.patch)

在三个仓库均已检出的情况下，将下列变量设为对应的绝对路径。在干净的工作分支应用补丁。

```bash
git -C "$PYPTO_LIB_ROOT" switch -c feature/orthrus-dsv41-a3 8dfbec1e11975cd3a8b2e7b2f8f45545de9bf360
git -C "$PYPTO_LIB_ROOT" am "$ORTHRUS_ROOT/online-speculation/runtime/pypto_dsv41_a3/lib-serving.patch"
git -C "$PYPTO_SERVING_ROOT" switch -c feature/orthrus-dsv41-a3 cd818945c3c4e5454b8c259306d4e6f9e865e05c
git -C "$PYPTO_SERVING_ROOT" am "$ORTHRUS_ROOT/online-speculation/runtime/pypto_dsv41_a3/serving-pair.patch"
export PYPTO_LIB_ROOT
```

Serving 校验 Lib 的工作树状态和 Git tree。补丁只开放基模 Serving 入口。Lib 的 `GenerationSession.draft_history` 根据 DP 分区、请求 slot 和 generation 绑定已经提交的行，拒绝过期请求、跨分区访问和错误的 anchor。各层的滑窗页、压缩源层页与 indexer 页直接返回带类型的设备视图，连同行步长、偏移和物理页表供 draft 读取。常驻运行时拒绝内核将该视图绑定为输出或原地写入参数，主模型仍可通过原有入口提交 KV。

在干净检出中重放两份补丁并通过源码配对检查。使用 PyPTO `72a62d13` 及其固定的 Simpler `6b6b88f`，Lib 的 12 项常驻资源和缓存测试、Serving 的 74 项 CPU 测试全部通过。PTOAS 使用该工具链要求的 v0.70。 [PyPTO 提交](https://github.com/hw-native-sys/pypto/commit/72a62d13efabbb25d240e35944f68ba5bcc22b7f) · [PTOAS v0.70](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.70)

## K32 SWA

PyPTO 程序接收 RoPE 后的 32 行 query、128 行分页滑窗的只读视图、32 行临时 latent KV，以及每头的 Attention sink。输入页号经过宿主侧的 `swa_spans` 转为最多两段物理行。设备只搬读对应的历史行。所有 draft 位置读取同一 anchor 历史和整块临时 KV。两部分分数和 sink 使用同一个 FP32 softmax，输出 FP32 Attention 结果和 LSE。

验收输入包括历史为空、滑窗刚满、跨乱序物理页和长前缀。所有不可见缓存行用 NaN 填充，设备输出与现有 Torch FP32 golden 比较，并逐字节复读目标缓存、元数据和临时块。默认用 8 个本地 query head、512 维 latent，与实模 TP8 的单 rank 一致。 [PyPTO 源码](../src/blockspec/pypto_dsv41_swa.py) · [Torch golden](../src/blockspec/vllm_ascend/dsv41_attention.py)

```bash
PYTHONPATH=src python3 scripts/validate_dsv41_pypto_swa.py \
  --device 0 --output /tmp/dsv41-swa-a2.json
```

设备编号以 PyPTO 进程可见的设备为准。`--simulate` 使用 A2/A3 模拟器，`--compile-only` 仅验证代码生成与 PTOAS 编译。

模拟器与 Ascend 910B3 各通过九组用例，输出最大绝对误差分别为 `5.36e-6` 和 `5.84e-6`，LSE 最大误差均为 `9.54e-7`。PV 计算将 FP32 softmax 概率拆为 BF16 值和 BF16 残差，两次乘法均累加到 FP32，减小近零输出的误差。实机各组的独立进程调用中位数约 1.0 ms，含 PyPTO 调度和同步，整模接入后的耗时需单独测量。 [模拟器记录](validation/dsv41/pypto-swa-a2a3sim.json) · [910B3 记录](validation/dsv41/pypto-swa-910b3.json)

## C1A/C2A 压缩历史 Attention

压缩历史程序接收各 draft query 已选中的物理 KV 行。每 128 行计算一段输出和 LSE，再在设备端与 SWA 和临时块的结果合并。SWA 侧已经计入 sink，各压缩分段无需再计。当前 demo 通过宿主机转换两段间的张量布局，后续 Serving 增加设备侧直接绑定。

在命令末加 `--heads 16 --compressed` 并指定新的输出路径，即可验证 A3 TP4 的单 rank 形状。用配套的 PyPTO、Simpler 和 PTOAS 在单张 910B3 上通过六组用例，覆盖 C1A/C2A、部分压缩组、乱序物理页、不同 query 的有效长度以及最多 512 行。C2A 使用每页 64 个有效压缩行、128 行物理跨度，避免选择行落到 padding 上。与 FP32 reference 比较，联合输出和 LSE 最大误差分别为 `5.90e-6` 和 `9.54e-7`，SWA 和压缩缓存逐字节一致。 [压缩历史源码](../src/blockspec/pypto_dsv41_sparse.py) · [TP4 单 rank 记录](validation/dsv41/pypto-compressed-tp4-910b3.json) · [模拟器记录](validation/dsv41/pypto-compressed-a2a3sim.json)

## 进入整模前的接口

1. **C2A/C1A 检索。** 将同一个 anchor 的已提交长度、源层的索引页和 diffusion query 交给检索。Full 计算本轮 Top-K，C1A Reindex 在本轮候选池内重排，Reuse 复用源层结果。先与现有 Torch golden 对照整数行号，再评估 A2/A3 上已有打分和 Top-K 内核的输入布局。
2. **Draft 前向。** 在同一 resident session 中绑定 diffusion head、主模型共享权重以及各层已提交的历史页。将检索、SWA 和压缩 Attention 串成 K32 的设备执行，让临时块的 KV 留在独立工作缓冲区。
3. **整块验证。** 当前 A3 Serving 每次 Decode 只处理一个 anchor，native service 每个 DP dispatch 最多 16 行。需要单独提供 K32 验证入口，在临时状态上完成验证，再由请求账本提交接受前缀。缓存验收要覆盖 SWA、压缩 KV、indexer、compressor 残余状态和 Engram 历史。DSpark 的事务与 DecodeResult 组织方式可以借鉴。整模执行和精度验收需要 16 张 A3。 [V4.1 DSpark 入口](https://github.com/hw-native-sys/pypto-serving/pull/324) · [候选回放脚本](../scripts/probe_dsv41_matched_candidates.py)
