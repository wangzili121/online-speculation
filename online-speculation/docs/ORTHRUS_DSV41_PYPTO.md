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

Serving 校验 Lib 的工作树状态和 Git tree。公开入口运行基模，内部 runner 提供给定候选的 greedy 验证入口。Lib 的 `GenerationSession.draft_history` 根据 DP 分区、请求 slot 和 generation 绑定已经提交的行，拒绝过期请求、跨分区访问和错误的 anchor。各层的滑窗页、压缩源层页与 indexer 页直接返回带类型的设备视图，连同行步长、偏移和物理页表供 draft 读取。常驻运行时拒绝内核将该视图绑定为输出或原地写入参数，主模型仍可通过原有入口提交 KV。

在干净检出中重放两份补丁并通过源码配对检查。使用 PyPTO `72a62d13` 及其固定的 Simpler `6b6b88f`，Lib 的 13 项常驻资源和缓存测试、Serving 的 75 项 CPU 测试全部通过。PTOAS 使用该工具链要求的 v0.70。 [PyPTO 提交](https://github.com/hw-native-sys/pypto/commit/72a62d13efabbb25d240e35944f68ba5bcc22b7f) · [PTOAS v0.70](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.70)

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

压缩历史程序接收各 draft query 已选中的物理 KV 行。每 128 行计算一段输出和 LSE，再在设备端与 SWA 和临时块的结果合并。SWA 侧已经计入 sink。连通 draft 单层时，压缩程序直接接收 SWA 在设备上输出的值和 LSE，并在设备端换成压缩段合并所需的布局。

在命令末加 `--heads 16 --compressed` 并指定新的输出路径，即可验证 A3 TP4 的单 rank 形状。用配套的 PyPTO、Simpler 和 PTOAS 在单张 910B3 上通过六组用例，覆盖 C1A/C2A、部分压缩组、乱序物理页、不同 query 的有效长度以及最多 512 行。C2A 使用每页 64 个有效压缩行、128 行物理跨度，避免选择行落到 padding 上。与 FP32 reference 比较，联合输出和 LSE 最大误差分别为 `5.90e-6` 和 `9.54e-7`，SWA 和压缩缓存逐字节一致。 [压缩历史源码](../src/blockspec/pypto_dsv41_sparse.py) · [TP4 单 rank 记录](validation/dsv41/pypto-compressed-tp4-910b3.json) · [模拟器记录](validation/dsv41/pypto-compressed-a2a3sim.json)

## Draft 单层投影和 Attention

先按 BF16 head 契约校验模型和权重身份。每个 TP rank 复制 Q 的低秩投影和潜在 KV 投影，按 head 切分 Q 的第二级投影，按完整分组切分 Wo-A 和相应的 Wo-B 列。投影复用 A3 的 BF16 GEMM、RMSNorm、RoPE 和分组输出实现。Q 和临时 KV、SWA 输出与 LSE，以及合并后的 Attention 输出都在同一个设备 worker 内交接。该单层入口输出当前 TP rank 对隐状态的贡献。

```bash
PYTHONPATH=src python3 scripts/validate_dsv41_pypto_qkv.py \
  --lib "$PYPTO_LIB_ROOT" --model "$TARGET_WEIGHTS" --head "$DRAFT_HEAD" \
  --layer 2 --ratio 2 --device 0 --output /tmp/dsv41-draft-c2a.json
```

单张 910B3 用从 DSV4.1 W4A8 checkpoint 初始化的 BF16 head、合成隐藏状态和历史页，测试第 0、2、20 层的 TP4 单 rank 形状。压缩路径的选中行由测试提供，覆盖最多 512 行和跨页访问。投影各算子与同输入的 BF16 参考分别对照，Attention 与同输入的 FP32 参考对照，历史页在执行前后逐字节一致。C2A 在 A3 固定的 PyPTO `72a62d13`、Simpler `6b6b88f`、PTOAS `0.70` 上复跑，误差结果一致。 [投影源码](../src/blockspec/pypto_dsv41_draft.py) · [实测记录](validation/dsv41/pypto-draft-layer-tp4-910b3.json)

## Draft Indexer

拥有 indexer 的层从 BF16 head 加载自己的 query 和权重投影。每个 draft 位置用自己的投影结果打分，全部 32 个位置共用同一个历史边界 `anchor // compress_ratio`。PyPTO 按页表读取 INT8 key 和 FP16 scale，复用 A3 QLI 的 query 量化、打分和 Top-512。选出的逻辑行在设备上转为压缩 KV 物理行，再交给压缩 Attention。

当前 A3 上下文上限为 2048 个 token。C1A 历史最多涉及 256 个 8 行候选块，checkpoint 允许保留 2048 个候选块，因此 Reindex 对全部可见历史重新打分与在候选池内打分等价。设备入口会拒绝候选池需要裁剪的配置。压缩历史消费层跨层复用选择结果，由整模接入时的 draft 调度负责。

在单层验证命令上加 `--indexer --anchor 1537` 即可验证第 2 层自行选行，`--layer 20 --ratio 1` 和 `--layer 24 --ratio 1` 分别用于 C1A Full 和 Reindex。固定工具链下，单张 910B3 跑过 7 组 TP4 单 rank 用例，覆盖空历史、短历史、奇数 anchor、跨页和 512/1024/2048 行桶。224 个查询的选中集合与 A3 QLI 参考一致，压缩 Attention 最大绝对误差为 `1.17e-5`，历史页逐字节一致。测试使用 BF16 初始化 head 和构造的分页历史。 [Indexer 源码](../src/blockspec/pypto_dsv41_indexer.py) · [实测记录](validation/dsv41/pypto-draft-indexer-tp4-910b3.json)

## K32 验证和提交

`A3Runner.verify_draft(batch, candidates)` 接收请求账本已分配页的 anchor 和最多 31 个候选。它用每 DP 128 行的原生因果 Prefill 程序验证整块，读取每个候选位置对应的 logits。验证期在设备端保存涉及的目标页，完成后原地恢复。CPU 上的 Engram 请求历史和已提交长度也恢复。runner 再用同一个主模型程序执行 anchor 和接受前缀，将其 KV 正式写入。目标补发的 token 留给下一轮处理。这个路径会重跑接受前缀，用于先验证缓存和输出语义，整模吞吐需要单独测。

CPU 测试覆盖完整 K32、部分接受、零接受、跨 128 行分页以及多个 DP 的请求顺序。单张 910B3 验证了设备范围快照在正常返回和异常退出时均可逐字节恢复。该用例直接测试设备复制和恢复接口，16 卡整模验证仍待执行。

## 整模联调

1. **Draft 前向。** 在整模 resident session 中绑定 diffusion head 和各层已提交的历史页。连起 40 层的单层 Attention、跨层 indexer 选择复用，以及共享的 Embedding、Engram、mHC、MoE 和 LM Head，再对 TP 的输出贡献做归约。
2. **开启 Serving 调度。** 将运行中的 draft 提案接到验证入口，让调度器为候选位置预留页，并覆盖尾部普通 Decode 回退。之后在 16 张 A3 上回放同一候选序列，和 target-only 对照输出 token、SWA、压缩 KV、indexer、compressor 残余状态及 Engram 历史。 [V4.1 DSpark 入口](https://github.com/hw-native-sys/pypto-serving/pull/324) · [候选回放脚本](../scripts/probe_dsv41_matched_candidates.py)
