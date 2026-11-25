# Orthrus-DSV4.1 的 PyPTO 接入

DSV4.1 Flash 的 A3 路径使用 W8A8 权重和 DP4/TP4/EP16，Engram 大表在 CPU，按本轮 token 查表并传输查得的行。整模和请求缓存由同一个原生 session 管理。公开 A5 路径中的 DSpark 调度可作为 Orthrus 的接入参考。 [A3 Serving](https://github.com/Adelaide999/pypto-serving/tree/cd818945c3c4e5454b8c259306d4e6f9e865e05c/models/deepseek_v41_a3) · [V4.1 DSpark](https://github.com/hw-native-sys/pypto-serving/pull/324)

全新机器先按 [Docker 环境准备](ORTHRUS_DSV41_PYPTO_ENVIRONMENT.md) 构建或导入配套镜像，然后在容器内执行完整预检。

## A3 Serving 基线

Lib 基线为 `e32726b`，Serving 基线为 `cd81894`。Lib 提供 8192-token Prefill 分块和 16384-token 总上下文。原 Serving 固定了较早的 `cabe227`；新版 Lib 保留了原生优化，但需要移植请求池、按请求清理以及接收 Serving 所生成 Decode 元数据的入口。以下补丁将两边配对，并保持 Engram 的 HOST 加载路径。 [Lib 基线](https://github.com/Adelaide999/pypto-lib/commit/e32726b7edb31ee17e398a0431c0272ebd545974) · [Lib 补丁](../runtime/pypto_dsv41_a3/lib-serving.patch) · [Serving 补丁](../runtime/pypto_dsv41_a3/serving-pair.patch)

在三个仓库均已检出的情况下，将下列变量设为对应的绝对路径。在干净的工作分支应用补丁。

```bash
git -C "$PYPTO_LIB_ROOT" switch -c feature/orthrus-dsv41-a3 e32726b7edb31ee17e398a0431c0272ebd545974
git -C "$PYPTO_LIB_ROOT" am "$ORTHRUS_ROOT/online-speculation/runtime/pypto_dsv41_a3/lib-serving.patch"
git -C "$PYPTO_SERVING_ROOT" switch -c feature/orthrus-dsv41-a3 cd818945c3c4e5454b8c259306d4e6f9e865e05c
git -C "$PYPTO_SERVING_ROOT" am "$ORTHRUS_ROOT/online-speculation/runtime/pypto_dsv41_a3/serving-pair.patch"
export PYPTO_LIB_ROOT
```

Serving 校验 Lib 的工作树状态和 Git tree。可通过 `--speculative-config` 启用 Orthrus，缺省走基模原生 Decode。Lib 的 `GenerationSession.draft_history` 根据 DP 分区、请求 slot 和 generation 绑定已经提交的行，拒绝过期请求、跨分区访问和错误的 anchor。各层的滑窗页、压缩源层页与 indexer 页直接返回带类型的设备视图，连同行步长、偏移和物理页表供 draft 读取。常驻运行时拒绝内核将该视图绑定为输出或原地写入参数，主模型仍可通过原有入口提交 KV。

在干净检出中重放两份补丁并通过源码配对检查。使用 PyPTO `72a62d13` 及其固定的 Simpler `6b6b88f`，Lib 的 13 项常驻资源和缓存测试、Serving 的 75 项 CPU 测试全部通过。PTOAS 使用该工具链要求的 v0.70。 [PyPTO 提交](https://github.com/hw-native-sys/pypto/commit/72a62d13efabbb25d240e35944f68ba5bcc22b7f) · [PTOAS v0.70](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.70)

### 1122 迁移检查（2026-11-25）

A3 W8A8 开发分支 `deepseek_v4.1_flash_1122` 更新到 `4e198fa`，包含 Prefill MoE 和 Indexer、Decode RMSNorm/RoPE、SWA、激活量化及宽批次归约的优化。公开主仓 `f957537` 的 V4.1/DSpark 实现面向 A5。已提供的 A3 Serving `27efbcb` 固定配对 `1121/dabf9e9`。 [A3 1122](https://github.com/Adelaide999/pypto-lib/tree/4e198fa9d10caf1911ddbe30d87e44bdc0337ac1/models/deepseek_v4_1_flash_a3) · [A3 1121 Serving](https://github.com/Adelaide999/pypto-serving/tree/27efbcb1cd3466547823955cb80fe09c63b3f754/models/deepseek_v41_a3) · [A5 V4.1](https://github.com/hw-native-sys/pypto-lib/blob/f957537eb0d3b1f5c3072569330e521554e38e40/docs/models/deepseek_v4_1_flash/index.md)

在独立 checkout 和 111 的 `72a62d1-offline` 镜像上，1122 的两个 CLI 可导入。用各自 checkout 自带的 golden runner 在 111 和 167 复测 `c33efdd`、`4e198fa`，容量 16、上下文容量 256 的 Prefill 和 Decode 编译均在旧版工具链的 JIT 形状特化时失败。Prefill 首处报 `_gate_panel_from_fp32__16/xg_buf` 缺少张量元数据，Decode 首处报 `gemm_int8_tiled__13/payload` 缺少张量元数据。编译阶段未使用 NPU。现有 Lib 补丁依赖的 `network.py`、`prefill_layer.py`、`prefill_resident.py` 等文件在 1122 已被替换。源码配对检查保留当前验收基线。

167 的独立候选镜像使用 PyPTO `2e31e0c2`、Simpler `a5349f4` 和 PTOAS `v0.78`。只构建 Simpler 的 `build_package_a2a3`，沿用 `runtime-release.patch` 解决 Release 下的未使用变量。用 `4e198fa` 源码编出 Prefill 容量 16、Decode 容量 16、上下文容量 256 的两张图，编译分别为 154.05 秒和 195.48 秒。开启通信窗口预分配后，上下文容量 512 的 Prefill 容量 32 和 Decode 容量 16 同样通过，分别为 153.54 秒和 186.46 秒。设备探针另需备好该 Simpler 版本的 PTO-ISA checkout。167 的物理卡 7 在新版和原镜像下均报初始化错误 `507899`，因此整图通过的范围是编译阶段。Decode 编译器同时报告了并发标量写入告警，原生 AR 实卡验收仍需检查重复输出和缓存。

新版接入分支以 `94d5dfc` 为基线。该版本进一步复用 Prefill 投影的片上 tile，并修正单 token Decode 的 mHC 精度。PTOAS v0.78 下，通信窗口预分配、上下文容量 512 的 Prefill 32 和 Decode 16 分别在 156.24 秒和 188.23 秒完成编译。 [上游源码](https://github.com/Adelaide999/pypto-lib/tree/94d5dfc19b900a15860fd4bdc1d57e8c4bc156be/models/deepseek_v4_1_flash_a3)

Orthrus 直接读取 1122 的 packed KV。滑窗 FP8 以 16 行为一组，连续行用整块加载，跨页用索引加载，空 tile 直接置零。每层的 128 行滑窗只解包一次，供同一 draft 块内所有 query 共用。FP4 支持按所选物理行解包，C2A 以每页 128 个**压缩后**的行寻址。Verify 的字节区间覆盖写入的 SWA、压缩 KV、Indexer 数据及各 C2A 源的滚动状态。全接受时可以保留验证图已写入的缓存，发生拒绝时恢复快照并重放已接受前缀。当前快照调用和测试使用独立接口，尚待绑定整图的驻留缓冲区。

在 1122 的配套工具链中，把 1122 checkout 放入 `PYTHONPATH`，可独立检查压缩历史的读取和 K8/K16/K32 滑窗链路。

```bash
PYTHONPATH="$PWD/online-speculation/src:$PYPTO_LIB_ROOT:${PYTHONPATH:-}" \
  python online-speculation/scripts/validate_dsv41_pypto_packed_history.py --compile-only --swa
```

A2/A3 模拟器通过了 129 个物理页、Top‑512、K8/K16/K32 共 12 个用例。FP8 和 FP4 解包与 Torch 参考逐元素一致。双向滑窗输出最大绝对误差为 `3.82e-6`，LSE 为 `9.54e-7`。这些结果覆盖独立 kernel 和驻留缓冲区交接。实卡启动需要另验，167 的单卡探针在 `rtSetDevice` 报 `507899`。

167 已构建独立镜像 `orthrus-dsv41-pypto:1122-94d5dfc`，并在禁用网络的干净容器中通过 `--compile-only --swa --pages 129`。镜像入口用于 1122 算子验证。

新版 `generation.py` 通过一个 L3 worker 管理 16 个本地设备。双机 Serving 要将全局 EP rank 的分派移到 L4，并接入 Simpler 的 A3 Fabric 跨机设备窗口。旧 Serving 的 HCCL 验收覆盖逐层通信。新版的 `global_tload_mixed_l3` 已在单机回环模拟中验证一个本地和一个服务端 L3 worker，两个 rank 的小张量结果误差均为零。真实跨机 Fabric 仍待验证。随后用同一 checkpoint 核对 AR、Draft/Verify/Accept 的逐 token 一致性。在线更新还需绑定新版的反向输入和 head 发布路径。当前 GSM8K 命令走独立旧版基线。 [Simpler 跨机验证用例](https://github.com/hw-native-sys/simpler/tree/a5349f42c71666bb8cacfb3759f34eeda1572beb/examples/workers/l4/global_tload_mixed_l3)

### 1122 最新基线与 Verify/Accept

首轮迁移使用上游 `7abc0df`，启用 v8 权重布局。通信窗口预分配、上下文容量 512 的原生 Prefill 32 和 Decode 16 编译分别通过，耗时 141.63 秒和 200.21 秒。加入 Orthrus 回放模式的 Decode 图以相同形状重新编译通过。 [首轮上游](https://github.com/Adelaide999/pypto-lib/tree/7abc0df731143f8cee99c9bfc2bb731523384deb/models/deepseek_v4_1_flash_a3)

- FP8 和 FP4 的码表每种占 1 KiB，可随权重上传一次，供同一 worker 的各层复用。129 页和 Top-512 下，12 个 A2/A3 模拟器用例通过。解包输出逐元素一致，K8/K16/K32 滑窗链路的最大输出误差为 `3.82e-6`。附加 `--compare-swa-chain` 比较驻留输入的单次与两次提交，三种块宽的输出和 LSE 在模拟器中逐元素相同。
- Verify 直接调用新版驻留 Decode 图。固定 head 的贪心路径只回读每行 32 字节的采样结果，最多 16 行。每轮只上传当前 TP 组的输入，DP 归属切换时清空前一组。C2A 的滚动状态需提前备份，页缓存的未提交后缀由绝对位置和因果可见范围控制。
- 整块接受时保留当前图写入的缓存。遇拒绝时恢复 C2A 状态，再回放 anchor 和接受前缀，复用本轮已读出的 embedding 与 Engram 行。使用 `--prepare-windows` 编出的 Decode 图可在回放时跳过 LM Head。Serving 分别记录输入准备、上传、图调用、备份、回滚与采样回读量。
- 镜像中的 `generate --orthrus-verify-k 8` 或 `16` 在同一次权重加载下先取 AR 序列，再交替注入正确块和有意拒绝的块，逐 token 对照，并记录 Verify/Accept 分段数据。它需要配套的 Prefill/Decode 编译产物和原生入口要求的设备。当前完成编译、模拟和软件回归，实卡整模逐 token 结果待验收。此入口用于验证候选处理，真实 draft 及在线更新仍需接入新版整图。
- 以新版原生单 token 打包为参考，核对 5 个 anchor 和 1、2、8、16 行输入的真实页表、mHC embedding、Engram 及 C2A 发布位置。20 组输入逐行相同，覆盖乱序物理页和跨页边界。

#### 2d72687 与拒绝后直接保留前缀

在同事的 `2d72687` 上重新编译通过 Prefill 32 和 Decode 16，两图启用通信窗口预分配并设上下文容量为 512，编译分别用时 142.45 和 203.67 秒。111 上用真实 W8A8 checkpoint 通过新版 v9 布局 67 项投影、TP 分片及 Engram 检查。新版加载器统计的静态权重为每 rank 36.55 GiB。新版验证图把每个 C2A 源的 FP32 状态环从 4 行扩到 32 行。K8 或 K16 的验证即使跨环，也会把前一已提交位置到最后一个候选位置写到不同槽位。拒绝后保留下一步要读的前一个已提交行，裁剪 Engram 历史，下一轮按提交的位置覆盖被拒绝的分页数据。这样一次 Verify 就能提交已接受前缀，省掉状态备份和 Accept 的基模重放。状态环每 rank 额外占 336 KiB（3 个 C2A 源，每行 1024 个 FP32 元素）。

新版 `generate --orthrus-verify-k 8` 或 `16` 默认使用上述路径。追加 `--orthrus-cache-mode compare` 可在一次装载权重内运行 AR 参考、备份回放和直接保留两组，分别报告结果和耗时。报告分别记录 `verify_calls`、`replay_calls`、`retained_prefix_calls`、备份字节数、上传和图调用时间。用新版 C2A 的 Torch golden 对照了 1080 次状态推进，覆盖跨环、跨页、连续拒绝和全接受，保留前缀的压缩结果及下一步所需状态逐元素一致。167 上再用原生 C2A 的设备函数做了 28 次前缀推进，结果逐位相同。[上游 2d72687](https://github.com/Adelaide999/pypto-lib/tree/2d7268765e78447f7127682b19bc9c1bc4441530/models/deepseek_v4_1_flash_a3)

167 的 910B3 上，关闭详细运行时日志后，采用同一个驻留 worker 交替计时 25 次，K8/K16/K32 的滑窗解包加双向 Attention 分别从 36.50/36.59/37.37 ms 降到 18.50/18.57/18.70 ms。该时间包含宿主提交。融合与拆分的输出和 LSE 逐元素相同，与 FP32 golden 比较的最大输出误差为 3.40e-6。新版整模尚待实机验证端到端逐 token 一致性和性能。[单卡原始数据](validation/dsv41/pypto-1122-packed-attention-910b3.json)

BF16 draft QKV 复用新版 GEMM 和 RoPE。Q-A 和 KV 权重在装载时打包为连续列，用一个 GEMM 读入隐状态，RMSNorm 直接按融合输出的行跨度分别读取两段。输出为 Indexer 所需的归一化低秩 Q、旋转后的多头 Q，以及临时 latent KV。61 的单张 910B3 已对照合成输入和真实 W8A8 checkpoint 初始化的第 0 层 TP rank 0、第 13 层 TP rank 3。K8/K16/K32 均通过，真实权重用例的最大相对 L2 误差为 `2.41e-4`。可用 `validate_dsv41_pypto_packed_qkv.py --checkpoint <W8A8目录>` 验证由真实 checkpoint 初始化的单层权重。此入口独立于整模起草。 [实现](../src/blockspec/pypto_dsv41_draft.py) · [实卡结果](validation/dsv41/pypto-1122-packed-qkv-910b3.json)

压缩 Attention 接收设备 Indexer 的每行物理索引、有效长度和掩码。一个 PyPTO 提交内完成 FP8 滑窗、整个临时双向块和 FP4 压缩历史的 Attention。每个被选中的 FP4 历史块解包后由该 query 的 16 个头共用，压缩段与滑窗结果按 LSE 合并，sink 计入一次。61 单张 910B3 覆盖 K8/K16/K32、乱序页表及每行 0 到 512 个选择，FP32 参考最大输出误差为 `3.94e-6`，LSE 最大误差为 `9.54e-7`，缓存和选择元数据逐字节一致。验证入口为 `validate_dsv41_pypto_packed_history.py --compressed --pages 16`。[实卡结果](validation/dsv41/pypto-1122-packed-compressed-910b3.json)

新版 Indexer 使用 draft 的 BF16 投影，经原生 FP4 量化、分页打分和 Top-K 选择，输出物理行号、有效长度与掩码。已提交的 C2A 位置数为 `anchor // 2`，每页仍有 128 个压缩行。C1A 的 2048 个候选块覆盖本版本最大 16K 上下文，因此 Full 和 Reindex 都可以直接打分已提交前缀。历史不超过 512 行时直接枚举物理行。融合入口保留设备上的选择结果，供后续 Reuse 使用。

61 的单张 910B3 在 K8、K16、K32 的 15 组输入上验证了完整 Top-K 集合、输出和 LSE。融合与拆分提交的结果逐位相同，目标索引缓存逐字节一致。相对 FP64 参考的最大绝对误差为 `1.32e-5`，LSE 为 `9.54e-7`。同一 worker 交替计时，15 组的调用中位数从 36.45 ms 降到 18.60 ms，包含宿主提交。可用 `validate_dsv41_pypto_packed_indexer.py --fused-attention` 复验。另以 K8 覆盖 8K 和 16K 的 C1A/C2A 历史，四组 Top-K 与 Torch 一致，融合与拆分输出逐位相同。扩展用例的最大输出误差为 `1.64e-6`，可追加 `--extended-context --blocks 8` 复验。

新版起草 Attention 子层均从 mHC 输入归一化执行到原生 NZ 布局 Wo，输出本地 TP rank 的贡献。SWA 层使用滑窗和临时双向块。Full/Reindex 层写出每个 query 的历史选择，Reuse 层直接读取同一份驻留选择，使用本层的 Q、临时 KV 和 Wo 重新计算 Attention。三种路径都以只读方式访问目标缓存。

单卡复验使用真实 W8A8 checkpoint 初始化 QKV/Wo，配合构造的分页历史。验证入口 `validate_dsv41_pypto_packed_layer.py --checkpoint <W8A8目录>` 覆盖 K8、K16、K32 和两种压缩率。相同输入下，带 Indexer 与复用选择的输出逐字节一致。更换复用层的隐状态后，输出与 Torch 参考一致。目标 SWA、压缩历史和 Indexer 缓存逐字节一致。[单卡原始数据](validation/dsv41/pypto-1122-packed-draft-layer-910b3.json)

BF16 子层输出用 FP32 Torch golden 逐元素比对，保留模型约定的中间舍入边界。每组按 CANN OpenBase BF16 规则检查 `atol = rtol = 1/128`、匹配率至少 99%、最坏点绝对误差不超过 `max(0.1, 32 × ULP)`，同时检查有限值和原有相对 L2 门槛。报告另列上游单测 Wo 投影的严格容差 `0.003 + |golden|/128` 下的超差点。该上游单测给两边相同的 Attention context，融合子层的比对还包含前置阶段的累计舍入。整模验收需要逐 token 核对验证输出及目标 KV 提交。[CANN OpenBase 精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) · [上游 Wo 验收](https://github.com/Adelaide999/pypto-lib/blob/2d7268765e78447f7127682b19bc9c1bc4441530/models/deepseek_v4_1_flash_a3/o_proj.py#L785)

第 0 层 rank 0 和第 13 层 rank 2 的 18 组输出均通过 CANN BF16 门槛，每组逐元素匹配率均为 100%，对 FP32 golden 的最大绝对误差为 `0.0099051`。对收窄到 BF16 的 golden，最大相对 L2 误差为 0.203%，用于描述这批张量的整体偏差。把上游 Wo 单测容差应用到完整子层时，第 0 层有 66 个元素超差，单组最大比例为 0.054%。第 0 层单独执行 Wo 在 K8、K16、K32 上超差数均为零。[精度逐项记录](validation/dsv41/pypto-1122-packed-draft-layer-910b3.json)

Wo 投影之后复用原生 TP4 归约和 mHC Post，输出给 MoE 的隐状态、预混合系数和折叠后的 FFN 输入。用真实 checkpoint 的 mHC 权重和四张 910B3 测试 K8、K16、K32，三个输出在四个 rank 间逐字节一致，最大相对 L2 为 `7.40e-5`。测试用各 rank 独立的 BF16 投影贡献，并在驻留窗口重复运行。真实 W8A8 初始化的 SWA 投影也与 TP4/mHC 组成同一张图，在 K8、K16、K32 四卡实测通过。12 份 rank 投影输出均通过 CANN BF16 门槛，回写给 MoE 的三个张量在四卡间逐字节一致。运行时补丁把编译后的 NZ 完整切片映射回原有驻留分片，不复制权重。入口为 `validate_dsv41_pypto_packed_draft_tp.py --checkpoint <W8A8目录> --devices <四个设备编号>`，加 `--swa` 复验同图的 SWA 路径。[TP4 验证记录](validation/dsv41/pypto-1122-packed-draft-tp-910b3.json)

#### 40 层驻留起草图

镜像固定上游 `949a18d`。该版本加入多请求 packed 会话，并对 Decode 每个有效位置保持单 token SWA 输出投影的 FP32 累加顺序。基于该提交在 PTOAS 0.78 上编译了上下文容量 512 的 packed Prefill 32、Decode 16 和 40 层 K32 Draft，分别用时 110.50、165.00 和 56.98 秒。Prefill 和 Decode 的编译用例每个 DP 预留两个请求槽，起草会话使用每个 DP 一个请求槽。[上游源码](https://github.com/Adelaide999/pypto-lib/tree/949a18db40fca66f677fb041e98b32f824e7afb1/models/deepseek_v4_1_flash_a3)

新 packed 会话可把四个 DP 的活动请求合成一次起草和一次验证。它为每条请求绑定自己的页表和 Engram 历史，并在验证拒绝后只提交已接受前缀。纯 AR 和普通验证都从设备回读贪心 ID。启用教师反馈时，仅为选中的请求回读已接受前缀和首个修正位置所对应的词表行。主机测试覆盖跨页拒绝、请求隔离、教师切片，以及 TP 组切换时的设备采样回读。这些入口仍使用原生整图的单机 L3 worker，双机 EP16 传输和新版在线反向及权重发布待接通。

起草图在同一张图中执行 40 层 Attention、TP4 和冻结的原生 MoE，并在第 1、14 层复用原生 Engram，最后复用 TP 词表层。图内使用 K32 原始 head 契约。目标验证窗口可以设为 K8 或 K16，按从 anchor 开始的连续源行取前 7 或 15 个候选。独立编译入口也支持 K16 起草形状，供后续短块 head 验证。

Serving 在启动时装载 head 并在 TP 间分片。Draft 直接借用目标设备上的六个滑窗、压缩和索引缓存缓冲，Engram 使用 anchor 与 MASK 的临时历史，退出准备函数即恢复原有历史。目标模型沿当前已提交前缀验证。比较命令先运行同输入的 AR，再记录候选来源、起草准备、上传、图执行、token ID 回读以及全轮接受结果。

在 1122 镜像内编译 K32 图，`--context` 与目标 Prefill/Decode 使用的历史容量一致。生成时将以下参数加到原生 `models.deepseek_v4_1_flash_a3.generate` 的模型、请求和设备参数中。

```bash
python /opt/dsv41-1122/compile_dsv41_pypto_packed_draft.py \
  --block 32 --context 512 --decode-capacity 16 --artifact-dir /results/graphs/draft

# 原生 generate 的附加参数
# --orthrus-draft-head <BF16 head 目录> --orthrus-draft-runtime-dir /results/graphs/draft
# --orthrus-verify-k 16 --orthrus-cache-mode retain --audit-caches
```

`--audit-caches` 在 Verify 提交后额外检查 TP 副本与非活跃 DP 的缓存，吞吐计时应单独运行。当前证据覆盖编译、产物重载、驻留参数 ABI 和主机历史管理。整模 NPU 执行、双机 EP16 以及该图的 last-N 在线更新需要继续验收。[整图验证记录](validation/dsv41/pypto-1122-packed-draft-graph.json)

## 内网权重预检

默认权重目录为 `/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8`，对应同事文档指向的公开版本。脚本优先使用 `--model` 指定的目录，其次读取环境变量 `TARGET_WEIGHTS`，最后使用默认目录。在任意 Python 3.10+ 环境下执行下列检查，只读取本地配置、索引和实际 safetensors header，无需安装 PyPTO，也无需 NPU。报告列出当前 A3 loader 所需的投影和 Engram 格式，以及每个不匹配的实际张量。此命令收集布局，成功退出表示报告生成完成。

```bash
python3 "$ORTHRUS_ROOT/online-speculation/scripts/preflight_dsv41_pypto.py" \
  --inspect --output checkpoint-layout.json
```

带 `1017` 的公开 W8A8 包把 `W8A8_DYNAMIC` 声明放在 `quant_model_description.json`。预检接收该声明。实读公开文件的结果如下。 [实际张量头记录](validation/dsv41/public-w8a8-header-layout.json)

| 项目 | 公开 1017 W8A8 权重 | 当前 A3 PyPTO 路径 |
| --- | --- | --- |
| Q-A、KV 投影 | BF16，无 INT8 scale/offset | INT8 GEMM，读取 scale/offset |
| Q-B 投影 | INT8 | INT8 |
| Engram 表 | INT8 | INT8 |
| Engram scale | F8_E8M0 | FP32 |
| 量化声明 | description 文件 | Serving loader 从 config 读取 |

带 `1017` 的公开权重需要在加载器中识别量化声明和混合精度，并在 Q-A/KV 计算处选择 BF16 投影。CPU Engram 查表需要按 E8M0 格式读取并解码命中的 scale 行，同时保留原导出已有的 FP32 路径。只修改 config 无法改变这些张量格式。现有 A3 适配与同事提供的模型路径配对；不带 `1017` 的公开包已抽查到 INT8 Q-A/KV 和 FP32 Engram scale，与该加载布局相符。完整权重校验和整模验收仍须执行。

把两份补丁应用到配套 PyPTO Lib 和 Serving 后，可在其 Python 环境下执行完整预检。若有 head，追加 `--head "$DRAFT_HEAD"`，验证它与目标权重的配置、索引和 tokenizer 哈希。两台使用本地盘时，对照报告的 `identity` 可以检查这些元数据是否相同，权重 payload 的完整性另行校验。

```bash
export TARGET_WEIGHTS="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
PYTHONPATH="$ORTHRUS_ROOT/online-speculation/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$ORTHRUS_ROOT/online-speculation/scripts/preflight_dsv41_pypto.py" --model "$TARGET_WEIGHTS"
```

当前 A3 服务入口使用单机设备上下文。双机 rank 布局、调度和 HCCL 通信可用独立探针验证，整模仍需接入按节点划分的权重和 KV 分配。 [双机调度与通信](ORTHRUS_DSV41_PYPTO_CLUSTER.md)


## K32 SWA

PyPTO 程序接收 RoPE 后的 32 行 query、128 行分页滑窗的只读视图、32 行临时 latent KV，以及每头的 Attention sink。输入页号经过宿主侧的 `swa_spans` 转为最多两段物理行。设备只搬读对应的历史行。所有 draft 位置读取同一 anchor 历史和整块临时 KV。两部分分数和 sink 使用同一个 FP32 softmax，输出 FP32 Attention 结果和 LSE。

验收输入包括历史为空、滑窗刚满、跨乱序物理页和长前缀。所有不可见缓存行用 NaN 填充，设备输出与现有 Torch FP32 golden 比较，并逐字节复读目标缓存、元数据和临时块。默认用 8 个本地 query head、512 维 latent，与实模 TP8 的单 rank 一致。 [PyPTO 源码](../src/blockspec/pypto_dsv41_swa.py) · [Torch golden](../src/blockspec/vllm_ascend/dsv41_attention.py)

910B3、25.5.1 驱动和 CANN 9.0.0 上复现了旧版融合 SWA 的间歇性输出错误。跨页时 SWA 输出已出现偏差，LSE 仍通过参考对照。单独拆开 PV 矩阵乘仍能复现。服务现在将 QK、softmax 和 PV 分成三个设备任务，用设备缓冲区交接中间结果。8 头原始布局和 16 头 token-major 布局各通过六组分页和压缩历史用例，每组重复 30 次，逐轮对照 FP32 参考。容器内的复验命令见[环境说明](ORTHRUS_DSV41_PYPTO_ENVIRONMENT.md)。

TP4 的 K32 压缩历史路径默认将 SWA 与压缩 Attention 放进同一个 PyPTO 调用，设备侧分阶段执行。单张 910B3 使用同一常驻 worker，5 组分页用例每组交替计时 50 次，单层调用中位数从 1.97–2.14 ms 降至 1.24–1.30 ms，输出和 LSE 逐字节相同。用真实 W8A8 权重初始化 draft head 后，单 rank 的 40 层双请求复验将调用从 333 次降到 257 次。三组交替计时的中位数从 956 ms 降到 853 ms，含测试脚本的结果读回和对照。输出、缓存和交替请求重放逐字节一致。复验入口为 `validate_dsv41_pypto_resident_draft.py --compare-joined-attention --replay`。 [单层对照记录](validation/dsv41/pypto-joined-attention-910b3.json)

Draft 投影和 Attention 复用会被完整覆写的输出缓冲区，EP 反向的打包和解包由算子初始化无效行。使用配套 Serving 补丁的 `workspace(fill=None)` 路径，单张 910B3 上 2049-token 历史、同一 DP 的 1、2、4 个 K32 请求，40 层检查用时分别从 501、910、1610 ms 降到 422、703、1248 ms。输出、缓存和交替重放逐字节一致。可用 `validate_dsv41_pypto_resident_draft.py --compare-scratch-fill --anchors 2049 2049 --replay` 比较缓冲区清零开销。

SWA 历史 mask 在加载 head 时一次性驻留，页段坐标在同一轮各层间复用。完整块的 indexer 直接覆写中间缓冲区，K8 对齐产生的无效行继续初始化。单张 910B3 上分别做同输入五轮交替对照。SWA 控制复用在两个请求（31、2049 行历史）下把 40 层检查中位数从 822 ms 降到 788 ms；indexer 在四个请求（31、2049、8193、16351 行历史）下从 1353 ms 降到 1298 ms，每轮少清零约 82 MB。两项检查的输出、历史缓存和交替重放均逐字节一致。对应开关为 `--compare-swa-controls` 和 `--compare-index-scratch`。

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

当前 A3 上下文上限为 16384 个 token。C1A 历史最多涉及 2048 个 8 行候选块，与 checkpoint 的候选池容量一致，所以 Reindex 可以对全部可见历史重新打分。设备入口会拒绝候选池需要裁剪的配置。压缩历史消费层在同一请求、同一 rank 上复用来源层的 Top-K。

在单层验证命令上加 `--indexer --anchor 1537` 即可验证第 2 层自行选行，`--layer 20 --ratio 1` 和 `--layer 24 --ratio 1` 分别用于 C1A Full 和 Reindex。固定工具链下，单张 910B3 跑过 7 组 TP4 单 rank 用例，覆盖空历史、短历史、奇数 anchor、跨页和 512/1024/2048 行桶。224 个查询的选中集合与 A3 QLI 参考一致，压缩 Attention 最大绝对误差为 `1.17e-5`，历史页逐字节一致。测试使用 BF16 初始化 head 和构造的分页历史。 [Indexer 源码](../src/blockspec/pypto_dsv41_indexer.py) · [实测记录](validation/dsv41/pypto-draft-indexer-tp4-910b3.json)

## K32 验证和提交

`A3Runner.verify_draft(batch, candidates)` 接收请求账本已分配页的 anchor 和最多 31 个候选。它用每 DP 128 行的原生因果 Prefill 程序验证整块，读取每个候选位置对应的 logits。验证期在设备端保存涉及的目标页，完成后原地恢复。CPU 上的 Engram 请求历史和已提交长度也恢复。runner 再用同一个主模型程序执行 anchor 和接受前缀，将其 KV 正式写入。目标补发的 token 留给下一轮处理。这个路径会重跑接受前缀，用于先验证缓存和输出语义，整模吞吐需要单独测。

CPU 测试覆盖完整 K32、部分接受、零接受、跨 128 行分页以及多个 DP 的请求顺序。单张 910B3 验证了设备范围快照在正常返回和异常退出时均可逐字节恢复。公司内网 8+8 卡进一步完成了真实 W8A8 整模 K32 验证、接受前缀提交和 `100663296` 字节缓存恢复。 [双机整模记录](validation/dsv41/pypto-cluster-company-ep16-acceptance.json)

### 短块验证

Draft 测试入口支持 `--block-size 8|16|32`。K8 的投影和 Attention 查询使用 8 个位置，临时 KV 用 16 行存储以满足 Cube 对齐，softmax 屏蔽后 8 列。Indexer 沿用 16 行规格，只检索 8 个有效查询。在单张 910B3 上，K8 的六组 SWA 和压缩 Attention 用例通过 FP32 参考对照，并使用非零填充验证了屏蔽逻辑。同一张卡、同一初始化 head 分片和两个请求（历史长度 31、2049）下，40 层 draft 验证脚本的单轮中位数为 K8 763 ms、K16 693 ms、K32 725 ms，含结果读回与检查。三种块宽的缓存和交替请求重放均逐字节一致。

K8 也在两张 910B3 上验证了两个 TP4 分片，三轮共执行 120 次归约，缓存和交替请求重放逐字节一致。`compile_dsv41_pypto_cluster.py --shape short --pto-only` 验证了基模 8 个有效位置、16 行容量的 64 个编译入口。双机 Serving 和在线更新目前使用 K32。短块的整模验证和吞吐需要在接通短块调度后测量。

## Serving 入口与实机验收

原生 resident session 中，draft 复用基模的 Embedding、CPU Engram 查表、mHC、MoE 和 LM Head。每层的 Attention 由 diffusion 参数计算，Wo-B 使用 TP4 分片，输出贡献经原生 TP 归约后再进入共享 MoE。临时块和 Top-K 保存在设备上，全部 head 参数只在初始化时上传。请求接近上下文上限或需要额外采样约束时，runner 走普通 Decode。

固定工具链下，单张 910B3 依次执行了 40 层的一个 TP4 rank，覆盖 SWA、C2A、C1A Full、Reindex 和跨层复用；校验了非零块偏移和 TP 归约前的打包、目标缓存逐字节一致以及 head 权重零重复上传。另测了 4096、8192、16384 行桶，96 个 query 的 Top-512 集合与参考一致，压缩 Attention 最大绝对误差为 `7.63e-6`。该测试使用初始化 head 和构造的分页历史。 [实机结果](validation/dsv41/pypto-resident-draft-tp4-910b3.json) · [运行入口](../scripts/validate_dsv41_pypto_resident_draft.py)

分阶段 SWA 在 25.5.1 驱动、CANN 9.0.0 上复跑了相同的 40 层 TP4 单 rank 检查。真实初始化 head 常驻约 3.17 GB，执行 167 次设备调度，缓存复读一致，运行期间没有重新上传权重。

### 单机多卡验证

在配套 PyPTO 环境中，从 `online-speculation` 包目录运行以下脚本。每个进程只暴露所申请的物理卡，脚本内使用从 0 开始的逻辑编号。

```bash
PYTHONPATH=src python scripts/validate_dsv41_pypto_multicard.py \
  --lib "$PYPTO_LIB_ROOT" --cards 4 --tp 2 --output transport.json

PYTHONPATH=src python scripts/validate_dsv41_pypto_resident_draft.py \
  --lib "$PYPTO_LIB_ROOT" --model "$TARGET_WEIGHTS" --head "$DRAFT_HEAD" \
  --cards 4 --requests 2 --output resident-tp4
```

第一项使用小张量检查原生 TP、DP、EP 通信和各 rank 的跨页还原。第二项加载真实 head 的四个 TP4 分片，检查各层请求缓存、逐层 pack 和 BF16 ReduceScatter。两卡或三卡可用 `--cards 2` 或 `--cards 3` 检查对应分片，每层对参与 rank 的贡献执行 AllReduce。实际测试已通过两卡和三卡通信、回滚及 40 层分片执行，三卡另通过了两个不同历史长度的请求共享一次调度。测试中的历史页和隐状态由脚本构造。

在单机四张 910B3 上，用公开 W8A8 checkpoint 导出的 BF16 初始化 head 完成了两个请求的 40 层 TP4 测试。所有 rank 的目标缓存逐字节一致，40 次层间 ReduceScatter 均通过 BF16 误差检查，head 只在启动时上传。单 rank 的 Reindex 另测到 anchor 16351，32 个查询的 Top-512 与参考一致。 [四卡实测记录](validation/dsv41/pypto-resident-w8a8-tp4.json)

在上述 draft 脚本加 `--replay` 会交替执行两组请求，复用工作区后再重放第一组。使用同一份公开 W8A8 导出的 head 在单卡完成了三轮、每轮 40 层。第一组两次输出逐字节一致，缓存复读一致，期间没有重复上传 head。 [重放记录](validation/dsv41/pypto-resident-w8a8-replay.json)

另使用公开 W8A8 的第 0 层真实 INT8 QKV 权重，在一个 TP4 分片上比较原生 Prefill K32 和逐行 Decode。构造的 32 行输入中，两条路径输出的 Q latent、带 RoPE 的 Q 和滑窗 KV 均逐字节一致。此检查覆盖投影子路径。 [基模 QKV 记录](validation/dsv41/pypto-target-w8a8-qkv-k32.json)

在 Serving 所用的 Python 环境安装本仓库的 `online-speculation` 包，或在已准备好依赖的环境中设置 `PYTHONPATH="$ORTHRUS_ROOT/online-speculation/src${PYTHONPATH:+:$PYTHONPATH}"`。申请一个完整的 A3 16-rank 作业，在同一设备映射和相同提示下分别运行 target-only 和 Orthrus。Engram 仍按配对 Serving 的 HOST 路径分配。

```bash
export PYPTO_LIB_ROOT
export TARGET_WEIGHTS="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
orthrus_config=$(python3 -c 'import json, sys; print(json.dumps(dict(method="orthrus", model=sys.argv[1], num_speculative_tokens=31)))' "$DRAFT_HEAD")
common=(
  --model "$TARGET_WEIGHTS" --devices "$TASK_DEVICE" --platform a2a3
  --dp 4 --tp 4 --ep 16 --max-num-seqs 4 --max-model-len 512
  --max-num-batched-tokens 128 --block-size 128
  --long-prefill-token-threshold 16 --enable-chunked-prefill --no-enable-prefix-caching
  --prompt 'Continue the list of positive integers from 1.'
  --generate-config '{"max_new_tokens":64,"temperature":0,"ignore_eos":true}'
)
mkdir -p "$RESULTS"/{target-native,orthrus-native}
TMPDIR="$RESULTS/target-native" python -m pypto_serving_a3.offline "${common[@]}" --output "$RESULTS/target"
TMPDIR="$RESULTS/orthrus-native" python -m pypto_serving_a3.offline "${common[@]}" --output "$RESULTS/orthrus" \
  --speculative-config "$orthrus_config"
python "$ORTHRUS_ROOT/online-speculation/scripts/verify_dsv41_pypto_acceptance.py" "$RESULTS"
```

`RESULTS` 使用新的绝对路径。验收脚本同时对照生成 token 和原生 dispatch 记录，要求至少一轮起草、验证、提交实际完成，核对轮内请求归属、epoch、KV 快照恢复字节数和退出清理。报告给出首个分叉 token 的下标，全部走普通解码也会判为失败。

`TARGET_WEIGHTS` 使用配套 A3 W8A8 导出，`DRAFT_HEAD` 使用与目标模型配置及权重索引哈希匹配的 BF16 head。预期公开模型为 `Eco-Tech/DeepSeek-V4.1-Flash-w8a8`。带 `1017` 的包将部分 Q/KV 投影保留为浮点，Engram scale 为 E8M0；此 Lib 的基模加载器要求相应投影使用带 scale/offset 的 INT8，并读取 FP32 Engram scale。配套权重已通过双机 EP16 的逐 token 一致性、K32 验证、接受与拒绝回滚；并发、长序列和训练 head 的性能仍需单独验收。 [公开 checkpoint](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8) · [双机整模记录](validation/dsv41/pypto-cluster-company-ep16-acceptance.json)
