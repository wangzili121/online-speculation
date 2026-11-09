# Orthrus-Qwen3.8 PyPTO Serving 迁移

本分支基于 OneServing 的 Qwen3.8 独立交付包接入 Orthrus。原生模型保持不变，推理读取原有训练路径导出的 draft head。另提供 PyPTO last-1 在线更新入口，完整离线训练与其他 last-N 配置继续使用现有 PyTorch 路径。

Orthrus-Qwen3.8-27B 已在单卡 A2 上完成真实 head 的 K32 前向、目标缓存只读、拒绝恢复、greedy 对照及 HTTP 检查。测试覆盖完整 64 层，主模型与 draft 在同一 worker 中共享 backbone。参数转换在启动阶段完成，推理复用已分配的缓冲。**目前为 TP1、一个活动请求、512 个总位置；起草阶段计时与服务吞吐分别验收。**

## PyPTO last-1 在线更新

`--online` 使用显式前向和手写反向更新最后一层 Full Attention 的 Q/K/V/O 投影及 Q/K norm，主模型参数始终冻结。反向仍经过该层共享 MLP、norm 和 LM head，以计算 draft 参数所需梯度；在第 63 层输入处停止，不回传前面 63 层或目标模型历史。

采集轮次保留末层输入、anchor 之前的私有历史副本及 verifier 教师 logits，只用真正验证到的行计算 TV loss。多个 replay 块统一按有效位置总数归一化。FP32 master、梯度、Adam 状态、裁剪和执行参数转换均在 NPU 上完成；Torch autograd 只用于独立验证。更新先写候选缓冲，有限值检查通过后，才在两轮起草之间发布新版本。主模型 cache 不参与参数发布。

64 GB A2 的完整主模型占用较高，在线更新使用第二张 NPU：推理卡继续运行共享主干的 target/draft，学习卡只加载最后一层所需的冻结 MLP、norm、LM head 及学习器状态。两边通过有界主机缓冲搬运字节；反向与优化器计算在学习卡执行。更新后的六组 head 参数复制到推理卡的备用缓冲，全部就绪后切换引用。这是两个独立 worker，模型仍为 TP1。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python scripts/serve_qwen38_pypto.py \
  --source /path/to/OneServing --model /models/Qwen3.8-27B \
  --head-dir /models/orthrus-pypto-head \
  --learner-head-dir /models/orthrus-raw-head --device 0 \
  --attention-backend paged --gdn-backend vector \
  --draft-group-size 4 --head-transfer packed \
  --online --learner-device 1 --last-layers 1 --update-stride 16 --replay-blocks 2 \
  --learning-rate 1e-5 --checkpoint-dir /data/online-heads
```

首次初始化需要原训练 head 的 BF16 norm；已转换 bank 中的 `1 + weight` 不能保证精确还原原值。`--head-dir` 直接使用原训练 head 时，可以省略 `--learner-head-dir`。每次更新可保存 `step-00000001` 等目录，包含六组参数的 FP32 master、Adam moment、步数和基模/head 哈希；重启用 `--resume-online /data/online-heads/step-00000001` 恢复，不需要重新初始化 norm。此 checkpoint 是 last-1 续训状态，不是原有完整 head 目录的替代格式。

`--device` 与 `--learner-device` 使用可见设备序号。设备内存足够时可省略独立学习卡；当前 64 GB A2 的单卡整模型在线配置不能容纳全部状态。末层反向组合为一个 CHIP 入口，内部仍有多个计算阶段；梯度直接累加到候选 master 缓冲，提交后旧 master 轮换为下一次梯度缓冲，减少约 800 MiB 重复存储，保留两套执行参数用于安全切换。

若仅需固定 head 推理，可以将在线 checkpoint 导出为标准 PyPTO head；`--head-dir` 仍指向初始化时使用的原 head，不能换成其他版本：

```bash
python scripts/prepare_qwen38_pypto_head.py \
  --model /models/Qwen3.8-27B --head-dir /models/orthrus-pypto-head \
  --online-checkpoint /data/online-heads/step-00000001 \
  --device 0 --output-dir /models/orthrus-updated-pypto-head
```

该步骤在 NPU 上从 FP32 master 生成执行参数，只替换输出文件的最后一层，保留原 head 文件与冻结层。导出的目录可用于固定 head 推理；继续在线训练时仍使用原 head 与对应 optimizer checkpoint。

代码入口为 `online.py`（反馈与重放）、`training_graph.py`（显式计算图）、`training_*_programs.py`（设备前向/反向）、`online_optimizer.py`（裁剪和发布）、`online_bridge.py`（跨 worker 搬运与切换）、`online_checkpoint.py`（文件保存/恢复）。不更新 GDN，也不将该入口当作通用 last-N 或全层训练。

独立检查使用 `scripts/validate_qwen38_pypto_training.py` 的 `--suffix` / `--optimizer`；真实模型闭环使用 `scripts/validate_qwen38_pypto_online.py`。当前独立手写梯度与优化器检查已通过，真实模型在线闭环仍在联调。

## 固定基线

| 组件 | 固定版本 |
| --- | --- |
| OneServing | `7aa40ba5238e3a83bb5f25e18817497c181e3e9b`，子目录 `pypto-serving-qwen3.8-27B` |
| PyPTO | `98752784548b777b2377a9084411c3716eb341d6` |
| Simpler | `c1f89ea431f5306f78c0e4ab9573c829756d7f1e`，由 PyPTO gitlink 固定 |
| PTO-ISA | `418de18dde8fa33773dc5c384fdfe5506463b628`，由 Simpler 固定 |
| PTOAS | `v0.65`，wheel 哈希见 PyPTO `toolchain/versions.env` |

参考 [OneServing 交付说明](https://github.com/ndleslx/OneServing/tree/7aa40ba5238e3a83bb5f25e18817497c181e3e9b/pypto-serving-qwen3.8-27B) 和 [PyPTO 官方容器说明](https://github.com/hw-native-sys/pypto/blob/98752784548b777b2377a9084411c3716eb341d6/docs/en/user/04-dev-container.md)。已有容器可以作为构建底座，但其内置 PyPTO/Simpler/PTOAS 不一定符合交付包，必须逐项核对，不关闭 ABI/pin 检查。

这份基线的 TP2 仅分片 LM head，其他权重仍在各 rank 上复制，不能据此估算完整 backbone 分片后的显存。原生上限为 512 个位置、greedy；其交付目录的短 smoke 不能替代完整数值和缓存验收。

## 代码与接口

代码位于 `src/blockspec/pypto_qwen38/`：

- `baseline.py`：检查 OneServing 与运行时版本，在独立模块命名空间导入原生代码。
- `head.py`：读取 `orthrus-qwen38-native-tp-v1`，检查基模哈希、K、mask、混合层策略和 tensor schema，按运行 rank 读取 TP1/2/4 checkpoint 的对应参数，并准备共享主机布局。当前服务运行 TP1，会合并训练分片。
- `loading.py` / `loading_programs.py`：加载阶段的 PyPTO 转换、有限值和副本检查，以及纯数据搬运；文件读取与形状、偏移等元数据继续由主机提供。
- `head_loading.py` / `base_loading.py`：将训练 head 的各 rank 区间合并为推理 bank，并按层加载主模型到最终设备位置，避免 CPU 上的矩阵拼接和数值变换。
- `prepared_head.py`：保存已转换的 head；用基模、TP、K、mask、schema 和文件哈希约束缓存，启动时直接读取字节和建立 bank 视图。
- `tp.py`：定义各 rank 的 heads 范围和 KV/GDN 状态形状，按参数布局重分片，检查目标历史的形状是否相容。
- `weights.py`：设备内存充足时保留全部 head；否则逐层或按四层复用暂存缓冲。打包上传通过公开的 `Buffer.tensor` 创建具有同一内存 owner 的参数视图，同组各层参数同时有效。
- `tp_programs.py`：组合原生 Linear、residual 和 PyPTO 内置 Sum，提供行并行投影、整块归约及归约后的 residual 组件；完整 TP 模型执行器尚未接入。
- `greedy.py`：比较整块候选；第一次拒绝时输出目标 token，全接受时输出 bonus；输出上限和停止 token 同样限制提交长度。
- `target.py`：在原生 runner 上执行验证；需要拒绝后缀时，恢复验证前的 GDN 状态和页表长度，再仅重放接受前缀。正常事务使用设备快照，每个存活快照独占一个槽；嵌套快照超出设备池时保留主机副本回退。
- `programs.py` / `group_programs.py`：组合原生投影、norm、GDN、Full Attention 和 MLP，可按层或四层提交。原生对照保留 16 行 tile；优化路径以 K32 准备和投影，Full Attention 先准备整个块的 K/V，分页核心使用 32 行 query，读取主模型历史但不写目标缓存。
- `attention.py` / `attention_programs.py` / `paged_attention_programs.py`：独立 Torch 参考、融合/分阶段对照及直接分页 Attention；接收原生完整 KV pool 和层号，读取真实历史与临时块，输出门控后的 Attention。
- `gdn_programs.py` / `gdn_convolution_programs.py` / `gdn_aux_programs.py`：Vector 递推、只读 K32 卷积、gate 及 gated norm。`gdn_block_programs.py` 将这些阶段接到整层，直接读取 packed QKV 和目标历史，省去展开复制及两段之间的矩阵交接。
- `projection_programs.py` / `shared_programs.py`：为 draft 提供 K32 投影、RMSNorm 和共享 MLP，减少分成两段时重复读取权重；共享参数仍直接引用主模型 bank。
- `serving.py` / `serving_programs.py`：设备端准备 draft 与目标模型的 RoPE/mask，在设备上检查 logits 并选 token，以及在设备内保存/恢复 GDN 状态；磁盘读取、页分配和请求调度仍由主机负责。
- `draft.py`：将 draft 与目标程序提前注册到同一 worker，共享 embedding、norm、MLP 和 LM head，组织各层计算与输出读取。

Q/K/V-packed 的 GDN 投影与卷积按 Q、K、V 三段分别合并；O projection 按列合并；复制 norm 必须在各 rank 一致。真实 vLLM head 的 `A_log` 是 FP32，转换时保持原精度；decay/beta 编排使用相同公式，去掉原生辅助函数对该参数的重复转换。其他参数按实际 dtype 校验。

Full Attention 的整个临时块双向可见；GDN 延续 head 契约的 `orthrus_qwen35_reference` 因果递推，在目标状态的临时副本上计算。执行分块不改变模型公式，主模型继续使用固定 OneServing 的原生程序。

初版事务适配仅针对 TP1、一个活动请求。原生 scheduler 每次消费一个 token，适配器缓存已验证的输出前缀，逐个交付；后续交付不再次运行模型，且检查请求位置和 anchor，拒绝错位的旧缓冲。真实 draft 应提供 `propose(request, anchor, width)`，读取已提交历史，不写入目标缓存。

## 可选 Attention 后端

服务、编译审计、完整 draft 验证和 benchmark 均支持 `--attention-backend native|fused|paged`，默认 `native`。三种路径共享模型参数、历史边界及 GDN 策略。以下 Attention 后端对照使用 `native` GDN，Vector GDN 的结果另列于下一节。

| 后端 | Full Attention 执行方式 |
| --- | --- |
| `native` | 复用交付包的分阶段 Attention，先准备连续历史与临时块，再执行 QK、softmax、PV 和门控 |
| `fused` | 保留连续 KV 准备，将 QK、softmax、PV 和输出门控融合 |
| `paged` | 按页表直接搬历史与临时块到 L1，不生成连续 KV 副本；融合 Attention/门控，按长度选择 128/256/512 容量，使用 32 行 query tile |

`paged` 还将 Full Attention 的 QKV 准备与该层后续计算合为一次 HOST 提交，K32 的计算提交数由 82 降为 66。QKV、O projection 和共享 MLP 的公式与权重不变，draft 可使用 K32 执行分块；HOST 编排合并不表示这些计算融合成一个硬件 kernel。

分页核心保留原生中间舍入，独立检查与 Torch 参考通过。完整 draft 的 K32 归一化会改变部分浮点归约顺序，逐层采用固定相对 L2 门槛检查，不以 logits 逐字节相同作为替代实现的前提。目标 cache 必须保持不变，验证和恢复后的 greedy 续写必须与 target-only 相同；起草与服务性能分开验收。

独立输入、Torch 参考、计时范围与调优入口见 [kernel 任务说明](QWEN38_PYPTO_KERNEL_TASK.md)，本轮原始数据和复现方式见 [分页 Attention 验证](QWEN38_PYPTO_PAGED_VALIDATION.md)。原生/融合路径的逐层对照及 TP4→TP2 文件重载记录见 [逐层对照与重载验证](QWEN38_PYPTO_PAIRED_VALIDATION.md)。

## 可选 GDN 后端

`--gdn-backend native|vector` 独立控制 GDN，默认保留 `native`。`vector` 先用整块投影和归一化准备 K32，再从只读历史计算整块因果卷积，直接读取卷积后的 packed QKV，完成 Q/K 归一化和一次 K32 递推；最后执行 gated norm、K32 O projection 和共享 MLP。它不把 GDN 改为双向，不写回 AR 状态，也不依赖未来模型分支的接口。

这一条路径由多个设备阶段组成，HOST 合并提交不等于整层只有一个硬件 kernel。独立参考、计时范围及接回结果见 [GDN kernel](QWEN38_PYPTO_GDN_KERNEL_TASK.md)。与分页 Attention、四层编排一起选择时，启动参数为：

```bash
--attention-backend paged --gdn-backend vector \
--draft-group-size 4 --head-transfer packed
```

所有模型前向均在 PyPTO/NPU 中执行。默认还将输入张量准备、logits 有限值检查/greedy 选择、GDN 事务快照移至设备端。权重加载默认使用 PyPTO 做分片合并、转置、补零、norm 加一、类型转换及数值检查；Torch/safetensors 用于文件读取和主机缓冲描述，参数分配、上传和常驻由 PyPTO 运行时负责。

流式上传 head 时，应在参数分配前将进程绑定到所用 NPU 的本地 CPU。先从 `npu-smi info` 的 PCIe Bus-Id 找到 `/sys/bus/pci/devices/<BDF>/local_cpulist`，并确认 CPU 属于进程允许集合，再用 `taskset -c <CPU列表>` 启动原有命令。CPU 编号取决于服务器拓扑。实测同一 NPU、同一程序，仅将 head 主机内存由远端改为本地，完整起草由约 1.740 降至 1.094 s，logits 相同；记录见 [K32 与 NUMA 对照](validation/pypto-qwen38/k32-integration-20261109/)。

可选 `--draft-group-size 4 --attention-backend paged --head-transfer packed` 将 3 个 GDN 层和 1 个 Full Attention 层合并为一次 HOST 提交，并按组上传 head；K32 的计算提交由 66 次降为 18 次。模型计算不变，约增加 0.45 GiB 流式暂存。同 worker 六轮对照起草耗时为 2.600 / 2.315 s，logits 与候选逐字节相同；六个历史边界、八组 verifier 用例及直接启动服务检查均通过。实际配置和记录见 [执行与验证](QWEN38_PYPTO_PAGED_VALIDATION.md#四层合并执行)。

## 与 vLLM 路径的对应关系

两条路径共享算法定义，执行能力需分别核对，不能将 vLLM 的算子与校正实现直接算作 PyPTO 已有能力。

| 能力 | vLLM-Ascend | 当前 PyPTO 接入 |
| --- | --- | --- |
| 参数驻留 | TP2/TP4 分片后，固定推理的主模型和 draft 参数驻留 NPU | 单卡服务的主模型驻留、head 逐层上传；另已验证真实 head 的 TP2 分片分别驻留两卡。主模型与 head 一起驻留的整网多卡路径尚未接入 |
| Backbone/KV 共享 | 引用原生 backbone 权重，读取主模型历史快照 | 同一 worker 复用 backbone 设备参数和目标历史缓存；draft 使用临时工作区，不转正 draft KV。两边都有布局整理或临时复制，并非所有读取均为零拷贝 |
| Attention 执行 | 使用 `npu_fusion_attention` 等镜像已有融合算子 | 默认复用 OneServing 的分阶段 Attention/GDN；可选 `fused`/`paged`；直接分页实现完成独立算子、真实 27B K32 和六个历史边界检查，默认仍为原生路径 |
| 整块校正 | 使用 vLLM 原生 sampler，支持 greedy 与标准 rejection sampling | PyPTO 设备端检查 logits 并选择 token，仅回传 256 字节；接受前缀由 `greedy.correct` 的 Python 循环确定。目前仅支持 greedy，未实现随机采样 |

当前单卡最大的增量成本仍是 head 流式上传。接受前缀匹配、停止条件和页所有权只处理几十个整数，继续由主机组织；正常路径不再把完整词表搬回 CPU 扫描。审计脚本需要完整 logits 时，显式调用 `draft.read_logits(rows)`，读回和哈希不计入起草时间。

## 参数加载与设备端收尾

服务默认使用 `--serving-backend device`；`host` 保留迁移前的输入准备、输出选择和事务路径，便于同一 worker 对照。两种方式调用相同的主模型与 draft 前向，不改变 head、mask 可见关系或接受规则。

| 工作 | 执行位置 |
| --- | --- |
| 读取 safetensors、解析 manifest、校验文件 SHA256 | CPU 文件接口；当前固定版本没有相应的 PyPTO 计算 kernel |
| 合并训练分片、卷积转置、补零、norm 加一和类型转换 | PyPTO；数值有限性和跨 rank 复制参数的一致性同样在设备检查 |
| 拼接主模型各层的完整矩阵 | PyPTO 运行时直接 DMA 到最终 bank 的偏移，不先创建 CPU 拼接矩阵 |
| 分配 NPU 内存、上传及保留参数 | PyPTO 运行时，通过 CANN 完成设备分配与数据传输 |
| 主模型和 draft 的完整前向 | PyPTO kernel，继续共享 backbone 与目标历史 |
| draft MASK 输入、目标因果 mask、RoPE 查表 | PyPTO kernel；RoPE 表常驻 NPU，页号来自 CPU 分配器 |
| logits 有限值检查、greedy 选择 | PyPTO kernel；并列时选最小 token ID，单次仅回传 256 字节 |
| GDN 快照、恢复和单请求换槽 | PyPTO 设备内复制；两个快照槽额外占用约 310 MB NPU 内存 |
| 页分配/回收、接受前缀、停止条件、HTTP 调度 | CPU 控制逻辑，不扫描词表，也不执行模型前向 |

设备池的每个槽在对应快照释放前保持独占。超过两个同时存活的快照时，事务保留主机副本回退；正常单请求事务不需要回传 GDN 状态。独立检查入口为 `validate_qwen38_pypto_serving.py` 和 `validate_qwen38_pypto_device_state.py`；完整同 worker 对照使用：

```bash
python scripts/compare_qwen38_pypto_serving.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B \
  --head-dir /models/orthrus-head --device 0 \
  --output-dir /results/device-serving
```

该对照交错运行主机/设备两条收尾路径，比较完整 logits、候选、目标验证结果与历史，随后检查全部接受、各位置拒绝、停止/截断及槽位和物理页复用。独立检查还覆盖非有限值、并列最大值、有效行数、512 边界和含任意原始字节的真实大小 GDN 状态复制。

计时中 `compute_dispatches` 仅统计模型前向；设备输入准备、输出选择各另有一次提交。设备路径的 `select_seconds` 包含选择 kernel 和 256 字节回传，`readback_seconds` 只单列主机路径的完整 logits 回传，不能据其为零推断没有设备传输。

2026-11-09 的 A2 对照使用同一真实 27B、同一测试 head、分页 Attention、Vector GDN 和四层编排。主机/设备路径交错预热和计时，缓存哈希、完整 logits 读取及落盘均在计时外：

| 范围 | 主机收尾 | 设备收尾 |
| --- | --- | --- |
| draft 输入准备 | 4.48 ms | 1.40 ms |
| draft 输出回传与选择 | 11.45 ms | 1.26 ms |
| 整轮起草，包含流式上传 | 1.165 s | 1.167 s |
| 全接受验证事务，包含状态快照 | 0.319 s | 0.284 s |
| 首位拒绝事务，包含快照、恢复和前缀重放 | 0.446 s | 0.409 s |

起草辅助阶段显著缩短，整轮仍由约 1 秒的 head 上传主导，当前样本没有体现稳定的整轮提速。事务计时使用主模型生成的候选构造接受/拒绝控制组，不代表测试 head 的实际接受率。两边完整 logits、候选和 verifier 历史相同；八组事务、七个历史边界（含 476 位置）、物理页/槽位复用通过；HTTP completion、chat、SSE 和正常退出通过，另通过 400 项 CPU 回归。原始数据、独立核检查和源码哈希见 [设备端收尾验证](validation/pypto-qwen38/device-serving-20261109/)。

### 权重准备与复用

`--weight-backend pypto` 为默认值。读取原始 head 时，先在一个短期 PyPTO worker 中逐个转换参数；该 worker 退出后再创建模型 worker。单卡流式 head 仍需一份规范布局的共享主机缓冲，但其中的参数由 NPU 计算，不在每轮或模型启动时重新进行 CPU 张量转换。`--weight-backend host` 保留原始转换路径作为独立对照。

同一个 head 多次启动时，可以先保存转换结果：

```bash
python scripts/prepare_qwen38_pypto_head.py \
  --model /models/Qwen3.8-27B --head-dir /models/orthrus-training-head \
  --device 0 --output-dir /models/orthrus-pypto-head
```

之后将服务原有的 `--head-dir` 指向 `/models/orthrus-pypto-head`。格式为 `orthrus-qwen38-pypto-banks-v1`，记录原 head 契约、运行 TP/rank、PyPTO 版本和完整数据哈希。加载时只读取字节、校验文件和建立类型视图，不再进行权重数值运算；缺少完成后的 manifest、基模或 TP 不符、长度不符或数据损坏都会拒绝。更换训练 head 后重新生成一个目录并重启服务。

该预处理只在加载时执行。矩阵、转置、归约等计算采用 PyPTO 基础算子；磁盘读取、JSON 解析和 SHA256 继续使用主机接口，PyPTO 自身的程序和工具链校验也使用这类主机接口。

本轮用真实 27B 与 TP4 测试 head 在单卡 A2 复验：64 层、528 个 head bank 和主模型全部 23 个 bank 均与独立 CPU 加载结果逐字节一致；转换结果保存、重载后仍一致。新路径的完整 K32 logits 和候选与旧路径相同，目标缓存只读、接受前缀恢复、后续续写及 scheduler 输出通过；HTTP completion、chat、SSE 与正常退出也通过。另通过 11 项独立 NPU 检查和 430 项 CPU 回归，原始记录见 [权重加载验证](validation/pypto-qwen38/weight-loading-20261109/)。

首次 NPU 准备约 155 s（含编译），CPU 对照约 41 s，保存后的 head 重载约 32 s。迁移并未加快首次转换；多次使用同一 head 时，应保存转换结果后加载。该加载计时不代表每轮起草或服务吞吐。

## 完整 TP 的接入准备

PyPTO 本身已有多卡运行能力；当前固定 OneServing 只分片 LM head，Orthrus 服务仍使用 TP1。这里完成的是后续完整 TP 所需的 head 加载与布局准备，未启用多卡整网推理。`NeuralDraft` 会拒绝把局部 head 送入现有单卡图。

2026-11-06 检查公开 [OneServing main](https://github.com/ndleslx/OneServing/tree/main)，仍为固定的 `7aa40ba`。本轮继续验证双卡参数驻留与行并行组件，没有修改原生模型源码或放宽版本检查。

| 运行 TP | 每卡 Full Attention Q / KV heads | 每卡 GDN K / V heads | 每卡共享 MLP 中间宽度 |
| --- | --- | --- | --- |
| 1 | 24 / 4 | 16 / 48 | 17408 |
| 2 | 12 / 2 | 8 / 24 | 8704 |
| 4 | 6 / 1 | 4 / 12 | 4352 |

每卡仍处理完整的 K32 位置，按 heads 和通道切参数。head 加载使用与现有 vLLM checkpoint 一致的连续 heads 分片：Q/K/V 投影切输出行，O projection 切输入列；GDN 的 packed QKV 与卷积按 Q、K、V 各段分别切分，保持 K/V heads 的 1:3 分组；norm 复制并检查各训练 rank 一致。训练 TP 与运行 TP 可以不同，加载只读取当前运行 rank 所需的区间，不先拼出完整 head。GDN 门控矩阵保留原有 tile padding，局部有效 heads 之外补零。

局部历史布局由 `TPLayout.history_shapes(slots=..., pages=...)` 给出：KV pool 为 `(16, 4/TP, pages, 16, 256)`，卷积状态为 `(slots, 48, 4, 10240/TP)`，递推状态为 `(slots, 48, 48/TP, 128, 128)`。`validate_history_shapes` 只检查形状；实际接入时，还须由原生适配器确认 heads 顺序、rank 归属和请求的提交边界。原生若采用不同排列，需要明确转换，不能仅凭形状匹配。

后续原生完整 TP 需要对接三处：按局部 heads/通道编译 draft 图，引用本卡的 backbone 和历史；让局部 Attention/GDN 的 O projection、共享 MLP 的 down projection 沿用原生归约，并在归约之后加 residual；让各 rank 同步 anchor、候选及接受长度，按各自页表提交或恢复同一段逻辑历史。当前 `programs.py` 及部分原生内联函数仍固定 TP1 宽度，需随原生局部算子接口一起调整。共享 MLP 等 backbone 仍使用原生实现，不在 draft 中另建副本。

检查一个运行 rank 的参数布局：

```bash
python scripts/audit_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --mode head --runtime-tp 2 --runtime-rank 0 \
  --output /results/head-tp2-rank0.json
```

`HeadSource(..., runtime_tp=2, runtime_rank=rank)` 提供相同的 Python 接口。`head.tp` 表示 checkpoint TP，`head.layout` 表示运行布局。使用 `audit_qwen38_pypto_tp.py --source ... --model ... --head-dir ... --tp 2 --output ...` 可核对所有 rank：逐层与独立整张量读取对照，检查重组后参数完全相同；省略 `--head-dir` 检查 AR 复制初始化。以上入口不分配 NPU，不是多卡推理验收。

本轮已用真实 27B 的 TP4 测试 head 检查 TP2、TP4 运行布局，以及从 AR 权重初始化的 TP2 布局，三组均覆盖 64 层，重组后的参数和 dtype 完全相同。另在一张真实 A2 上复跑 TP1：K32 起草前后目标缓存不变，拒绝后的已提交历史与重放一致，后续续写和服务调度输出均与 Target 相同。331 项 CPU 用例通过，包含跨 TP 的局部读取、复制参数校验，以及小尺寸 Full Attention/GDN 分头后合并输出的数学对照；未执行整网多卡通信。证据见 `validation/pypto-qwen38/tp-preparation-20261105/`。

### 双卡驻留与行并行组件

`DraftWeights(..., worker_id=rank)` 将显存查询、参数分配、上传和释放绑定到指定设备 worker。`worker_id` 是 `DistributedWorker` 内的逻辑编号，不是物理卡号，也不自动从 checkpoint rank 推断。各 rank 的共享主机参数应在创建 worker 前以 `ReadOnlyHostTensor` 注册；释放失败时保留未释放的句柄及 owner，允许重试。

行并行链路为 `本卡投影 → 全块 Sum → residual`。`tp_programs.py` 直接调用原生 Linear 和 residual；归约调用 PyPTO 现有 `pld.tensor.allreduce`。固定版本的归约支持 FP16/FP32，不能直接接收 BF16，因此在设备上转 FP32、完成 Sum 后转回 BF16。归约以整个 `32 × 5120` 输出为单位，residual 在归约后只加一次，中间结果不经过 CPU。

使用 InCore Sum 是为了复用其可重复使用的 signal 协议；该版本的 HOST Sum 不能直接放进循环。测试入口在一次 dispatch 内连续归约三次，并改变输入重复调用，检查信号状态没有污染后续计算。

原生 Linear 的 K tile 为 512、输出 tile 为 128，现有循环不处理不足一块的尾部。TP2 的 Attention O projection 和 MLP down projection 分别有 3072、8704 个输入通道，均符合要求；TP4 的 MLP down 为 4352，需要补齐到 4608 或使用支持尾块的原生实现。`validate_linear_tiles` 会拒绝直接提交不整除的形状，防止静默漏算最后 256 列。这是算子调用布局要求，不改变模型公式。

在已分配的两张卡上运行：

```bash
python scripts/validate_qwen38_pypto_tp_components.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --devices 0,1 \
  --output /results/tp-components.json
```

添加 `--compile-only` 只编译四个入口，不分配 NPU。实测入口只加载真实 head 分片及一层共享 MLP down 权重，逐卡回读全部 64 层 head，并检查驻留后不再上传、错误 owner 释放被拒绝，以及两类投影的 Sum/residual 结果。CPU 只生成对照数据、校验回读值；这不是完整 TP2 模型服务或吞吐测试。

2026-11-06 在 167 的两张 A2 上完成实测：每卡约 7.34 GB head 完整驻留，64 层参数逐字节回读一致，重复调用不再上传 head；原生 Attention O projection 与共享 MLP down projection 各三轮，局部投影、整块 Sum 和 residual 均与对照完全相同。每次 Sum dispatch 内连续执行三次归约，信号重复使用通过。测试结束后两卡的 worker、控制进程和设备锁均已释放。记录见 `validation/pypto-qwen38/tp-components-20261106/`，后续仍需将这些组件接入原生完整 TP 的 Attention/GDN/MLP 计算图和跨 rank 验证事务。

## 运行入口

在 `online-speculation/` 内安装本项目，模型和 head 均作为只读外部目录挂载。原生服务入口只启动主模型基线，**不启用 Orthrus**：

```bash
python scripts/serve_qwen38_pypto_base.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --devices 0 --tp 1 --port 8000
```

检查真实 head 的基模契约、64 层参数、布局和 dtype，不执行 NPU 计算：

```bash
python scripts/audit_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --mode head --output /results/head.json
```

将 `--mode` 改为 `environment` 可检查固定 PyPTO/Simpler/PTOAS；改为 `compile` 可编译五个 draft 入口并检查 PTOAS 产物，不分配 NPU。所有审计输出必须使用新文件，保留失败证据。初始化 head 可省略 `--head-dir`，仅用于参数与链路检查，不代表训练后的接受率。

`validate_qwen38_pypto_flow.py` 是原生目标模型事务验证入口：先运行 target-only，再使用显式 oracle/拒绝候选检查验证和恢复。**候选来自测试 fixture，不运行神经 draft，不作为接受率或加速证据。**

真实神经 draft 的实验验证入口如下。`--audit-block` 单独执行一次完整 K32 起草，逐字节检查起草前后的目标 KV 和 GDN 状态，随后检查已提交历史与接受前缀重放是否一致。默认另生成 3 个 token，比较服务调度路径与 target-only 的续写；增加 `--tokens 33` 可延长请求验证，不以初始化 head 的接受率验收性能：

```bash
python scripts/validate_qwen38_pypto_neural.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --audit-block --output /results/neural.json
```

`serve_qwen38_pypto.py` 保留原生 OpenAI API、SSE、停止与取消处理：

```bash
python scripts/serve_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --head-storage auto --port 8000
```

`--head-storage auto` 按加载主模型和工作区后的空闲设备内存选择 `resident` 或 `stream`。前者一次加载全部 head，后续轮次不再上传参数；后者保留共享主机参数，复用设备缓冲，推理时不重新合并、校验或分配各层参数。显式指定 `resident` 但内存不足时会报错，不会静默切换。head 在进程内保持不变，换 head 后重启服务。

`--head-transfer packed` 在 `stream` 下将每层参数作为一个字节区间上传，再以类型视图传入原有程序；head 上传调用从每轮 528 次变为 64 次，传输字节量基本不变。默认仍为 `individual`：同一模型中的交错测试观察到小幅收益，但完整请求测试未见稳定提速。该选项不改变 Attention 公式或缓存语义；`resident` 不需要逐轮上传。

完整 head 的共享主机布局约 14.5 GB，另有原生模型加载和状态缓冲开销；容器建议使用至少 32 GiB 的 `/dev/shm`，或按部署规范共享宿主机 IPC。设备 `resident` 还需额外容纳完整 head。当前原生主模型在 64 GiB A2 上约占 57 GiB，实际选择的是 `stream`；增加卡数不会自动让此 TP1 路径分片。

`/metrics` 在原生指标之外返回 `orthrus` 字段，包括投机轮数、接受数、恢复次数、真实神经 draft 调用次数、head 存储方式和最近一轮各阶段耗时。使用 `smoke_qwen38_pypto_base.py` 时添加 `--head-dir /heads/version4 --tokens 3`，可检查 completion、chat、SSE、draft 调用计数及服务关闭。

固定同一前缀测量 K32 起草，并检查目标缓存、完整 logits 和验证后的历史：

```bash
python scripts/benchmark_qwen38_pypto_draft.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --head-storage auto \
  --rounds 3 --output /results/draft-bench.json
```

该入口的计时仅包含 `propose`，缓存哈希审计在计时区间外执行。`benchmark_qwen38_pypto_snapshots.py` 使用同样的 `--source`、`--model`、`--device` 和新 `--output` 路径，交错比较独立状态副本与共享快照，不加载 draft head。`validate_qwen38_pypto_storage.py --device 0 --output /results/storage.json` 使用小张量在真实 NPU 上检查两种存储方式和只读边界，不加载 27B。

完整请求的接受长度对照使用独立入口：

```bash
python scripts/benchmark_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --device 0 --head-storage auto --tokens 128 --repeats 3 \
  --output /results/request-speed.json
```

它在同一 worker 中依次运行原生目标路径、真实 draft 和全接受控制组，每组先预热一次。默认从 AR attention 初始化 head，也可传入 `--head-dir`。计时包含 prefill、调度和请求释放，忽略 EOS、定长生成；另记录参数上传、设备执行、验证和 CPU greedy 校正时间。控制组先完成真实起草，再替换为目标路径预先保存的正确候选，不跳过 head 计算，不伪造 verifier 的接受结果。128-token 请求与 vLLM 对照采用相同提示和输出预算，但 PyPTO 为 TP1、512 上下文，不能直接将两个框架的绝对 tok/s 当作同配置对比。

## 环境复现

官方开发镜像可作为构建底座。本次使用的镜像 digest 为 `swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:7712c0bae0f8a3c46cae127be4802324682e33ad1c49436e17db7bd60975dbee`，内置版本与交付包不同，需要重建固定依赖。

先将上表固定版本的 OneServing、PyPTO 及其子模块准备好；PTO-ISA 放在 `PyPTO/runtime/build/pto-isa`，PTOAS wheel 从 `hw-native-sys/ptoas` 的 `v0.65` release 获取。容器内执行：

```bash
/opt/pypto/venv/bin/python scripts/build_qwen38_pypto_environment.py \
  --pypto-source /sources/pypto --oneserving-source /sources/OneServing \
  --ptoas-wheel /wheels/ptoas-0.65-cp310-cp310-manylinux_2_34_aarch64.whl \
  --output-dir /results/build-1 --jobs 4
```

构建脚本检查源码 revision 和 wheel SHA256，保留各阶段日志。依赖采用 OneServing 自身的 `requirements.txt`，其中 Transformers 固定为 `5.17.0`。构建后使用该 Python，将 `/opt/pypto/venv/bin` 放到 `PATH` 首位，并将 `/sources/pypto/python` 与本项目 `src` 放入 `PYTHONPATH`；清除旧镜像的 `PTOAS_ROOT`，避免重新选中旧工具链。

先运行 `probe_qwen38_pypto_environment.py --device 0 --output /results/probe.json`，通过后再加载真实模型。它检查版本并实际执行向量加法，输出文件必须使用新路径。

本次真实 NPU 验证采用宿主机原生用户进程，使用 Python 3.10.21、Torch 2.10.0+cpu、CANN 9.1.1 和驱动 26.1.1；PyPTO、Simpler、PTOAS、PTO-ISA 仍为上表固定版本。Python/Torch 负责权重与主机侧数据，模型计算由 PyPTO 程序在 NPU 执行。

在可写的用户源码目录中，使用宿主机 CANN 重建设备运行时：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.1
python -m simpler_setup.build_runtimes --platforms a2a3
```

依赖准备好、分配到空闲设备后，可执行下面的原生服务检查。`ASCEND_RT_VISIBLE_DEVICES` 指定物理卡，脚本中的 `--device 0` 对应可见设备的第一个逻辑编号：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/smoke_qwen38_pypto_base.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --device 0 --output-dir /results/native-smoke-1
```

脚本启动独立服务，检查 completion、chat、SSE 和 metrics，随后正常关闭；保留 `server.log` 和 `result.json`。输出目录必须为新目录。上面的命令不加载 Orthrus head。

模型从其他下载工具复制过来时，可能缺少 OneServing 检查的 Hugging Face 下载记录。本次在 61 上逐一核对了 18 个权重分片和推理配置的 SHA256，确认与已验证副本一致，再在独立模型视图中复制原始下载记录。共享模型目录未修改，版本校验保持开启；哈希记录见 `validation/pypto-qwen38/http-20261105/checkpoint.json`。

## 验证与后续

已通过 338 项 CPU 用例，覆盖 K32 每个拒绝位置、输出截断、停止 token、异常恢复、跨页回收、递推状态恢复、缓冲边界、TP 参数合并及原始设备错误保留；还包括打包视图的 owner/布局检查、指定 worker 的内存操作与释放重试、全接受测试保留真实起草，以及全部 BF16 编码和并列最大值的输出选择检查。另读取真实 27B 离线训练导出的 TP4 head，检查完整 64 层的布局转换和 dtype，保持输入只读。构建脚本已实际执行通过；五个 draft 入口通过前端编译和 PTOAS 代码生成，向量计算通过模拟器执行（误差 0）。初次验证证据见 `validation/pypto-qwen38/port-20261105/`，缓存布局修正后的记录见 `neural-20261105/`。

真实 NPU 已完成以下检查，证据见 `validation/pypto-qwen38/target-only-20261105/`：

- 固定软件栈的向量加法实际在 NPU 执行，最大绝对误差为 0。
- 完整 64 层 BF16 主模型在单张 64 GiB A2 上加载，以 TP1、一个活动请求运行。
- completion 返回文本，chat 对 `1+1` 返回 `2`；SSE 返回 token 及 `[DONE]`。
- 三次请求完成、请求错误数为 0；应用正常关闭，服务和 NPU worker 进程均退出，测试卡恢复无用户进程状态。

此次原生服务的 HBM 峰值约 57.3 GiB，包含当前运行时配置；这是短请求检查，不是吞吐、长上下文或完整数值一致性验收。

K32 验证与拒绝恢复已在同一真实模型上通过，证据见 `validation/pypto-qwen38/transactions-20261105/`。全接受、首位拒绝、第 17 个候选拒绝、最后一个候选拒绝四类用例均生成 33 个 token，与 target-only 逐个一致；首位拒绝用例连续执行 31 次恢复。验证跨越 16 行 tile 和 KV 页边界，结束后请求均已释放。这里的候选由测试 fixture 提供，只证明目标验证与缓存事务可工作，不证明神经 draft 前向、接受率或加速效果。

真实 head 检查使用 TP4 离线更新 4 步导出的测试 head，按层合并并在 TP1 上执行，证据见 `validation/pypto-qwen38/neural-20261105/`：完整 K32 起草返回 31 个候选；起草前后全部目标 KV、卷积状态、递推矩阵及页归属保持不变。验证接受 1 个候选后发生拒绝，提交后的历史与仅重放接受前缀逐字节一致，随后 3 步 AR 续写一致。另通过真实 scheduler 生成 3 个 token，与 target-only 完全相同，确实调用了神经 draft；请求释放、worker 与验证进程退出均已确认。

首次神经 draft 执行暴露了运行时对缓存视图连续性的要求：选中 slot/layer 后须先去掉相应长度为 1 的维度，再 reshape。GDN 和 Full Attention 读取均已按此修正。本次检查证明了前向与验证链路可运行，尚未比较 PyPTO 与 PyTorch draft 的逐层数值，也不代表长请求、并发或生产验收。测试 head 只经历少量更新，不据此评价接受率或服务加速。

Orthrus HTTP 检查在 61 的单张 A2 上通过，加载的是 TP2 在线 last-4 更新导出的 version298 测试 head，部署仍为 TP1。completion 返回 `, I'm`，chat 对 `1+1` 返回 `2`，SSE 正常返回 token 和 `[DONE]`。三次请求全部完成，错误与取消数为 0；指标记录 3 次神经 draft、3 轮验证和 1 次恢复。服务正常关闭，验证进程、服务进程及 NPU worker 均已退出，证据见 `validation/pypto-qwen38/http-20261105/`。

## 推理开销对照

在同一张 A2、同一个真实 TP4 测试 head 和相同历史前缀上，分别测量初版和参数缓冲版本。CPU 使用 4 个线程，所有 NPU 调用同步完成后计时结束；不计入启动、缓存哈希和落盘时间。记录见 `validation/pypto-qwen38/optimization-20261105/`。

| 项目 | 初版 | 参数缓冲版本 |
| --- | --- | --- |
| K32 起草中位数 | 56.63 s（2 轮） | 2.70 s（3 轮） |
| 每轮 head 准备 | 重新读文件、合并和校验 | 已在启动时完成 |
| 每轮设备参数分配/释放 | 各 528 次 | 0 次 |
| 实际 head 存储方式 | 逐层分配并上传 | `stream`，复用两组缓冲 |
| 完整 31 × 248320 logits | 与右侧逐字节相同 | SHA256 相同 |
| 目标缓存只读、接受后历史 | 通过 | 通过 |

起草阶段约快 21 倍，主要消除了重复转换和跨进程临时搬运，并未改变 Attention 算法。转换占用约 55 秒，被移至启动阶段；这组记录中完整启动从约 157 秒增至 234 秒。单卡仍需每轮上传约 14.5 GB 的 head，传输约占 2.3 秒，设备执行约 0.35 秒。设备内存足够时可使用 `resident` 避免重复上传；两种模式均已通过真实 NPU 小张量检查，但完整 27B 加全 head 驻留未在这张 64 GiB 卡上验收。

验证侧分开测量了两项变化：批量读回有效 logits，使验证前向段减少约 70–80 ms；预注册 GDN 快照缓冲，减少主机临时分配和跨进程中转。后者在同一个模型中交错运行 24 个事务，首位拒绝、接受一个候选后拒绝和全部接受三类用例，输出及已提交缓存均一致。排除第一轮后，前两类事务中位数约从 1.05–1.07 秒降至 0.54–0.55 秒，全部接受从 0.65 秒降至 0.38 秒。共享快照占约 310 MB 主机内存，未增加 NPU 缓存。

合入共享快照后的完整版本另在 A2 复验：三轮 K32 起草中位数 2.52 秒，随后验证事务约 0.51 秒，完整 logits 及接受后的历史检查均通过；completion、chat、SSE 也重新通过，三次请求无错误，服务正常退出。该复验使用另一张卡，未用于上表的同卡提速比值。

打包上传另在真实 NPU 上核对了全部 64 层参数字节和原生 Full Attention 的投影输出。随后在同一个完整模型中交错运行两种上传方式，各三轮：完整 logits 的 SHA256、候选 token、目标缓存只读和拒绝后的历史恢复均一致。K32 起草中位数为 2.00 s / 1.92 s，差异较小，不能按上传调用数的降幅推算整体提速。

输出选择复用 Torch 现成归约：以 `aminmax` 检查非有限值，直接取 BF16 `max(...).indices`，省去整个 logits 矩阵的 FP32 转换和布尔中间张量。在真实 31 × 248320 logits、4 个 CPU 线程的交错测量中，中位数由 59.62 ms 降至 17.19 ms，选择结果相同；NaN/Inf 检查和并列时取首个 token 的语义保留。这是主机侧选择阶段的计时，不是整个请求的提速比。相应原始记录见 `validation/pypto-qwen38/speed-20261105/`。

最终版本另完成同一 worker 上的 128-token 完整请求对照，各组预热一次、计时三次，输出逐个一致：原生 Target 中位数 7.81 tok/s，AR 复制初始化的 head 为 1.24 tok/s，全接受且保留真实 draft 计算的控制组为 19.25 tok/s。真实 head 平均每轮接受 1.08 个候选；控制组只用于检查理想接受条件下的净加速。统计口径与 vLLM 对照见 [接受长度与推理开销](QWEN38_SPEED_BENCHMARK.md)，本轮所有测试进程及设备锁已释放。

上述数据用于定位和消除接入开销。接受率、完整请求吞吐和 goodput 还需使用训练充分的 head 与代表性请求评估。

早期容器中的设备枚举错误记录保留在 `port-20261105/`。本次通过用户目录下的原生环境完成验证，没有修改 OneServing 模型源码；原容器启动失败的具体原因仍未确认。

后续扩展请求边界，并随原生模型演进接入完整 TP、head 驻留、上下文和采样。完整训练和其他 last-N 更新继续使用现有 PyTorch 路径；本分支的 PyPTO 在线入口限定 last-1，重启加载完整 head 与进程内在线更新是不同操作。
