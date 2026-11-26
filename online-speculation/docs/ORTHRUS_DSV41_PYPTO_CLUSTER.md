# DSV4.1 Orthrus 双机调度和通信

目标布局为两台各 8 张 NPU，DP4/TP4/EP16。每台持有两个完整 TP4 分区。节点 0 持有全局 rank 0–7，节点 1 持有 rank 8–15。本机设备编号与全局 rank 分开。TP 在节点内通信，DP 按同一 TP lane 跨节点，MoE EP 覆盖全部 16 个 rank。

调度端发送 Serving 的 DP_engines 请求波次。工作端保留本节点负责的全局 DP、request slot 和 generation，空分区仍保留。两端依次确认 Prefill、Draft、Verify、Accept、Decode 和 Release。Draft 和 Verify 保持 epoch，Accept 成功后才推进。任一侧失败，当前会话进入失败状态。需要返回结果时，控制连接只汇总带请求归属的 token ID 行。

设备通信复用配对 PyPTO Lib 的 HCCLBytes 和 resident tensor 接口。跨机组通过 HcclGetRootInfo 和 HcclCommInitRootInfo 建立，按全局 rank 组织 HBM 上的 AllToAll、AllGather、ReduceScatter 和 AllReduce。通信前核对对端字段顺序、形状和 dtype。

## 验证入口

公司 node7 和 node8 使用同版仓库及已验证的离线镜像。生成一次作业密钥，并把同一个文件经 SSH 或 SCP 放到两台机器执行账号的 `~/.orthrus-cluster.key`。可用 `CLUSTER_KEY_FILE` 指定其他位置。

```bash
(umask 077; openssl rand -hex 32 > "$HOME/.orthrus-cluster.key")
```

先启动 node7，再在两分钟内启动 node8。控制测试复用两卡探针已验证的 TCP 端口 29641。测试用 8+8 的逻辑布局，走请求路由、阶段顺序、接受提交和 token 回收，无需 NPU。每次启动会校对两端的代码标识。

```bash
# node7
bash online-speculation/scripts/dsv41_pypto_env.sh cluster-check node7 --control-only

# node8
bash online-speculation/scripts/dsv41_pypto_env.sh cluster-check node8 --control-only
```

实卡测试只挂载 `NPU_DEVICES` 中指定的物理卡。两台各八卡空闲时执行下方命令，检查 TP4、DP4、EP16 的四种集合通信。若先用两台各两卡联调，两端把卡号改成实际空闲的卡，并加 `--counts 2 2 --tp 2`。卡号应按容器映射顺序列出。

```bash
# node7
NPU_DEVICES=0,1,2,3,4,5,6,7 bash online-speculation/scripts/dsv41_pypto_env.sh cluster-check node7

# node8
NPU_DEVICES=0,1,2,3,4,5,6,7 bash online-speculation/scripts/dsv41_pypto_env.sh cluster-check node8
```

两台最后一行均为 `passed: true` 且 `device_collectives` 包含 `ep/all_to_all`、`ep/reduce_scatter` 和 `ep/all_reduce`，才算对应实卡通信通过。日志和结果写入各自的 `orthrus-cluster.*` 目录。

其他机器可以直接运行 Python 入口，并显式传入本机编号、rank 0 地址、密钥和 PyPTO Lib 路径。用 `--control-only` 可先验证 8+8 布局。

```bash
export PYTHONPATH="$ORTHRUS_ROOT/online-speculation/src"
python "$ORTHRUS_ROOT/online-speculation/scripts/validate_dsv41_pypto_cluster.py" \
  --control-only --node "$NODE_RANK" --master-address "$RANK0_IP" \
  --key-file "$JOB_KEY" --build-id "$BUILD_ID"
```

使用设备探针时，设置 `ASCEND_RT_VISIBLE_DEVICES` 为本节点分配的物理卡，`HCCL_IF_IP` 为本节点用于 HCCL 建链的内网地址。默认每节点八卡、TP4。少卡联调用 `--counts 1 1 --tp 1`。选卡时按实际设备的 HCCL 拓扑约束选择通信组。控制端口可达后，设备间仍需满足 HCCL 的建链条件。 [CANN 通信域](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclug_000008.html) · [AlltoAll 约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/comm_op_interface/HcclAlltoAll.md)

```bash
python "$ORTHRUS_ROOT/online-speculation/scripts/validate_dsv41_pypto_cluster.py" \
  --node "$NODE_RANK" --master-address "$RANK0_IP" \
  --key-file "$JOB_KEY" --build-id "$BUILD_ID" --lib "$PYPTO_LIB_ROOT"
```

## 实测边界

公司内网 node7 和 node8 已通过 8+8 卡的 DP4/TP4/EP16 实卡验证。两端构建标识一致，全局 rank 0–15 执行了 TP/DP AllGather、EP AlltoAll、ReduceScatter 和 AllReduce，覆盖从 prefill 到 release 的六个调度阶段。这次使用测试张量验证调度和通信，整模 W8A8 验收见下一节。 [公司 16 卡记录](validation/dsv41/pypto-cluster-company-ep16.json) · [公司双卡记录](validation/dsv41/pypto-fabric-company-ep2.json)

8+8 逻辑 rank 的控制流程通过了双进程和双离线容器的 TCP 回环。外部实验机上的单机双卡测试覆盖了驱动 25.5.1/CANN 9.0.0 与驱动 26.1.1/CANN 9.1.1，25.5.1 环境另复验了物理到运行时卡号映射。三卡 2+1 进入 EP AlltoAll 后报设备链路超时 `507048`，相同卡两两连接的用例通过。 [单机两卡记录](validation/dsv41/pypto-cluster-loopback-ep2.json)

## 整模接入

原 A3 Serving 由一个 worker 持有 16 个逻辑设备。对应 Lib 的 `run_network_runtime` 用全局 EP 数量配置本地设备，集体通信调用 `HcclCommInitAll`。1121 配对更新了原生执行接口，启动仍要求单个任务分配 16 个设备。 [原 Serving](https://github.com/Adelaide999/pypto-serving/blob/cd818945c3c4e5454b8c259306d4e6f9e865e05c/models/deepseek_v41_a3/README.md) · [原 Lib 组网](https://github.com/Adelaide999/pypto-lib/blob/3de8339fdd60f7a4aae9f221c85dcdf3c7a17612/models/deepseek_v4_1_flash_a3/moe_transport.py) · [1121 启动说明](https://github.com/Adelaide999/pypto-serving/blob/27efbcb1cd3466547823955cb80fe09c63b3f754/models/deepseek_v41_a3/RUNBOOK.md)

`ResidentDraftAttention` 可接收本机拥有的全局 rank，按全局 TP lane 选参数，映射到本地设备执行。使用目标 W8A8 checkpoint 导出的初始化 head，将全局 rank 8 映射到本机设备 0 完成 40 层 K32 draft。输入历史缓存逐字节一致，运行期间没有重复上传 head。 [实卡记录](validation/dsv41/pypto-resident-w8a8-rank8-910b3.json)

基模的本机权重加载通过独立的 [Lib 补丁](../runtime/pypto_dsv41_a3/lib-rank-weights.patch) 提供，应用在现有 `lib-serving.patch` 之后。CPU 契约测试验证了两个本机 worker 只读取 rank 8 和 9 的路由专家，静态句柄按全局 rank 保留。配对 Lib 的 14 项会话测试通过。现有离线镜像仍用原配对。

本机缓存分配通过 [Lib 缓存补丁](../runtime/pypto_dsv41_a3/lib-rank-cache.patch) 提供，应用在权重补丁之后。两端保留全局请求、页表和编译尺寸信息，各自只准备本机 rank 的 KV、索引缓冲和 Engram 历史。15 项 CPU 会话契约测试通过，覆盖全本地与仅 rank 8–15 持有缓存时的历史绑定和请求回滚。

整模验收入口将本机权重和缓存接进 `GenerationSession`，`NetworkExecutor` 把 TP、DP 和 EP 的张量交给已验证的 HCCL 通信层。专家归属及编译尺寸仍按全局 EP16 计算，空闲 DP 也参加通信。Engram 从本机 checkpoint 文件映射，按需读取 CPU 行。两端同步 Serving 请求的页表和阶段，生成时只在控制连接上传输贪心 token ID。 [Lib 执行补丁](../runtime/pypto_dsv41_a3/lib-cluster-execution.patch) · [Serving 补丁](../runtime/pypto_dsv41_a3/serving-cluster.patch)

### 双机整模验收

两台机器拉取同一个 `feature/orthrus-dsv41-pypto` 提交，保留上次已通过 `cluster-check` 的离线镜像和密钥。模型默认使用 `/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8`。命令会在临时容器内应用补丁，无需重新构建镜像或联网下载依赖。

```bash
# node7
CLUSTER_KEY_FILE=/root/.orthrus-cluster.key \
  bash online-speculation/scripts/dsv41_pypto_env.sh cluster-accept node7

# node8，在另一终端启动
CLUSTER_KEY_FILE=/home/wangzili/.orthrus-cluster.key \
  bash online-speculation/scripts/dsv41_pypto_env.sh cluster-accept node8
```

两端命令都默认使用已预留的卡 0–7。无训练 head 时，每台从同一份 W8A8 基模生成 BF16 初始化 head，用于验证推理链路。已有 head 时，在两端命令前设置 `DRAFT_HEAD=/绝对路径`，脚本会校验两份 head 的内容一致。`PYPTO_RESULTS_ROOT` 可指定结果盘，默认 `/workspace`。首次启动会读取 Engram、编译模型算子并加载权重。

重跑会优先使用结果盘上最近一次导出且与基模匹配的初始化 head。Engram 分片的 SHA256 保存在结果盘的 `.orthrus-dsv41-state` 中，只有底层文件标识和时间戳一致时才复用。首次写入此缓存时，日志会逐分片打印校验进度。当前离线镜像通过 `pl.tile.rsqrt(x, tmp)` 保留基模所需的高精度倒平方根路径。

程序先运行 target-only 再运行 Orthrus，在两个节点的 DP 分区各运行一个请求。两次生成复用同一份已加载的基模和通信组，各自重新分配并清理请求缓存。起草前先比较 prefill 生成的第一个 token，之后每轮验证立即比较新增 token。默认追踪 prefill 最后一个 prompt token 的 embedding、每层 Attention/HC/MoE 子阶段和输出头；`--trace-position` 同时指定 Decode 与 Verify 的绝对位置。首 token或验证阶段分叉时，退出前都会自动打印第一处不同的阶段、rank 和张量误差。`--tokens 4` 可快速覆盖前四个输出位置，两端需传相同参数。两次独立冷启动的精度一致性仍需另起进程验证。

程序检查全部 40 层、逐 token 的贪心一致性、K32 起草和验证、接受前缀提交、验证后的 KV 恢复及请求释放。两台各自记录 `orthrus-cluster-accept.*/acceptance/acceptance.json`，详细日志为同目录的 `cluster-accept.log`。先看两端的 `passed`，再看 `compared_tokens`、`draft_rounds` 和 `verifier_restored_bytes`。此入口用于离线推理验收。初始化 head 的接受数供诊断使用，性能结论需用训练 head。

已有验收结果可在任一节点用下列命令汇总，无需重新运行模型。报告区分各阶段首次调用与后续中位数，并附上本机的验证缓存搬运量、draft 权重上传次数和耗时。首次调用可能包含编译，计时也含本次验收的诊断开销。

```bash
python3 online-speculation/scripts/summarize_dsv41_pypto_cluster.py /workspace/orthrus-cluster-accept.RUN/acceptance
```

对已完成生成的在线 profile 加 `--online-profile` 可直接读取保留的各阶段结果和更新日志，查看贪心输出对照、draft 版本、接受数及本机冷启动和稳定更新耗时。

Draft 和 Verify 只回读实际参与本轮的词表行。Draft 结果直接交给候选解码；Verify 每个活跃 DP 只保存一个规范 TP rank 的有效 logits，供分叉诊断和复验。Accept 继续执行已接受前缀的状态提交，省去最终 LM Head 和贪心输出交换，回执用 `output_head_skipped` 记录。采用 128 行容量、32 行有效输入和 129280 词表的单卡小测，单 rank 词表回传从 33.1 MB 降到 8.27 MB，回读输出逐字节一致。双机整模收益用 `draft`、`verify`、`accept` 的阶段计时分别衡量。

投机阶段的 Host 元数据按 DP 构造，TP rank 保留自己的序列并行切片。常规 K32 Draft 和 Verify 共享相同的位置、页表和分片控制，并复用同一份元数据；Verify 的候选 token ID 保存在独立的输入契约。每次写盘将动态控制合成一个文件，原生调用继续接收完整布局和 Engram 常量。111 的四请求 K32 小测中，按字段核对结果一致，构造元数据从 188 ms 降到 47 ms，单次证据序列化及校验从 431 ms 降到 92 ms，文件体积从 19.6 MB 降到 0.71 MB。两项均为 Host 局部计时。

### 1122 原生整图双机推理和在线更新

原生入口使用固定的 `949a18d`。node7 和 node8 各管理 8 张本机卡，Simpler L4 为两个 L3 worker 建立设备共享窗口。AR 与 Orthrus 共用同一套目标 Prefill 和 Decode 整图。教师 logits 随校验提交在设备侧分段保存，last‑N 反向完成后再在设备侧发布参数。

两台切换到 `orthrus-dsv41-pypto-1122` 并更新到同一提交，各执行一次 `dsv41_pypto_env.sh refresh-1122`。从 node7 仓库根目录运行下列命令。默认通过免交互 SSH 启动 node8，可用 `NODE8_SSH` 和 `NODE8_CHECKOUT` 指定远端账号和路径。

```bash
RUN=online-speculation/scripts/run_dsv41_pypto_company_pair.sh
bash "$RUN" native-fabric
bash "$RUN" native-prepare 1
bash "$RUN" native-windows 1
bash "$RUN" native-smoke 1
bash "$RUN" native-study 1
```

`native-fabric` 使用每台一张空闲卡检查设备窗口导出和跨机访问。默认物理卡分别为 node7 的 1 号和 node8 的 0 号，可通过 `NODE7_FABRIC_NPU` 和 `NODE8_FABRIC_NPU` 调整。`native-prepare` 用 CPU 编译并分发整图和初始化 head，再在 node7 装配并登记 Prefill、Decode、Draft、反向和 Adam 的二进制，无需空闲 NPU。产物目录中的 `callable-assembly.json` 记录每张图注册的 callable 数量和总耗时，详细日志保存在 `callable-assembly.log`。

`native-smoke` 需要两台各 8 张空闲卡，一次装载后从 node7 和 node8 的请求分别触发一次更新，检查 Prefill、Draft 和 Decode/Verify 中所有采样行的有限性及 greedy ID，再逐 token 对照两组输出。数值检查失败时，`result.json` 记录具体阶段、DP 和出错行。公司双机的原生 Fabric 和整模仍待实测。

`native-windows` 可单独用两台各 8 张空闲卡预留整图的实际通信窗口。它涵盖 Prefill、Decode、Draft、后缀反向和 Adam 中需要通信的图，跳过设备 callable 装配和权重上传。报告写入 `native-windows.*/result.json`。窗口预留失败时，日志注明对应图、参与 rank 数和每卡窗口大小。这个检查和常规短验、GSM8K 都在启动设备进程前核对两端的权重清单、head 清单和编译图标识，在上传权重前预留同一组窗口。

按默认产物的张量形状、通信窗口和 ring heap 核算，峰值 rank 的已知 HBM 分配约为 last‑1 51.8 GiB、last‑4 53.5 GiB。实卡峰值还会包含运行时的其他分配。原生在线路径复用驻留教师缓冲。Adam 把整个后缀的梯度范数和有限性检查结果一次回读，校验全部通过后才更新并发布权重。

没有通信窗口的 Adam 参数更新合批派发。共享同一编译图的更新分批提交，等全部参数更新结束后再发布。用真实编译图做宿主回放，last‑1 每次更新的 Adam Apply 从 7 次控制往返降至 2 次，last‑4 从 28 次降至 8 次。111 上用同机两张 910B3 和真实 L4/L3 调度比较七项小尺寸参数更新，合批前后 FP32 参数和动量逐字节一致，热态中位耗时从 145.5 ms 降至 135.2 ms。计时涵盖参数更新和调度，整模在线吞吐仍由双机实验测量。

Prefill、Decode/Verify 和 Draft 在同一次整图派发中把采样 ID 回写给调度端。词表 logits 留在 NPU，供教师快照和短验诊断使用。Draft 的 `graph_seconds` 包含随派发返回采样 ID 的时间。

同一批请求结束时间不同时，协调端只在派发头中标记刚空闲的 DP。归属节点用自己的空输入模板重置其 TP4。对活跃组，协调端把输入直接写入复用的发送缓冲。超过 Simpler 16 MiB 单帧上限的输入分段写入同一端点缓冲，派发前完成拼接。

Prefill 只发送有效 token 的 embedding、Engram 和滑窗索引行，接收端按编译布局恢复填充。按默认 GSM8K 的 64 道在线题和 16 道留出题计算，四并发时 node8 每批输入的中位体积由 24.0 MB 降至 10.1 MB，20 批都可以用单帧传输。设备输入布局在整图调度回归中逐字节一致。

Draft 派发只发送 anchor 和 mask 两行嵌入。Engram 也只查询并发送前五行，后续位置的最长 n-gram 已全部由 mask token 组成，归属节点用第五行恢复尾部。K16 每个活跃 DP 的输入从 448,000 字节降至 177,664 字节。用公开 W8A8 权重检查不同历史位置和教师采样长度，完整还原的 Draft 输入与旧路径逐字节一致。K16 宿主查询、构造和打包的中位时间由 2.70 ms 降至 2.51 ms。同机双卡以等量数据测试原生控制通道，小图派发中位时间从 23.50 ms 降至 21.40 ms。整模吞吐还需双机计时。

Decode/Verify 每个活跃 token 的输入残差只传一个 mHC 流。归属节点广播到各流并清零填充行，仍用完整输入执行原生整图。每 DP 的 K8 Verify 输入为 570,880 字节，单 token AR 输入为 427,520 字节。

node7 的执行容器可以写编译产物目录，供 PyPTO 装配整图二进制时创建缓存和锁。node8 和两台的模型权重均使用只读挂载。

验证器仅推进已提交的前缀，C2A 环形状态和压缩页继续驻留。用 W8A8 检查点第 2 层的 BF16 压缩权重和合成激活，在单张 910B3 上执行 `validate_dsv41_pypto_verify_state.py`，覆盖 K8/K16、环形状态回绕和压缩物理页切换。逐 token 基准与“先计算全块、拒绝尾部再续跑”在已提交的环槽、FP4 行及续跑结果上逐字节一致。

一个批次结束时，调度端一次通知两端清理各自的 KV 页和 C2A 状态。相邻物理页合并拷贝。两端确认清理完成后才归还请求槽位，清理失败时会停止当前 session。

GSM8K 默认使用 64 道在线题和 16 道留出题，四并发，每题最多生成 64 token，K8，last‑1，TV loss，学习率 `1e-5`。训练每隔 16 个验证轮次至多采一个反馈块，每块最多 7 个位置。`result.json` 记录 AR 与在线更新的实测吞吐、扣除更新时间的估算、候选接受率、首次及后续更新耗时，以及训练前后冻结 head 在同一留出集上的结果。预热和正式计时批次都逐轮与同批 AR 对照。发现首个分叉时立即结束本批次，记录请求、轮次、候选和输出，只用通过对照的完整批次统计吞吐。每个批次的记录写入 `study-waves.jsonl`，完整运行日志写入同目录的 `node7.log` 和 `node8.log`。

### 1122 分阶段双机推理

`1122` 的八卡节点分别执行本机的 TP4 Attention 和输出头。每层 FFN 的专家派发和回传使用跨机 EP16 HCCL。Draft 复用目标模型的只读历史页和冻结的 FFN，临时双向块及输出缓冲由 Draft 独立持有。Verify 根据接受长度推进提交位置。候选写过的页可以留在本请求的分配里，后续读取由提交位置和可见长度限制，在复用前写入新值。只有请求所在节点更新或截断 Engram 历史。

公司 node7 和 node8 使用同一个 `1122` 镜像、同版 W8A8 权重和同一份集群密钥。node7 默认读取 `/root/.orthrus-cluster.key`，node8 默认读取 `/home/wangzili/.orthrus-cluster.key`。镜像基于现有的 `orthrus-dsv41-pypto:72a62d1-offline`。在可以访问依赖的 ARM64 主机上执行一次 `bash online-speculation/scripts/dsv41_pypto_env.sh build-1122`，再将生成的 `orthrus-dsv41-pypto:1122-949a18d` 镜像复制到另一节点。已经配置的 `PYPTO_BUILD_NETWORK`、`PYPTO_BUILD_PROXY`、`PYPTO_BUILD_CA` 和 `PYPTO_BUILD_PYPI` 可继续使用。

两台已经有 `1122-949a18d` 时，更新 checkout 后可在各自机器执行 `bash online-speculation/scripts/dsv41_pypto_env.sh refresh-1122`。它从保留的 1122 基础镜像离线更新模型补丁和公司脚本，连续刷新复用同一基础镜像，避免叠加 Docker 层。接着运行 `prepare`，它会回载图产物并重编译不兼容的产物组。旧目录移入 `.superseded.*`。两端新的作业会核对初始化或预训练 head 的实际权重字节摘要。

1122 的基模、起草、反向和 Adam 图共享一个驻留 Worker。公司镜像将 Simpler 的主机和 AICPU callable 注册表统一扩至 256 项。如果已有镜像仍使用 64 项，`refresh-1122` 会用镜像里的源码和工具链离线重建 Simpler。已有的编译图产物可以继续使用。

111 的 ARM64 测试镜像使用同一组固定版本，离线刷新和启动检查用时约 7 秒。旧测试产物的目标图源码指纹不一致时，`prepare` 重编了 Prefill 和 Decode，并回载了此前的 6 个 Draft 图、13 个 last-1 反向图和 24 个 Adam 图。再次执行复用了全部图。初始化 head 的 40 个权重分片均参与校验。
同一主机也从 `72a62d1-offline` 完整构建了 `1122-949a18d`，在新镜像上执行了离线刷新，并复验了实际 W8A8 checkpoint、已准备的图产物和 GSM8K 输入。

两台 checkout 更新到同一提交后，可从 node7 仓库根目录一次启动双机。默认从 node7 用免交互 SSH 登录 `root@7.150.0.37`，并在 node8 使用同一绝对 checkout 路径。可用 `NODE8_SSH` 和 `NODE8_CHECKOUT` 指定远端账号和目录。远端账号需要能运行 Docker、读取权重和密钥。已有 GSM8K 分区默认读 `/workspace/data/orthrus-dsv41-gsm8k`。

```bash
RUN=online-speculation/scripts/run_dsv41_pypto_company_pair.sh
bash "$RUN" prepare 1
PYPTO_PACK_DECODE=1 bash "$RUN" smoke 1
PYPTO_PACK_DECODE=1 bash "$RUN" study 1
```

`prepare` 在两端依次用 CPU 准备初始化 head、编译产物和实验请求。`smoke` 和 `study` 启动前在两端检查镜像内的运行入口、模型补丁和在线更新模块与 checkout 是否一致，并检查权重分片、密钥、输入文件和设备路径。镜像代码落后时，在提示的节点执行 `refresh-1122` 和 `prepare`。仅检查时在命令前加 `PYPTO_CHECK_ONLY=1`。短验用 DP0、DP2 上的请求验证 AR 逐 token 对照、draft、verify、接受、在线更新与发布。两端的 `passed` 和 `online_passed` 均为 `true` 后，再执行 `study`。两种运行都保存 `run.log` 与 `result.json` 到脚本打印的目录。需要分别启动时，两台可各自调用 `run_dsv41_pypto_company_staged.sh`，依次传入节点名、模式和更新层数。

终端显示阶段切换、首个及每十个批次或更新的进度、验收结果和错误摘要。完整编译和运行日志分别保存在 `prepare-*.log` 与 `run.log`。每次在线更新的耗时、发布版本和分阶段计时保留在运行日志和结果报告中。

1122 同时使用 PyPTO Worker 内的通信域和进程内 HCCL EP/DP 通信域。启动脚本默认设置 `HCCL_NPU_SOCKET_PORT_RANGE=auto` 和 `HCCL_HOST_SOCKET_PORT_RANGE=auto`，以避免竞争 NPU 的默认端口 16666。公司网络要求固定端口时，可从 node7 传入指定范围，脚本会传给两个节点。

`smoke` 启动时可设置 `PYPTO_TEACHER_PROBE=1` 检查教师 logits。它沿用同一请求批，每题最多生成 17 个 token，保留第一次更新前的完整 K8 验证宽度，并在 `teacher-trace.jsonl` 保存 AR、Verify 和训练读取的同位置记录。出现非有限教师值时，`teacher-failure.jsonl` 还会记录三个读数及 Verify 和训练读取是否逐位相同。

在 111 的 ARM64 CANN 环境，按上述尺寸编译了 Prefill、Decode、6 个 Draft 入口、13 个后缀反向入口和 24 个优化器入口，并将编译目录移动后回载。使用指定 W8A8 权重导出了初始化 head 的全部 40 层，验证了 K16 启动时读取的 head 格式和真实 tokenizer 生成的 GSM8K 请求。两台公司机器仍需实跑两种模式。

这组实验的学习率为 `1e-5`，固定 16 行起草图，每轮最多验证 8 行，每次更新最多用 7 个教师位置。每个 DP 最多驻留两个请求，四个 DP 合计最多八个请求。同一 DP 每个起草 wave 服务其中一个请求。`study` 使用 4 道预热题、128 道在线题、32 道留出题，在线更新至少间隔 16 个 wave，上限为 128 次。首轮实际更新计入实测吞吐，同时记录扣除更新时间的估算。

两端设置 `PYPTO_PACK_DECODE=1` 可用 Verify 的空余行同时解码同一 DP 的其他驻留请求。每个 DP 仍只起草一个请求，其他请求各用一行，沿用独立的请求槽和 KV 页。在此模式下，短验会在 DP0、DP2 分别放两个请求。正式实验继续使用同一份 GSM8K 输入，可比对开关前后的 `decode_steps`、`packed_decode_requests`、`decode_tok_per_s` 和逐 token 对照。候选全部拒绝的宿主用例中，四个请求各生成 6 token，Decode 轮次从 10 降到 5，逐 token 输出一致。

在 111 上用 TP4 加载真实 W8A8 权重，把一条请求的三行验证和另一条请求的单行解码分别执行，再合成一个验证批。第 0 层 Attention 激活、输出头 logits 和贪心 ID 均逐元素一致。双机吞吐待使用相同 GSM8K 配置对照。 [四卡结果与复跑命令](validation/dsv41/pypto-1122-949a-packed-verify-910b3.json)

教师 logits 的设备快照已在 910B3 验证，7 个教师位置的 TV loss 和 BF16 梯度与主机采集逐元素一致。最初对照中，设备快照省下约 7.24 MB 主机往返，采集加 loss 的热态中位数为 54.58 ms，主机采集为 39.32 ms。设备快照可用 `--resident-teacher` 单独复验。 [单卡记录](validation/dsv41/pypto-1122-949a-staged-teacher-910b3.json)

默认的分块采集现用一次 CPU `aminmax` 归约检查教师 logits。111 上以相同输入交替比较，两次和七次 Verify 的采集加 TV loss 中位数分别从 39.24 ms 降到 34.86 ms、从 60.69 ms 降到 56.26 ms，loss 和 BF16 梯度逐元素一致。在线报告的 `stage_seconds` 另记录教师数据回读、有限值检查、LM head、Attention 和 QKV 等反向阶段耗时。这些计时包含调用和等待，部分阶段是嵌套的。 [单卡计时和复跑命令](validation/dsv41/pypto-1122-949a-teacher-check-910b3.json)

在 1122 镜像中创建三个新输出目录。两台使用同一份源码和相同的编译配置，产物可在相同环境之间复制。

```bash
python /opt/dsv41-1122/validate_dsv41_pypto_packed_prefill.py \
  --capacity 272 --prefill-moe-slab-rows 64 --context 512 --slots-per-dp 2 --output /results/prefill
python /opt/dsv41-1122/validate_dsv41_pypto_packed_decode_attention.py \
  --decode-capacity 16 --context 512 --slots-per-dp 2 --output /results/decode
python /opt/dsv41-1122/compile_dsv41_pypto_packed_draft.py \
  --local-ranks 8 --block 16 --decode-capacity 16 --context 512 --slots-per-dp 2 --artifact-dir /results/draft
```

两台都用已验证的 HCCL 网卡和 8 张可见设备。将 `NODE_RANK` 在 node7 设为 0、在 node8 设为 1，`RANK0_IP` 设为 node7 的 HCCL 地址。CANN 9.0 的 `HCCL_CONNECT_TIMEOUT` 合法范围为 120–7200 秒。两端使用同一份请求 JSON 和相同的 draft head。请求形如 `[{"id":"r0","dp":0,"input_ids":[42,43,44],"new_tokens":4},{"id":"r1","dp":2,"input_ids":[45,46,47],"new_tokens":4}]`。两个 DP 归属不同节点。

```bash
python /opt/dsv41-1122/run_dsv41_pypto_staged_target.py \
  --node "$NODE_RANK" --rank0-ip "$RANK0_IP" --key-file "$JOB_KEY" --checkpoint "$TARGET_WEIGHTS" \
  --prefill-graph /results/prefill --decode-graph /results/decode --draft-graph /results/draft \
  --prefill-capacity 272 --prefill-moe-tokens 64 --decode-capacity 16 --context 512 --slots-per-dp 2 \
  --draft-head "$DRAFT_HEAD" --draft-block 16 --verify-width 8 --workload /results/workload.json \
  --output "/results/compare-node-$NODE_RANK.json"
```

报告分别记录 AR 与 Orthrus 的生成 ID、起草和 Verify/Accept 阶段耗时、候选接受数、逐 token 对照，以及端到端 Decode 计时。当前这条入口覆盖固定 head 推理，已通过编译、产物回载和宿主回归。真实双机整模和 last-N 在线学习仍需验收。 [编译和宿主验证记录](validation/dsv41/pypto-1122-949a-staged-draft.json)

要检查在线更新所需的前向状态，用同一条 draft 编译命令追加 `--last-layers 4`，写到新的产物目录。启动命令使用该目录，并追加 `--capture-last-layers 4 --teacher-rows 7`。两个节点对同一条请求和 anchor 发起捕获。持有该请求的 DP 保留末层隐藏状态、词表分数和指定后缀各层的 Attention、FFN 激活及实际 MoE 路由。候选拒绝后，教师行沿修正后的前缀继续收集。报告的 `teacher_replay.passed` 检查只有一个节点收到完整教师行。 [编译与回放验证记录](validation/dsv41/pypto-1122-949a-staged-capture.json)

两台都追加 `--check-backward`，可在释放请求前执行一次 TV loss 的后缀反向。每层的专家输入和梯度通过驻留缓冲派发及返回，两个节点都计算自己持有的 W8A8 专家。请求节点按前向的 Top‑K 顺序累加输入梯度，再完成该层的 Attention 参数梯度及向上一层的反传。报告记录实际反传层、教师行、loss 和反向用时。 [反向图与实卡数值验证](validation/dsv41/pypto-1122-949a-staged-reverse.json)

要边生成边更新，在同一个 1122 环境编译一份优化器产物，并放到两台机器相同的路径。不同 last-N 共用这份产物。

```bash
python /opt/dsv41-1122/compile_dsv41_pypto_staged_adam.py --artifact-dir /results/adam
```

在上一条双机生成命令里使用编译了 `--last-layers N` 的 draft 目录，并追加 `--capture-last-layers N --adam-graph /results/adam --teacher-rows 7 --update-every 16 --max-updates 32 --learning-rate 1e-5`。首次检查可用两台各一个请求、每题生成 24 token、`--update-every 2 --max-updates 2`，让两端轮流提供教师反馈，并在发布后继续跑 draft。脚本在一次权重加载内用相同请求跑 AR 和在线 Orthrus，逐 token 比对。报告记录每轮损失、梯度范数、反向、梯度通信和暂存、Adam、发布，以及含更新和扣除更新时间的 Decode 吞吐。扣时结果是同一次在线运行的估算。真实双机整模在线结果待验证。

追加 `--eval-workload /results/heldout.json` 时，用同一份独立请求测 AR、初始 head 和更新后冻结 head。`arms` 中每组的 `wall_tok_per_s` 包含请求准入、Prefill、Decode 和释放。在线组的 `decode_tok_per_s_excluding_updates` 单列扣去更新阻塞时间的估算。

运行 `bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-summary /workspace/orthrus-dsv41-1122/study-node7.RUN/result.json` 可在宿主汇总任一节点的 1122 报告。汇总按相同请求和逐 token 输出对照 AR，分别给出整段和除首个请求批外的 Decode 吞吐，并列出各 DP 首次更新时间、后续更新中位数、发布及反向分阶段耗时。传入两个节点的报告时各自呈现，整模 token 和更新次数按同一次实验解读。

1122 可直接复用已下载的 GSM8K 分区，用同一份 checkpoint 的 tokenizer 生成实验输入。已有数据时可跳过第一条。两台使用相同的三个 JSON。

```bash
python /opt/dsv41-1122/prepare_dsv41_gsm8k.py --output /results/gsm8k-data
python /opt/dsv41-1122/prepare_dsv41_pypto_staged_gsm8k.py \
  --dataset /results/gsm8k-data --checkpoint "$TARGET_WEIGHTS" --output /results/gsm8k
```

运行 1122 生成命令时，把 `--workload` 指向 `/results/gsm8k/online.json`，再追加 `--warmup-workload /results/gsm8k/warmup.json --eval-workload /results/gsm8k/heldout.json`。默认使用 128 道在线题、32 道留出题和 4 道预热题，并以固定生成长度做吞吐对照。用 `--update-every 16 --max-updates 128` 可测更长的在线曲线。每个 DP 每批的准入上限为 `--slots-per-dp`，每个 draft wave 从该 DP 中选一个请求起草。批间释放请求状态，保留 Adam 和已发布的 head。预热和装载单独计时，报告同时给出每批进度、候选接受比例和每次请求起草平均接受数。预热先比对 AR 与固定 head 的输出，正式运行中如果出现首个 token 分叉，会立即写出阶段、请求和 token 位置。

梯度通信复用每 rank 一对 40 MiB 的驻留发送和接收缓冲，按参数大小建立视图，导出仍指向同一对分配。报告的 `gradient_scratch_bytes_per_rank` 只计这对缓冲。Adam 的 FP32 参数和动量账本另占显存，随更新层数增长。

通信域建立后，EP 派发、回传和 DP 梯度归并都先在本机验证设备缓冲，再用同一次跨机握手核对张量布局和就绪状态。宿主回归覆盖冷启动的两次握手、热路径的一次握手，以及布局不匹配时拒绝提交设备通信。

111 的四张 910B3 使用公开 W8A8 checkpoint 的第 39 层初始化 TP4 参数。构造梯度的三次发布与 Torch Adam 对照，权重抽样最大绝对误差为 `1.86e-9`。一次非有限梯度跳过发布，版本依序为 `1, 2, 2, 3`。该测试用宿主模拟 DP 求和。另用同机双卡实测 FP32 HCCL AllReduce 直接读写 IPC 缓冲视图，六轮逐元素一致。 [实卡报告与复跑命令](validation/dsv41/pypto-1122-949a-staged-adam-910b3.json)

进一步在 111 的同一容器内开两个进程，各管一张卡，用真实 root-info 建 DP2 通信域。两个进程轮流产生构造梯度，经 NPU 上的 FP32 HCCL 归并后更新和发布真实第 39 层权重。两端每次发布的权重抽样逐字节一致，与 Torch Adam 最大绝对差为 `1.49e-8`。首轮约 2.57 秒，热态有效更新约 0.57–0.61 秒。该计时覆盖梯度归并、Adam 和发布，整模反向耗时需由双机在线试验测量。复跑时容器内保持逻辑设备 0、1 都可见，两端分别将 `NODE` 设为 0、1，使用同一鉴权文件 `KEY`。 [双进程记录](validation/dsv41/pypto-1122-949a-staged-adam-910b3.json)

### 早期 Serving 后端的 GSM8K 在线更新报告

在两台公司节点使用同一份 checkout，分别执行 `run_dsv41_pypto_company_online.sh node7 report 1` 和 `run_dsv41_pypto_company_online.sh node8 report 1`。脚本使用现有 GSM8K 目录，从 `tuning.jsonl` 取 8 题在线更新，从 `quality_test.jsonl` 取 4 道留出题。默认每个 DP 分区放 1 个请求，最多 4 个并发，每题最多生成 32 个 token。两端可用相同的 `DSV41_REPORT_TRAIN`、`DSV41_REPORT_EVAL`、`DSV41_REPORT_TOKENS` 调整规模。

一次加载基模，依次跑 AR、初始固定 head、在线更新和更新后冻结 head。在线阶段只用主模型反馈，last-1 默认使用全词表 TV、学习率 `1e-5`，前两次升到目标学习率，前四次每 4 轮检查是否可收集一块反馈，之后间隔 8 轮，每次更新后发布。AR 和两组固定 head 使用相同留出题并做贪心输出对照。短输出上限会截断部分解题过程，报告记录停止和截断数。

`acceptance/gsm8k-online-report.json` 每批覆盖保存，包含 AR 和 draft 预热、含更新与发布的在线吞吐、扣除这些耗时的估算、实际冻结吞吐和接受率。首轮更新保留在在线实测中，并另外列出。完整实验结束后 `complete` 为 `true`，原始更新记录在 `acceptance/gsm8k-online-updates.json`。

如遇到 `verify/accept` 分叉，在持有该请求的节点用离线镜像读已有 logits。脚本会先比对两个输入直到待预测位置的前缀，再输出同一位置的最大误差、平均误差和两边的 Top-5。

```bash
A=$(ls -td /workspace/orthrus-cluster-accept.*/acceptance | head -1)
docker run --rm --network none -v "$A:/a:ro" \
  -v "$PWD/online-speculation/scripts:/tools:ro" orthrus-dsv41-pypto:72a62d1-offline \
  python /tools/summarize_dsv41_pypto_cluster.py /a --verifier-logits
```

若需定位当前分叉，双机启动时都加 `--trace-position 11 --tokens 3`。这里的 11 是已确认分叉的 12-token 前缀最后一个输入位置，从零开始计数。一次运行同时保留 prefill 最后一个 prompt token 和该 Verify 位置的完整子阶段证据；无论先在哪个阶段失败，`Automatic failure diagnosis` 都会给出本次运行可得到的最具体结论。该功能只读激活，不改变候选、验证或接受逻辑。

离线镜像已通过补丁和启动检查。CPU 测试覆盖全局 rank 到本机设备的映射、Engram 文件映射，以及两端通过控制连接同步起草、验证、提交和释放。真实 W8A8 双机验收曾定位到第 0 层 TP Attention 的 BF16 集合归约不稳定：四个 Wo-B 输入逐字节一致，而两次归约有 33/5120 个元素不同，最大差为 `0.015625`。当前路径将 BF16 局部结果拓宽到 FP32，执行 TP4 ReduceScatter 后再一次性舍入回 BF16；四卡连续 30 次与固定顺序 FP32 参考逐元素一致。

修复后，node7 和 node8 的 8+8 卡整模验收均通过。相同 W8A8 checkpoint 下，target-only 与 Orthrus 的 6 个贪心输出 token 逐 token 一致；Orthrus 实际执行了 2 轮 K32 起草与验证，提出 62 个候选、提交 2 个候选，并恢复 `100663296` 字节验证缓存，随后两端均完成请求释放。这证明了双机 EP16 上的 prefill、draft、verify、accept、回滚和清理链路。测试使用基模初始化 head，因此不用于判断接受率或加速收益。 [双机整模验收记录](validation/dsv41/pypto-cluster-company-ep16-acceptance.json)

## 精度验收

- SWA 和压缩 Attention 的实卡门槛按 FP32 参考设定，使用绝对误差与相对误差的组合。例如六组 TP4 压缩历史测试的输出门槛为 `atol=1e-5`、`rtol=1.3e-6`，最大绝对误差为 `5.90e-6`。Indexer 另比较 Top-K 集合，KV 验证逐字节一致。 [参考对照记录](validation/dsv41/pypto-compressed-tp4-910b3.json)
- BF16 集合归约按输入幅度与 BF16 舍入误差校验。四卡真实 head 的结果通过该门槛。两卡通信探针用可精确表示的小张量，结果要求逐元素相同。 [四卡归约记录](validation/dsv41/pypto-resident-w8a8-tp4.json)
- CANN 的精度工具按张量类型和用途配置阈值，并提供绝对误差、相对误差及余弦相似度等指标。整模 greedy 验收使用同一量化基模和相同输入逐 token 对照，也检查缓存提交与回滚。真实 EP16 整模已通过 6-token 功能验收；长序列、并发及训练 head 的性能需单独验收。 [CANN 算子精度配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_10_0091.html) · [CANN 比对结果分析](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/modelaccuracy/atlasaccuracy_16_0045.html)
