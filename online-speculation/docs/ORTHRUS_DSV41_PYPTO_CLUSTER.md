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

程序先运行 target-only 再运行 Orthrus，在两个节点的 DP 分区各运行一个请求。两次生成复用同一份已加载的基模和通信组，各自重新分配并清理请求缓存。起草前先比较 prefill 生成的第一个 token，之后每轮验证立即比较新增 token。默认记录 prefill 每层激活的散列值。首 token 分叉时报告会附上本端最早的变化和候选分数，验证阶段分叉时会附上候选和目标判定。`--tokens 4` 可快速覆盖前四个输出位置，两端需传相同参数。两次独立冷启动的精度一致性仍需另起进程验证。

程序检查全部 40 层、逐 token 的贪心一致性、K32 起草和验证、接受前缀提交、验证后的 KV 恢复及请求释放。两台各自记录 `orthrus-cluster-accept.*/acceptance/acceptance.json`，详细日志为同目录的 `cluster-accept.log`。先看两端的 `passed`，再看 `compared_tokens`、`draft_rounds` 和 `verifier_restored_bytes`。此入口用于离线推理验收。初始化 head 的接受数供诊断使用，性能结论需用训练 head。

已有验收结果可在任一节点用下列命令汇总，无需重新运行模型。报告区分各阶段首次调用与后续中位数，并附上本机的验证缓存搬运量、draft 权重上传次数和耗时。首次调用可能包含编译，计时也含本次验收的诊断开销。

```bash
python3 online-speculation/scripts/summarize_dsv41_pypto_cluster.py /workspace/orthrus-cluster-accept.RUN/acceptance
```

如遇到 `verify/accept` 分叉，在持有该请求的节点用离线镜像读已有 logits。脚本会先比对两个输入直到待预测位置的前缀，再输出同一位置的最大误差、平均误差和两边的 Top-5。

```bash
A=$(ls -td /workspace/orthrus-cluster-accept.*/acceptance | head -1)
docker run --rm --network none -v "$A:/a:ro" \
  -v "$PWD/online-speculation/scripts:/tools:ro" orthrus-dsv41-pypto:72a62d1-offline \
  python /tools/summarize_dsv41_pypto_cluster.py /a --verifier-logits
```

离线镜像已通过补丁和启动检查。CPU 测试覆盖全局 rank 到本机设备的映射、Engram 文件映射，以及两端通过控制连接同步起草、验证、提交和释放。真实 W8A8 双机已分别完成 target-only 和 Orthrus 生成，首次验收发现两次独立基模 prefill 的 logits 有差异。共享基模会话的双机逐 token 结果仍需复验。

## 精度验收

- SWA 和压缩 Attention 的实卡门槛按 FP32 参考设定，使用绝对误差与相对误差的组合。例如六组 TP4 压缩历史测试的输出门槛为 `atol=1e-5`、`rtol=1.3e-6`，最大绝对误差为 `5.90e-6`。Indexer 另比较 Top-K 集合，KV 验证逐字节一致。 [参考对照记录](validation/dsv41/pypto-compressed-tp4-910b3.json)
- BF16 集合归约按输入幅度与 BF16 舍入误差校验。四卡真实 head 的结果通过该门槛。两卡通信探针用可精确表示的小张量，结果要求逐元素相同。 [四卡归约记录](validation/dsv41/pypto-resident-w8a8-tp4.json)
- CANN 的精度工具按张量类型和用途配置阈值，并提供绝对误差、相对误差及余弦相似度等指标。整模 greedy 验收使用同一量化基模和相同输入逐 token 对照，也检查缓存提交与回滚。真实 EP16 的全模型结果仍待测。 [CANN 算子精度配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_10_0091.html) · [CANN 比对结果分析](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/modelaccuracy/atlasaccuracy_16_0045.html)
