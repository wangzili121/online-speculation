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

整模还需把本机权重和缓存接进 `GenerationSession`，按本机 rank 分配运行时张量，并把 `NetworkExecutor.exchange` 接到跨机通信。编译时保留全局 EP16，空 DP 也参与集合通信。接好后，使用真实 W8A8 基模的同一组输入跑 target-only 和 Orthrus，由[整模验收](ORTHRUS_DSV41_PYPTO.md#serving-入口与实机验收)检查实际投机阶段和逐 token 一致性。

## 精度验收

- SWA 和压缩 Attention 的实卡门槛按 FP32 参考设定，使用绝对误差与相对误差的组合。例如六组 TP4 压缩历史测试的输出门槛为 `atol=1e-5`、`rtol=1.3e-6`，最大绝对误差为 `5.90e-6`。Indexer 另比较 Top-K 集合，KV 验证逐字节一致。 [参考对照记录](validation/dsv41/pypto-compressed-tp4-910b3.json)
- BF16 集合归约按输入幅度与 BF16 舍入误差校验。四卡真实 head 的结果通过该门槛。两卡通信探针用可精确表示的小张量，结果要求逐元素相同。 [四卡归约记录](validation/dsv41/pypto-resident-w8a8-tp4.json)
- CANN 的精度工具按张量类型和用途配置阈值，并提供绝对误差、相对误差及余弦相似度等指标。整模 greedy 验收使用同一量化基模和相同输入逐 token 对照，也检查缓存提交与回滚。真实 EP16 的全模型结果仍待测。 [CANN 算子精度配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_10_0091.html) · [CANN 比对结果分析](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/modelaccuracy/atlasaccuracy_16_0045.html)
