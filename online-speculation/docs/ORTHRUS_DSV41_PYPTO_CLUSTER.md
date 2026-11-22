# DSV4.1 Orthrus 双机调度和通信

目标布局为两台各 8 张 NPU，DP4/TP4/EP16。每台持有两个完整 TP4 分区。节点 0 持有全局 rank 0–7，节点 1 持有 rank 8–15。本机设备编号与全局 rank 分开。TP 在节点内通信，DP 按同一 TP lane 跨节点，MoE EP 覆盖全部 16 个 rank。

调度端发送 Serving 的 DP_engines 请求波次。工作端保留本节点负责的全局 DP、request slot 和 generation，空分区仍保留。两端依次确认 Prefill、Draft、Verify、Accept、Decode 和 Release。Draft 和 Verify 保持 epoch，Accept 成功后才推进。任一侧失败，当前会话进入失败状态。需要返回结果时，控制连接只汇总带请求归属的 token ID 行。

设备通信复用配对 PyPTO Lib 的 HCCLBytes 和 resident tensor 接口。跨机组通过 HcclGetRootInfo 和 HcclCommInitRootInfo 建立，按全局 rank 组织 HBM 上的 AllToAll、AllGather、ReduceScatter 和 AllReduce。通信前核对对端字段顺序、形状和 dtype。

## 验证入口

在两台机器使用同版源码。为本次作业生成一个密钥文件，通过已授权的文件传输方式放到两端。两端设置同一个 rank 0 内网 IPv4、端口和版本标识。用 `--control-only` 可先验证 8+8 布局、阶段推进和 token 行回收。

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

公司内网 node7 和 node8 各使用一张卡，完成 HCCL 建链、AllReduce 和 AlltoAll。两端的会话 ID 一致，数据与参考逐元素一致。实际 8+8 卡的 EP16 硬件测试仍待执行。 [公司双机记录](validation/dsv41/pypto-fabric-company-ep2.json)

8+8 逻辑 rank 的控制流程通过了双进程 TCP 回环。外部实验机上的单机双卡测试覆盖了驱动 25.5.1/CANN 9.0.0 与驱动 26.1.1/CANN 9.1.1。三卡 2+1 进入 EP AlltoAll 后报设备链路超时 `507048`，相同卡两两连接的用例通过。 [单机两卡记录](validation/dsv41/pypto-cluster-loopback-ep2.json)

完整模型 Serving 仍通过单进程创建所有 rank。接入时，`GenerationSession` 下的权重、KV 页和运行时张量按本机 rank 分配。编译仍保留全局 EP16，MoE 专家归属和收发桶按全局 rank 编号。层循环通过 `NetworkExecutor.exchange` 接到新通信层，两端执行同一阶段序列，空 DP 仍参与 collective。真实 W8A8 基模用同一组输入跑 target-only 和 Orthrus，再由[整模验收](ORTHRUS_DSV41_PYPTO.md#serving-入口与实机验收)检查实际投机阶段和逐 token 一致性。

## 精度验收

- SWA 和压缩 Attention 的实卡门槛按 FP32 参考设定，使用绝对误差与相对误差的组合。例如六组 TP4 压缩历史测试的输出门槛为 `atol=1e-5`、`rtol=1.3e-6`，最大绝对误差为 `5.90e-6`。Indexer 另比较 Top-K 集合，KV 验证逐字节一致。 [参考对照记录](validation/dsv41/pypto-compressed-tp4-910b3.json)
- BF16 集合归约按输入幅度与 BF16 舍入误差校验。四卡真实 head 的结果通过该门槛。两卡通信探针用可精确表示的小张量，结果要求逐元素相同。 [四卡归约记录](validation/dsv41/pypto-resident-w8a8-tp4.json)
- CANN 的精度工具按张量类型和用途配置阈值，并提供绝对误差、相对误差及余弦相似度等指标。整模 greedy 验收使用同一量化基模和相同输入逐 token 对照，也检查缓存提交与回滚。真实 EP16 的全模型结果仍待测。 [CANN 算子精度配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_10_0091.html) · [CANN 比对结果分析](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/modelaccuracy/atlasaccuracy_16_0045.html)
