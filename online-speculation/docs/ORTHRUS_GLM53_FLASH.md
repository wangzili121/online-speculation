# Orthrus GLM 5.3 Flash 推理适配

目标是让 draft 复用 GLM-5.3-Flash 的 backbone 和已提交历史，一轮提出多个候选，由原生 vLLM-Ascend 验证并决定接受前缀。当前目标权重为 `Eco-Tech/GLM-5.3-Flash-w8a8`；不涉及训练、在线更新和多模态输入。

## 架构与算法边界

该模型有 45 层，其中 34 层为 KDA 线性注意力，11 层为稀疏 MLA；另外使用 mHC 和 MoE。它不是 DSV4 的 SWA/C4/C128 结构。

- 稀疏 MLA：draft 的每行读取同一段真实历史，以及整个临时 draft 块。历史由主模型的 KPool indexer cache 检索；临时块不经过 Top-K，全部可见。
- KDA：历史是卷积状态和递推矩阵，不能通过一个 SDPA mask 变成双向 Attention。当前实验扩展采用与 Orthrus Qwen3.5 混合模型相近的方式：复制注意力参数，从真实历史快照出发，在临时状态上因果递推。

第二项是 **GLM 混合模型的实验定义**，不是原始普通 Qwen3 算法已提供的 GLM 实现。默认策略为 `experimental_causal_kda_bidirectional_mla_v1`，入口要求显式实验确认。

同一 proposer 也提供实验性的门控双向 KDA：正向读取 AR 状态，反向只遍历临时块、从零状态开始，两路在输出归一化之前融合。通过 `--kda-mode bidirectional` 选择，head 标记为 `experimental_residual_bidir_kda_bidirectional_mla_v1`。这不是 SDPA 换 mask，也不是已确认优于因果 KDA 的训练方案；方向性测试及三组小规模蒸馏对照见 [KDA 方向选择](GLM53_KDA_DIRECTION.md)。

## 算子需求

下表是需要适配的计算入口，不代表必须开发三个独立新 kernel。现有算子能组合完成时，优先组合。

| 模块 | 当前已有能力 | 适配内容 | 当前实现 |
| --- | --- | --- | --- |
| Diffusion KPool 检索 | PyPTO `models/glm5_3_flash/indexer.py` 已有投影、池级打分、Top-K、token 索引展开和未满池尾部处理；正常 AR 路径还会生成 key、压缩和写 cache。 | 接收 draft query 与 head weights，只对 anchor 前的完整池打分；展开为原始 token 位置，并追加未满池的真实历史。不要调用写 cache 的完整 AR 入口。 | `select_history` 用 `einsum → ReLU → 加权求和 → topk → 索引展开` 实现。历史池由引擎读取，不在 draft 中重新压缩。 |
| Diffusion 稀疏 MLA | PyPTO `attention.py` 已有低秩 query 吸收、稀疏历史 Attention 和输出投影；现有 decode 面向 AR 的查询与 cache 生命周期。 | 增加当前块的双向分支，固定历史边界。历史使用 AR 的 KV 展开权重，临时块使用 draft 权重，按共同 softmax 分母合并。 | 两次 NPU FIA 分别返回输出和 LSE；转换到相同 value 空间后用 `npu_attention_update` 合并。不展开整段历史的每头 K/V。 |
| KDA 临时状态执行 | PyPTO `kda.py` 已有投影、短卷积、因果递推和输出；这些计算会更新传入的状态。 | 在上述实验策略下，算子仍做因果 KDA，只把状态参数换成当前请求的临时副本。主模型状态的选取、复制和接受位置管理属于引擎适配。 | draft 使用现有 AscendC recurrent KDA。当前原生 GLM 路径最多处理 8 个位置，首轮接入采用 K=8；没有宣称 K=32 整模型已支持。 |

Embedding、mHC、norm、MoE、LM head 和 TP collective 仍调用主模型已有实现，不复制第二套 backbone。目标模型的 MTP 不作为 Orthrus draft 使用。

## 稀疏 MLA 的数学契约

设第 $i$ 个 draft query 为 $q_i$，主模型低秩历史为 $c_j^{AR}$，临时块低秩表示为 $c_t^D$。$W_K^{AR},W_V^{AR}$ 和 $W_K^D,W_V^D$ 分别表示主模型与 draft 的低秩展开权重；下式省略 head 下标。

历史 token 集合为 $\mathcal H_i$，临时块为 $\mathcal B$。正确的输出是：

$$
o_i=\frac{
\sum_{j\in\mathcal H_i} e^{q_i^\top W_K^{AR}c_j^{AR}/\sqrt d}\,W_V^{AR}c_j^{AR}
+\sum_{t\in\mathcal B}e^{q_i^\top W_K^Dc_t^D/\sqrt d}\,W_V^Dc_t^D}
{\sum_{j\in\mathcal H_i}e^{q_i^\top W_K^{AR}c_j^{AR}/\sqrt d}
+\sum_{t\in\mathcal B}e^{q_i^\top W_K^Dc_t^D/\sqrt d}}.
$$

不能将两种 latent 简单拼接后统一乘 draft 的 KV 展开权重，除非已经约束两套展开权重完全相同。当前实现对两段分别吸收 query，取得局部输出 $o_{i,r}$ 和自然对数 LSE $L_{i,r}$，再计算

$$
L_i=\operatorname{logaddexp}(L_{i,AR},L_{i,D}),\qquad
o_i=e^{L_{i,AR}-L_i}o_{i,AR}+e^{L_{i,D}-L_i}o_{i,D}.
$$

这与显式展开历史及块内 K/V 后做一次 Attention 等价。空历史只计算临时块。

## 推理引擎接入

`GLM53OrthrusProposer` 使用 `custom_class` 入口，复用同步 propose/verify 的请求记账和原生 rejection sampler。原生目标模型、验证时的 cache 写入和接受前缀处理保持不变。

每轮验证结束后，根据本轮实际接受位置读取历史：KDA 选择对应的递推状态及卷积尾部；MLA 读取有效分页 latent 和完整历史池。然后构造 `anchor + MASK`，执行 draft。整个 draft 不写目标 cache；候选的 draft 状态不会“转正”，正式状态始终由目标模型验证产生。

draft 只持有注意力投影和内部归一化等参数。执行时临时替换各层 Attention 的调用，仍使用原模型完成 embedding、mHC、MoE、norm 和输出头；`finally` 恢复 AR 调用。该方法仅用于 eager、同步、单请求实验，不用于并发或图模式。head 文件必须匹配基模配置、量化描述、TP 布局、K、mask token 和混合策略。

## 验证与启动

验证分为数学与适配器回归、单卡 NPU 算子检查、原生引擎闭环三层。114 项定向回归通过，覆盖 Attention 与显式 SDPA 等价、双向信息流、KPool 边界、接受位置的状态选取、分页寻址、head 契约、异常恢复，以及启动环境检查。NPU 检查覆盖 MLA 的 K=2 至 8 及 K=32、KDA 的 K=1 至 8；MLA 对 FP32 的相对 RMS 误差约 0.3%，KDA 输出及逐位置状态的相对 RMS 误差不超过 1.2e-5。

原生引擎测试使用源权重的前 5 层和前 9 层，保留原始宽度、W8A8 权重、mHC 及 dense/MoE。前 5 层有 4 层 KDA、1 层 MLA；前 9 层有 7 层 KDA、2 层 MLA，用于继续检查多层历史状态和 TP 分片。它们不是随机缩小网络，也不是官方小模型；截层后的回答不用于语义或加速收益评测。

| 实测范围 | 结果 |
| --- | --- |
| 5 层，单卡，K=8，两种 KDA 模式 | 每种均有 96 个 greedy token 与 target-only 相同；真实整块验证、接受/拒绝、缓存审计、head 保存及重载通过。双向分支使用非零门控。 |
| 5 层，单卡，K=2/4，两种模式 | Greedy、重载和 temperature=0.7 采样通过；K=2 实际经历了整块接受，也覆盖部分接受及拒绝。 |
| 5 层，双卡，K=8，两种模式 | Greedy 对齐；各 rank 的 head 参数重载哈希一致，计数一致；两种模式的原生 rejection sampling 均运行通过。 |
| 5 层，单卡，4911-token 前缀 | 两种模式均 greedy 对齐，覆盖缓存跨页和超过 Top-K 上限的稀疏检索；因果版本另通过该长前缀的重载及采样。 |
| 单卡，加载后投影权重生效 | 实测 KDA QKV/O、MLA QKV/O、indexer Q 五类原生投影：修改 head 权重会改变输出，恢复后固定输入的误差均为 0，主模型权重和投影输出不受影响。 |
| 111 物理 0，5 层，单卡，K=8，两种模式 | 调整容器访问与网络配置后重新跑通 target-only、因果及双向 draft；每种 96 个 greedy token 与本机基线一致，缓存审计分别通过 53/56 轮。双向模式实际执行 224 次反向 KDA。 |
| 9 层，四卡，K=8，两种模式 | 起草、验证、缓存审计、四分片 head 保存/重载和采样通过。双向初始化运行每 rank 378 次反向 KDA，重载后运行 385 次。Greedy 验收尚未通过，见下文。 |
| 完整 45 层，八卡 | 已在 111 节点实际尝试 target-only；worker 在导入原生 `camem` 扩展时中止，尚未进入 Orthrus，也未验证完整模型的显存和输出。 |

9 层四卡的逐 token 验收暂不能通过：两个相同配置、固定种子的 target-only 运行本身不一致；不加载神经 head、不安装本项目 hooks 的原生 n-gram 对照也出现差异，实际进行了 12 轮投机、接受 18 个候选。因此差异并非只在神经 draft 路径出现，但这不证明适配器不存在其他问题，也不能把该项标记为通过。采样跑通和重载哈希一致只证明相应功能，不替代分布等价验收。

每轮审计逐字节比较当前请求的有效 MLA latent、indexer 池和尾部环形缓存、KDA 卷积及验证 checkpoint。测试实际经历了整块接受、部分接受、拒绝和多条请求依次复用引擎。审计检查的是 draft 前后的状态，不覆盖未分配给当前请求的整个缓存池，也不代替验证器的提交/回滚正确性或并发隔离验收。

联调中修复了插件配置被后端严格校验拒绝、NoPE 空 RoPE 占位未识别，以及带 padding 的大缓存被整体物化等适配问题。后者已有 NPU 内存回归：复制 512 KiB 的选中状态，额外峰值为 513 KiB。单卡测试使用实测通过的 34816 上限，双卡和四卡使用 18432；不将这些测试参数作为任意 TP 配置的通用容量公式。

111 的完整权重已逐文件校验 SHA256。该节点原生 TP8 两次在 `camem.py` 导入 `vllm_ascend_C` 时出现 `basic_string::_S_construct null not valid`，当时尚未加载模型。后续仅调用 CANN 的单卡对照确认：默认隔离容器访问 0 号卡时，`acl.init()` 返回 500000，设备枚举返回 507899，芯片名为空；内核同时记录 `Conflict open udevid`。同镜像换到 3 号卡正常，因此不能将该异常归因于模型大小或 draft Attention。

`camem` 是 sleep/wake 使用的设备内存池模块，worker 即使未开启 sleep 也会导入它。当前处理不是删除该模块或屏蔽异常，而是先检查 CANN 初始化、可见设备数量和芯片信息；失败时保存错误并退出，不再进入会中止进程的原生扩展导入。在 111 的显式特权测试容器内，已确认使用的物理卡为 0，小张量计算、worker 导入以及 MLA/KDA/缓存预检均通过。这是绕过该节点容器访问冲突的启动方式，不代表已清理冲突来源、修复驱动，或完成八卡模型验收。

继续加载 target-only 时还发现该节点自动选择通信网卡会使 `hcclGetRootInfo` 报错 19。独立的单 rank 对照也复现；显式指定业务网卡及 IP 后，communicator 创建和 AllReduce 通过。脚本现在透传启动环境中的 `HCCL_SOCKET_IFNAME`、`HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`VLLM_HOST_IP` 和 `TP_SOCKET_IFNAME`，不在代码内硬编码节点地址；预检与模型使用同一组网络配置。预检增加真实 HCCL communicator 创建和单 rank AllReduce，避免把张量分配成功等同于通信可用；这仍不代替 TP8 的多 rank 通信验收。

结果、参数和代码哈希保存在 `docs/validation/glm53/`；`native-5l-*` 为五层测试，`native-9l-tp4/` 与 `native-9l-repeat/` 保留四卡差异及独立对照，`native-45l-tp8/` 和 `native-device0/` 保留原生启动失败记录。`native-device0-access/` 记录 CANN/HCCL 对照及启动处理，`native-5l-device0-validated/` 是在 111 物理 0 上重跑通过的模型测试。测试容器在结束或失败后删除，不提交模型或 head 权重。

单卡算子检查：

```bash
PYTHONPATH=src python scripts/probe_glm53_ops.py --output artifacts/operators.json
```

原生闭环先运行 `prepare_glm53_runtime.py`，再用 `validate_glm53_native.py` 指定物理卡、模型和输出目录；它依次完成基线、两种起草、重载及采样，任一阶段不通过则停止。`--block-size`、`--max-model-len` 和 `--long-prompt-repeats` 指定测试档位。`run_glm53_vllm.py --mode ngram` 提供不加载本项目 head/hooks 的独立原生对照；`--check-head-replacement` 只在明确加载 head 后启用验证专用 worker 扩展，使用具名 RPC，不放开函数序列化。

八卡准备使用 `prepare_glm53_runtime.py --all-devices --retry-seconds 300`：等待八卡空闲，检查 CANN、原生 worker 导入、每张卡的小张量计算、单 rank HCCL 建链及 draft 算子；失败后先释放容器，再等待重试。`watch_glm53_validation.py` 在完整权重校验、八卡预检和资源检查通过后，依次运行 target-only、两种 draft、两种 head 重载和两种采样，并核对各 rank 的 head 哈希。每次监控使用独立结果目录。单卡预检不能授权八卡任务；源码或网络配置变化后必须重新预检。源码变化后还须重启验收进程。本地文件锁只协调本项目，不预留其他用户的资源。

模型和代码始终只读挂载，容器在独立的可写缓存目录运行，Torch、Triton 和 CANN 按需编译产物留在该目录。这样新增输入形状或检查算子不需要向源码目录写入文件。

默认仍使用设备隔离容器。对受信任、已协调资源的测试节点，可在准备、验证和监控命令中同时显式加入 `--privileged-runtime`；预检记录必须与实际启动模式一致。该选项会传递 Docker `--privileged`，扩大宿主权限并使全部设备可见，不适合作为多租户默认配置，也不能替代卡的分配。程序仍只计算指定的可见设备，拒绝特权模式下的非恒等设备重排，八卡任务仍须等八卡空闲。生产环境应优先协调独占分配并解除冲突，不停止其他用户容器、不重置设备。

各节点可独立运行 `watch_npu_resources.py --output /path/to/status.json`，每 30 秒记录一次空卡和心跳，发现完整八卡窗口后继续监控。它仅查询资源，不预留设备或启动模型；进程列表为空但 HBM 仍高占用的卡也不会被判为空闲。资源监控和模型测试是两件事，某个节点具备空卡后，仍须在该节点准备并校验权重及运行镜像。

**当前五层模型通过了 eager、同步、单请求、K=2/4/8 的相应推理测试；九层四卡通过功能链路，但 greedy 尚未验收；完整 45 层 TP8 尚未验收。** 并发、图模式和训练/在线更新未接入。采样分布统计与真实训练 head 的性能需另行评测，不能由初始化 head 或截层模型推断实际加速收益。

参考：[GLM-5.3-Flash 官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)、[Ascend 部署说明](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.3-Flash.html)、[PyPTO GLM 模型目录](https://github.com/hw-native-sys/pypto-lib/tree/main/models/glm5_3_flash)。
