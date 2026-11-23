# Orthrus-DSV4.1 GSM8K 在线更新

本实验回答两个不同问题：从基模 Attention 复制初始化后，在线更新是否能改善接受率；短预热后继续在线更新，能否在计入训练和发布开销后提升端到端吞吐。Qwen3.8 的有效配方只作为扫描起点，DSV4.1 的最终参数由独立 tuning 集确定。

## 数据与对照

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-prepare \
  /workspace/data/orthrus-dsv41-gsm8k
```

脚本从 GSM8K 官方仓库下载并校验固定 SHA256，生成互不重叠的 128 题预热集、128 题调参集、384 题在线评测集和官方质量测试集。标准答案只用于最后的正确率统计；draft 更新只使用主模型在真实修正前缀上的 logits，避免把答案标签混入在线教师反馈。

内网机器不能联网时，可在联网机器保存官方的 `train.jsonl`、`test.jsonl`，把目录作为第二个参数传入；脚本仍执行同一组 SHA256 校验。失效的 `127.0.0.1/localhost` 隧道代理会自动绕过，不影响普通公司代理。

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-prepare \
  /workspace/data/orthrus-dsv41-gsm8k /workspace/data/gsm8k-source
```

四个实验臂使用相同题目顺序和 greedy 生成：target-only、固定初始化 head、初始化 head 直接在线更新、短预热 head 继续在线更新。每个在线实验重新初始化优化器，训练和权重发布时间计入总耗时。配置见 [`gsm8k-online.json`](../configs/dsv41/gsm8k-online.json)。

## 在线监督

推理仍使用 K32。一次更新默认最多监督 7 个未来位置，但拒绝后的行不能直接使用：系统保留原 anchor 的 draft 快照，沿主模型修正后的真实前缀继续收集验证分布，直到收齐 7 行或请求结束。last-N 只更新连续后 N 层的 diffusion Attention；W8A8 主模型、indexer、共享 mHC、MoE、Norm 和 LM Head 均保持冻结。首轮在相同题目顺序下比较 N=1/2/4/8，不预设 Qwen3.8 的 last-1 也是 DSV4.1 最优值。

Torch 参考可对最后一层做可微重放、TV/KL 更新和 head 导出；通用及 C1A/mHC 的 PyPTO 反向叶子、last-N 连续重放契约和双机驻留双槽发布也已准备。默认候选为 `lr=1e-5`、8 次线性 warmup、前 256 次更新每 16 个 verifier round 更新一次，之后每 128 轮一次、每 4 次更新发布。`last_layers`、`teacher_rows`、学习率和频率均须在 tuning 集扫描，不能把 Qwen3.8 的数值直接当作 DSV4.1 结论。

## 双机执行

先在两台机器各刷新一次本地镜像。该命令不访问网络，也不重编 PyPTO/CANN；它只把当前仓库的专家状态 ABI 和 Python 编排叠加到已经固定的镜像上：

```bash
PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline \
  bash online-speculation/scripts/dsv41_pypto_env.sh refresh-image
```

先在任一空闲卡运行单机 readiness gate。它会执行全部 DSV4.1 回归、在固定 serving 源码上实际应用 online patch 并做 Python 编译，然后复验 129280 维完整词表 TV/KL、投影输入与参数梯度、RMSNorm 前反向、C1A Attention、mHC、MoE、AdamW 和发布 kernel：

```bash
NPU_DEVICE=0 PYPTO_VISIBLE_DEVICE=0 PYPTO_RUNTIME_DEVICE=0 \
  bash online-speculation/scripts/dsv41_pypto_env.sh online-ready
```

当前验收基线为：158 项 DSV4.1 测试、4 项跳过和 5 组子测试通过，22 个设备数值用例通过，61 个生产形状程序完成编译；其中包含 TP4/K32 的逐 query C1A 历史重放、sink、临时块 K/V 梯度，K32×5120 的 mHC 反向，以及固定 Top-6 路由、冻结 W8A8 专家输入 VJP、EP16 编排、AdamW 和发布。inactive head bank 的设备到设备发布也已实卡通过，完整 head 不经 CPU 往返。`readiness.json` 汇总测试、serving 补丁和实卡 kernel 证据，并保留为宿主用户可读；它不替代双机 serving 中的一次完整在线更新。

先在两台机器放置相同的 `tuning.jsonl`。固定 head 推理链路已经通过双机验收；在线更新应直接使用统一入口同时启动 4 题、每题 32 token 的 smoke。它会加载在线策略，执行真实反馈收集、last-N 反向、同步更新与安全发布：

公司环境的镜像、模型、数据、八卡列表、结果目录和密钥路径统一保存在一个入口中。两边只需同时运行短命令：

```bash
# node7
bash online-speculation/scripts/run_dsv41_pypto_company_online.sh node7 smoke 1

# node8（与 node7 同时执行）
bash online-speculation/scripts/run_dsv41_pypto_company_online.sh node8 smoke 1
```

公司入口按节点选择同一份密钥的实际位置：node7 使用 `/root/.orthrus-cluster.key`，node8 使用 `/home/wangzili/.orthrus-cluster.key`。

结果目录会在启动时打印。任一节点可汇总 Orthrus 结果：

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-summary \
  /workspace/orthrus-cluster-accept.XXXXXX/acceptance/orthrus
```

在线版本必须额外满足三件事才允许扩大样本：保存真实修正前缀上的完整教师 logits；从同一 anchor 重放最后一层并完成有效梯度更新；TP4/DP4 在 verifier 轮次边界原子切换同一 head 版本。缺少其中任何一项都只能称为 fixed-head 推理。

## 验收顺序

1. 固定反馈重放：loss 下降、仅配置的 last-N draft 参数变化，教师、indexer 和主模型参数不产生梯度。
2. 16～32 题 smoke：确认延迟教师行连续、版本发布原子、greedy 输出与 target-only 相同。
3. tuning 集选择配置：先看接受长度是否提高，再看计入更新后的吞吐；接受率不动时先排查反馈、梯度与初始化，不盲目增加更新频率。
4. 384 题三种顺序正式比较。分别报告 AR、固定 head、直接在线和预热在线的总 token/总时间、接受长度、训练及发布成本；训练 head 的质量与加速结论不得由初始化 head 的 smoke 替代。
