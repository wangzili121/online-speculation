# GLM-Flash Draft 的 KDA 方向选择

**推理接入先保留因果 KDA，MLA 使用块内双向 Attention；不把双向 KDA 当作接入前置条件。**

这里选择的是新增 draft 分支的结构，主模型始终保持原生因果计算。目前只有从 AR 参数复制初始化的 draft，没有训练好的 GLM head。本次实验用于检查信息流和实现成本，不能代替真实 GLM 的训练及吞吐评测。

## 为什么不要求每层双向

整块起草的输入是已经给定的 `anchor + MASK`。KDA 虽然按位置递推，但不等待前一个候选 token 被采样；一次前向仍能产生整块候选分布。它不是逐 token 自回归起草。

同时，双向 MLA 可以先把块尾特征传到块首，再交给后续的因果 KDA。因此，某一层 KDA 是因果的，不代表整个 draft 只能从左往右传递信息。受控扰动测试验证了这一点：保留所有 KDA 的因果方向，只改变临时块末尾输入，经过 MLA 和后续 KDA 后，块首输出也会改变。这个测试只检查信息通路，实际 draft 不会读取未来答案。

Orthrus 的 Qwen3.5 参考实现也在 diffusion 分支中保留了 GDN 的因果卷积和递推，只有普通 Attention 使用双向块 mask。它提供了混合结构的参考，但不能证明 GLM KDA 的最佳设计已经确定。

## 对照结构

两种结构都使用相同的双向 MLA、同一份冻结 backbone 和主模型历史。区别仅在 draft KDA：

| 结构 | 计算方式 | 初始化与状态边界 |
| --- | --- | --- |
| A：因果 KDA | 从 AR 历史状态出发，对当前块做一次正向卷积和 KDA 递推。 | 注意力参数复制自 AR；历史只读，临时状态不会提交。 |
| B：门控双向候选 | 保留 A，再增加一次当前块的反向卷积和反向 KDA，融合两路输出。 | 反向分支从零状态出发，不反向读取 AR 历史；两路共享参数，新增每头一个零初始化门控。 |

设两路 KDA 核心输出为 $O_i^{\rightarrow}$ 和 $O_i^{\leftarrow}$，第 $h$ 个头的新增标量为 $\gamma_h$，B 在输出归一化及投影之前计算：

$$
O_{i,h}^{B}=O_{i,h}^{\rightarrow}+\tanh(\gamma_h)O_{i,h}^{\leftarrow},
\qquad \gamma_h=0\ \text{at initialization}.
$$

所以零门控初始化时，两组第零步的 logits 完全相同。B 是一个明确的研究候选，不是唯一或标准的“双向 KDA”；其反向分支也不是修改 SDPA mask 得到的。

目前 B 已作为显式实验选项接入与 A 相同的 proposer，使用独立的 `experimental_residual_bidir_kda_bidirectional_mla_v1` 策略标识；默认仍为 A。`--kda-mode bidirectional --kda-backward-weight 0.25` 用于初始化 head 的推理联调，非零系数确保测试确实执行反向通路。外部 head 必须携带自己的门控参数，不能在加载时用此初始化选项覆盖。两种 head 的策略不匹配会被拒绝；具体引擎验收范围见 [推理适配](ORTHRUS_GLM53_FLASH.md)。

## 实测结果

### NPU 核心计算

在 A2 上，用当前 vLLM-Ascend 的原生 KDA 接口测试 K=8、每卡 8 个头、head_dim=128。输入为随机张量，没有加载 GLM 权重；B 的反向融合系数固定为 0.5，以检查非零反向通路。

| 检查项 | A：因果 KDA | B：门控双向候选 |
| --- | --- | --- |
| 与参考递推的数值核对 | 通过 | 通过 |
| 改变块尾 V，块首输出是否变化 | 不变化 | 变化 |
| 原始历史状态 | 未改写 | 未改写 |
| 核心适配调用的中位耗时 | 0.552 ms | 1.452 ms |

耗时包含 Python 适配、临时状态及单路或双路递推；不包含完整投影、两路短卷积、归一化、MoE 或验证。双路组合耗时约为单路的 2.63 倍，但不能据此推断整个 draft 或服务慢多少；它只说明反向信息不是免费获得的。测试采用 5 次预热、5 组各 50 次调用，每组末尾同步 NPU。

### 小规模蒸馏

使用自包含的合成混合网络：4 层 KDA、1 层 NoPE 低秩 MLA，hidden_size=32，K=8。没有 GLM 权重、mHC、MoE 或量化路径，不能把它称为真实 GLM 小模型验收。

数据是 31 种起点和 4 种步长组成的模运算 token 序列，长度 24；96 组训练、28 组留出验证。每个 seed 先训练 AR 老师 300 步，再冻结老师及共享参数，复制注意力参数，构造 anchor+MASK，用完整词表 KL 蒸馏 200 步。两组使用相同初始化、batch 和数据顺序；预算是相同步数，不是相同计算量。验证数据不参与更新。

下表均为留出集指标，KL 和 TV 越低越好；top-1 一致性越高越好。

| Seed | A：KL | B：KL | A：TV | B：TV |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1.4695 | 1.4567 | 0.52548 | 0.52703 |
| 19 | 1.4552 | 1.4303 | 0.52808 | 0.52292 |
| 31 | 1.0801 | 1.0893 | 0.44664 | 0.44875 |
| 均值 | 1.3349 | 1.3254 | 0.50007 | 0.49956 |

两组初始化 KL 均值均为 3.9857，训练后均有下降。B 的平均 KL 比 A 低约 0.71%，但 TV 在三组中有两组更差；平均 teacher-forced top-1 一致性为 A 64.74%、B 64.68%。三组短实验不足以支持稳定的质量收益，更不能证明所有双向 KDA 方案无效。

这些指标使用真实序列的 teacher-forced 条件分布，不是完整投机解码的接受率。尤其不能把逐行 top-1 一致性当作连续接受长度。

## 接入决定

当前选择 A：它保留已有原生 KDA 路径，具备整块起草及通过 MLA 交换块内信息的能力；本次没有发现足以支持默认增加反向分支的证据。它仍是 GLM 的实验适配，不是宣称与原始纯 Attention 模型逐层等价。

若继续评估 B，需要在真实 GLM 上训练两个结构一致的 head，再比较连续接受长度、draft/verify 时间和端到端 goodput。判断标准是新增接受收益能否抵消额外 draft 时间，不是哪个结构看起来更“双向”。主模型和精确验证器不因此改变。

## 复现与证据

脚本：`benchmarks/glm53_kda_direction.py`。完整代码 SHA256 为 `1a552261d1597b9b33157afaf00514d9fa97bec3662c4557073f63ddbf9cd9ed`。

```bash
PYTHONPATH=src TORCH_DEVICE_BACKEND_AUTOLOAD=0 python benchmarks/glm53_kda_direction.py \
  --case train --seeds 7 19 31 --teacher-steps 300 --draft-steps 200 \
  --output artifacts/kda-direction-train.json

PYTHONPATH=src python benchmarks/glm53_kda_direction.py \
  --case npu --repeats 50 --output artifacts/kda-direction-npu.json
```

NPU 镜像固定为 `quay.io/ascend/vllm-ascend@sha256:61584d790fa3668e7c43f0eb50a3e3a86d94b3eed15bce5ac82391ed383df229`。CPU 蒸馏使用 Torch 2.10.0、单计算线程。

原始结果在 `docs/validation/glm53/kda-direction-train.json` 和 `kda-direction-npu.json`；信息流、初始化一致性、缓存前向一致性和冻结参数反向检查在 `tests/test_glm53_kda_direction.py`。含相关回归共 68 项定向测试通过。单卡测试结束后容器已删除，未保留占卡进程。

参考：[Orthrus Qwen3.5 混合模型源码](https://github.com/chiennv2000/orthrus/blob/main/src/models/modeling_orthrus_qwen3_5.py)、[KDA 递推定义](https://arxiv.org/html/2510.26692v2)、[原生 GLM KDA 接口](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/glm5next/ops/kda.py)。
