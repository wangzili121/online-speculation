# GLM-5.3-Flash 推理算子复核

日期：2026-11-03。仅检查推理，不推进训练与在线更新，不开发新 kernel。

## 推理接入

主模型继续走原生 vLLM-Ascend GLM 路径。插件通过 custom proposer 调用独立 draft 投影，共享 embedding、mHC、norm、MoE 和 LM head。MLA 读取已提交的 latent/indexer 池，KDA 读取接受位置的卷积尾部和递推矩阵；这些历史先形成请求自己的快照，draft 只修改临时状态。

共享历史并不等于零拷贝。`NativeGLM53History` 目前复制有效前缀/状态，`select_history` 用 Torch 组合打分、选池，再 gather 各 query 选中的 latent。它们仍是长上下文下值得进一步 profiling 的部分，不能把当前实现称为所有场景的最优版本。

MLA 保持块内双向；KDA 保留因果、门控反向块两种显式实验策略，分别加载匹配 head。没有因算子优化改动其数学定义。完整原生 KDA 入口仍限定 K≤8；单独 MLA 的 K=32 测试不等于放开模型限制。

## 本轮调用选择

| 部分 | 当前实现及选择依据 |
| --- | --- |
| 共享主干及投影 | 保留原生量化线性、mHC、MoE、norm 等模块，不替换主模型计算。 |
| MLA 选中历史 | 每行使用自己的索引，通过 FIA 返回 latent 输出和 LSE；AR 的 K/V 展开参数用于解释 AR 历史，draft 参数只解释临时块。 |
| MLA 临时块 | 从逐 query 展开 KV 改为一份临时块、K 行 query 的 FIA；已测局部耗时约 220–259 → 168–175 μs。 |
| 联合 softmax | 各段先经过自己的 value 展开，再用 `npu_attention_update` 合并，不把两个归一化结果直接相加。 |
| KDA 递推 | 继续使用目标 GLM 的 `run_recurrent_kda`，只传独立临时状态槽。 |
| KDA 卷积 | 接入同版 GLM 原生 `causal_conv1d`；实测 32/64 heads 短块更快，8 heads 更慢。默认仅在 BF16、K≤8、32/64 heads 时采用融合路径，其余保留 `conv1d + SiLU`。 |

卷积比较包含临时状态、非连续输入、FP32 参数到激活精度的转换与元数据构造：K=8 时，32 heads 约 618 → 568 μs，64 heads 约 1067 → 560 μs；8 heads 约 264 → 566 μs，后者没有设为默认。

**少复制临时 KV 不等于总 workspace 更小。** 例如 K=8/8-head 的 FIA 调用峰值约 18.3 → 27.8 MB，64-head 约 28.3 → 105.4 MB；更快的卷积同样增加临时 workspace。它们是短暂峰值，不是持久缓存增长，仍需完整模型评估余量。计时基于合成输入，只能说明局部选择，不能推出模型吞吐提升。

## 环境与验证

GLM 固定镜像：

```text
quay.io/ascend/vllm-ascend@sha256:61584d790fa3668e7c43f0eb50a3e3a86d94b3eed15bce5ac82391ed383df229
```

对应 vLLM/vLLM-Ascend 0.30、torch-npu 2.10.0.post7、CANN 9.2.1。DSV4 的 0.26 镜像没有相同 KDA/卷积接口，不能混用。111 的 HCCL 测试需显式选择节点网卡；这些环境值由脚本参数/环境传递，不写死在算子里。

单卡检查包含：不同 AR/draft 展开权重的 MLA、空/短/超过 512 个池的历史、两种 KDA 策略、非零反向门、K=1/2/8/32 的独立块计算、卷积尾部和目标状态不变。原始结果与完整模型以外的复验记录归档于 `validation/glm53/ops-20261103/`。

CPU 为 624 passed、44 skipped；单卡 demo 与新增专项共 31 passed，其中 18 项为 NPU 专项、13 项为通用复验，另有 15 组局部计时/数值对照。卷积覆盖 BF16/FP32 参数，以及真实融合投影中 QKV 切片的非连续行 stride。实际模型验证发现的参数精度约束已按原生 GLM 的转换方式处理，不用合成 BF16 参数替代这一项检查。

```bash
# 同版镜像内运行；源目录为 online-speculation。
GLM53_DEMO_NPU=1 ORTHRUS_RUN_NPU_TESTS=1 PYTHONPATH=src \
  python -m pytest -q tests/test_glm53_orthrus_demo.py tests/test_glm53_ops_audit.py
python scripts/run_in_runtime.py scripts/benchmark_glm53_attention_paths.py --output /tmp/glm53-paths.json
```

完整 45 层 TP8 尚未通过全部验收。真实源权重截层、单卡算子及初始化 head 只能验证所覆盖的链路，不能代替正式 head 的效果或整模型生产验收。PyPTO 当前能力、算法及开发接口见 [需求报告](ORTHRUS_GLM53_PYPTO_REPORT.md)。

## 本轮原生模型复验

使用真实源权重的前五层，保留原始宽度、W8A8、4 层 KDA 和 1 层 MLA。配置为单卡 TP1、K=8、eager、同步单请求，最大上下文 34816。每阶段运行相同三个请求，各输出 32 token；不是完整 45 层模型。

| 阶段 | 实测结果 |
| --- | --- |
| Target-only | 生成本轮固定基线。 |
| 因果 / 门控双向 draft | 两者各 96 个 greedy token 与基线逐个一致；分别完成 53/56 次 draft/cache 审计。双向门为 0.25，实际执行反向 KDA。 |
| 两种 head 独立进程重载 | 重载哈希与各自导出一致，重新生成的 greedy token 与基线一致；不是同一进程沿用内存里的 head。 |
| 两种策略 temperature=0.7 | 各完成 96-token 采样、真实 propose/verify 和缓存审计；这是功能测试，不是分布等价验收。 |

七个阶段全部通过，六个 draft 阶段累计 324 次目标缓存审计通过。各报告的推理代码哈希均与 `b30cf57` 提交一致；未提交模型/head 权重。结果、环境锁和压缩日志见 `validation/glm53/ops-20261103/native-5l/`。

这些结果不替代九层四卡和完整 45 层的严格验收，也不包含并发或图模式。本轮测试容器已删除，设备锁已释放。
