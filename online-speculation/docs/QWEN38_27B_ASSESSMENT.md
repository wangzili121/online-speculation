# Qwen3.8-27B Orthrus 训推可行性评估

评估日期：2026-10-29。目标模型按 `Qwen/Qwen3.8-27B`，首阶段仅处理文本，冻结原模型，只训练起草分支；不改用内置 MTP 替代 Orthrus。以下是代码/文档核查与内存估算，不是该模型的 NPU 实测结果。

## 判断

相比 DSV4，27B 模型更适合在现有 A2 资源上推进，特别是只更新后缀的在线续训。但资源更小不代表接入已经完成，也不代表它是旧 Qwen3 的直接替换。

[官方配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) 使用 Qwen3.5 架构：64 层，每四层为三层 Gated DeltaNet（GDN）加一层完整 Attention，即 48 个 GDN、16 个完整 Attention，最后一层为完整 Attention。它没有 DSV4 的 C4/C128/indexer 路径，但需要处理 GDN 的卷积和递归状态。

[Ascend 官方部署文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html) 已给出 A2/A3 的 W8A8、TP2 推理示例。示例中的 `qwen3_5_mtp` 是内置 MTP，不是 Orthrus，也不证明训练或热更新已接好。新的工作应单独固定适合 Qwen3.8 的版本，不直接覆盖 DSV4 的固定镜像。

## 现有代码能复用到哪里

当前 `src/blockspec/parallel/weights.py::load_ar_base` 明确只接受 `model_type=qwen3`；权重桥接只映射 `self_attn`。因此不能只换模型路径运行 Qwen3.8。可复用的是数据、监督、冻结/梯度审计、后缀回放、优化器及请求边界发布等设计，混合层权重桥接与模型执行需新增适配。

[Orthrus 上游 Qwen3.5 实现](https://github.com/chiennv2000/orthrus/blob/main/src/models/modeling_orthrus_qwen3_5.py) 已有完整 Attention 和 GDN 的独立 diffusion 参数，可作为混合架构参考；它不是当前常博仓的实现，不能未经语义对齐就当作同一算法。须先明确 GDN 的起草定义、初始化、真实前缀状态及拒绝恢复，不自行删去 48 层或换成 MTP。

[常博当前说明](https://github.com/xchang1121/inference_scaling/blob/main/online-speculation/README.md) 和 [运行配置](https://github.com/xchang1121/inference_scaling/blob/main/online-speculation/docs/RUNNING.md) 的在线路径复用已到达的验证行，以冻结前段的边界表示重放、更新起草后段，并有 TV 监督与预算门控。应以实际选定版本为准，不能把普通 LoRA/SFT 或上游其他训练流程当成已经接好了这个闭环。

## 内存预算

下表是十进制 GB，仅算权重和优化器主要张量，不含激活、上下文/递归状态、回放缓存、通信、算子 workspace、视觉模块和独立服务副本。假设参考混合实现为全部 64 层建立独立起草 mixer，不把它视为很小的末端 head；具体算法选定后需重新按真实 tensor schema 统计。

| 项目 | 估算 | 依据 |
| --- | --- | --- |
| 冻结文本基模 BF16 | 约 54GB | 约 27B 参数，每参数 2 字节 |
| 全套起草 mixer BF16 | 约 14.5GB | 16 个完整 Attention + 48 个 GDN 的独立参数，约 7.24B |
| 两视图主要权重 | 约 68.5GB | 主干共享，只增加独立 mixer；已超过单张 64GB 卡可用容量 |
| 仅最后一层续训的 FP32 master、梯度及 Adam 状态 | 约 1.68GB | 约 104.86M 参数，每参数额外 16 字节；BF16 执行权重已在上面计入 |
| 最后四层同类状态 | 约 7.24GB | 3 个 GDN + 1 个完整 Attention，约 452.49M 参数 |
| 全部起草 mixer 同类状态 | 约 115.8GB | 不含上面约 68.5GB 执行权重，更不含激活 |

估算由配置维度及参考 diffusion 投影形状推导，不是已加载模型的显存测量。完整 Attention 约为 `5120 * (3 * 24 * 256 + 2 * 4 * 256)` 个投影参数，另有小量 norm；GDN 每层约 115.88M。选择其他合法起草结构时预算也会不同，不能将此数字作为算法变更理由。

资源建议：双卡 64GB 作为共享权重、短上下文、batch 1 的后缀训练验证起点，可能需要分层放置或 offload，需实测峰值；四卡更适合作为独立推理与训练 worker 的首期预算，例如各分两卡。两套独立进程不会自动共享 NPU 权重，不能只算一份模型。全套 draft 离线训练需另算，其权重和主要优化器张量合计约 184GB；四卡只能作为带分片/优化器 offload 的小 batch 探索起点，不能承诺四卡任意配置都够。

## 为什么先做最后一层

第 64 层是完整 Attention。若严格缓存并 detach 第 63 层后的 draft 边界，只训练最后一层起草参数，反向可止于该边界，不必穿过前面 48 层 GDN；后缀内冻结的 MLP、norm、LM head 仍需对输入计算梯度。这样能先验证最小在线更新闭环，避免一开始就要求全网 GDN 反向。

最后一层更新不能自动提供全套可用 draft。其余起草参数仍须按选定算法初始化、离线训练或预算式冷启动，接受率和加速收益另测。扩到最后四层或全套 draft 时，将再次需要 GDN 反向、状态截断与检查点兼容。

[ms-swift 的 Qwen3.5/3.6/3.8 训练文档](https://swift.readthedocs.io/en/v4.6/BestPractices/Qwen3_8-Best-Practice.html) 有 Ascend GDN 训练路径说明；[VeOmni 的 Qwen3.5 NPU 指南](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html) 提供 gated RMSNorm、causal conv1d 和 chunk gated delta rule 的 NPU 后端选择。它们说明不一定需要从零写算子，但不等于 Orthrus 特殊路径已被验证。普通 Attention 的 mask、GDN 初始/终态梯度、前缀状态及拒绝回滚都要单测。

## 建议实施顺序

1. 固定模型配置、checkpoint 和算法版本，核对混合层的 AR/draft 参数及状态定义；先不启动长期服务。
2. 用真实头维度检查完整 Attention 和 GDN 的 NPU 前向、必要反向、前缀状态及回滚，不用过小 toy 维度替代真实算子约束。
3. 以 BF16、短序列、batch 1 跑通 27B 的最后一层后缀更新，看到有限且可重复的 loss/梯度、冻结参数不变、draft 参数确实变化；记录实际峰值和每步耗时。
4. 实现验证数据采集、训练回放、候选 head 验证及请求边界发布。先在 PyTorch 训练路径闭环，再接 vLLM-Ascend 的服务侧采集与 head 版本切换；不在 vLLM 的前向专用量化 kernel 上直接强行 backward。
5. 再扩展后缀层数和离线初始化，最后评估量化服务、独立 trainer、并发及真实接受率收益。量化推理权重与训练视图要单独定义，不能把 int8 部署张量直接交给普通优化器。

结论：可以转向，在线后缀更新的资源可行性明显优于 DSV4；当前最应先验证的是混合架构兼容和最小训练闭环，而不是先承诺单卡、固定训练时长或已有一键训推支持。
