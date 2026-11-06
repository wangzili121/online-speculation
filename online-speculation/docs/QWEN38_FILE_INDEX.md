# Qwen3.8 相关文件索引

此表逐文件说明 Qwen3.8 vLLM 学习范围；DSV4、GLM、普通参考算法的其他组件仍保留在仓库中，但不是此模型的必经执行路径。历史验收记录和 JSON 证据不属于执行代码。

## 核心源码和服务/训练入口

| 文件 | 职责 |
| --- | --- |
| [scripts/prepare_qwen38_data.py](../scripts/prepare_qwen38_data.py) | 数据准备入口：将语料编码为训练脚本读取的 token ID 记录；数据准备与模型参数更新分开。 |
| [scripts/run_qwen38_vllm.py](../scripts/run_qwen38_vllm.py) | 服务入口：解析命令行，拼出原生模型与 custom_class 配置，再执行 vllm serve。它不实现模型 forward。 |
| [scripts/train_qwen38_draft.py](../scripts/train_qwen38_draft.py) | 离线全层训练入口：读取 tokenized JSONL，抽样 anchor，通过 worker RPC 驱动蒸馏与导出；不是对话请求训练。 |
| [src/blockspec/feedback.py](../src/blockspec/feedback.py) | 训练反馈数据结构：把 MASK 输入、历史、教师 logits、有效位置与冻结边界打包，供 learner 重放。 |
| [src/blockspec/parallel/backbone.py](../src/blockspec/parallel/backbone.py) | 公共模型输出与边界类型，以及普通 Transformer 参考 backbone；Qwen 主要复用 BackboneOutput / DraftBoundary。 |
| [src/blockspec/parallel/fitting.py](../src/blockspec/parallel/fitting.py) | 训练数据及独立训练工具：Qwen vLLM 离线入口主要复用 TokenDataset / BatchStream，文件内其他 fitting 入口不是该服务的必经调用。 |
| [src/blockspec/parallel/online.py](../src/blockspec/parallel/online.py) | 通用后缀训练器：选中 last-N draft 参数，维护 FP32 master / AdamW / replay，重放反向并发布更新。离线也复用它。 |
| [src/blockspec/parallel/qwen38.py](../src/blockspec/parallel/qwen38.py) | 模型结构层：AR/draft 双视图、Full Attention 与 GDN 混合层、临时 cache，以及可重放的后缀 forward。 |
| [src/blockspec/parallel/qwen38_training.py](../src/blockspec/parallel/qwen38_training.py) | 离线监督构造：对干净文本执行冻结 AR 教师前向，构造 anchor+MASK，保存待训练后缀入口。 |
| [src/blockspec/parallel/sampling.py](../src/blockspec/parallel/sampling.py) | 采样策略包装：从 draft logits 取候选与真实 q；其独立 verify 方法不等于 vLLM 在线路径的原生 sampler。 |
| [src/blockspec/sampling.py](../src/blockspec/sampling.py) | 概率与校正数学工具：温度/top-k/top-p、抽样、greedy 和随机校正参考实现。vLLM 主路径复用的是原生 rejection sampler。 |
| [src/blockspec/vllm_ascend/qwen38_attention.py](../src/blockspec/vllm_ascend/qwen38_attention.py) | 主模型 verifier 的分页 Attention 适配：K32 保持不变，内部按 query 上限切片并保持因果可见边界。不是 draft 双向 Attention 的实现文件。 |
| [src/blockspec/vllm_ascend/qwen38_gdn_backend.py](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py) | 原生 GDN 后端适配：处理递推 query 的分片与 prefill 路径，供原生模型及 draft 算子包装使用。 |
| [src/blockspec/vllm_ascend/qwen38_history.py](../src/blockspec/vllm_ascend/qwen38_history.py) | 历史读取层：从原生分页 KV / GDN checkpoint 取得已提交前缀的快照。共享历史来源，不代表所有数据零拷贝。 |
| [src/blockspec/vllm_ascend/qwen38_loss.py](../src/blockspec/vllm_ascend/qwen38_loss.py) | 可选词表并行损失：每卡只保留自己的 student 词表分片，通过 collective 得到全局归一化与散度。 |
| [src/blockspec/vllm_ascend/qwen38_ops.py](../src/blockspec/vllm_ascend/qwen38_ops.py) | 计算后端：给双视图模型安装现成 NPU 算子或可微分组合。根据设备、形状和是否启用梯度选路，不等同于自研 kernel。 |
| [src/blockspec/vllm_ascend/qwen38_proposer.py](../src/blockspec/vllm_ascend/qwen38_proposer.py) | 投机流程组织层：绑定原生模型，构造起草输入，保存 pending，提供 q，消费验证反馈并触发训练。 |
| [src/blockspec/vllm_ascend/qwen38_runtime.py](../src/blockspec/vllm_ascend/qwen38_runtime.py) | 引擎接入层：包装指定版本 NPUModelRunner 的生命周期、起草和采样入口；不重新实现原生 scheduler 或 rejection sampler。 |
| [src/blockspec/vllm_ascend/qwen38_weights.py](../src/blockspec/vllm_ascend/qwen38_weights.py) | 权重与 TP 适配层：引用原生 backbone 参数，复制 draft mixer，并为冻结权重输入梯度、TP collective 和 learner 提供适配。 |

## 支持文件与实验脚本

这些文件不需要先于模型主链阅读；遇到配置/验证问题时再打开。

| 文件 | 职责 |
| --- | --- |
| [src/blockspec/parallel/branches.py](../src/blockspec/parallel/branches.py) | 独立参考 branch/proposal 抽象；Qwen38Branch 继承它。 |
| [src/blockspec/parallel/feedback.py](../src/blockspec/parallel/feedback.py) | 独立参考生成/校正路径的反馈构造，不等于 native proposer.observe_verified。 |
| [src/blockspec/parallel/generation.py](../src/blockspec/parallel/generation.py) | 独立参考 AR/投机生成循环；vLLM 服务使用原生引擎循环。 |
| [src/blockspec/parallel/training.py](../src/blockspec/parallel/training.py) | 普通参考模型蒸馏与 anchor 采样等通用训练工具。 |
| [src/blockspec/parallel/weights.py](../src/blockspec/parallel/weights.py) | 普通参考模型 checkpoint/权重及文件哈希工具。 |
| [src/blockspec/vllm_ascend/draft_sampling.py](../src/blockspec/vllm_ascend/draft_sampling.py) | 其他通用 proposer 的 draft 采样适配，不是 Qwen38 proposer 的入口。 |
| [scripts/accept_qwen38_vllm.py](../scripts/accept_qwen38_vllm.py) | 验收编排：依次启动 target、fixed/online/head 重载等独立验证步骤。 |
| [scripts/validate_qwen38_vllm.py](../scripts/validate_qwen38_vllm.py) | 原生引擎功能验证：构造请求、收集 worker 计数、缓存/冻结审计与结果断言。 |
| [scripts/qwen38_validation_worker.py](../scripts/qwen38_validation_worker.py) | 仅实验使用的 worker 扩展：提供哈希、内存、数值、mask、重放和离线分阶段计时等 RPC。不是生产算法主体。 |
| [scripts/validate_qwen38_api.py](../scripts/validate_qwen38_api.py) | 通过 OpenAI API 检查服务端请求、返回、流式和更新相关行为。 |
| [scripts/run_qwen38_validation_window.py](../scripts/run_qwen38_validation_window.py) | 资源窗口验证编排，区别于生产服务启动入口。 |
| [scripts/run_qwen38_orthrus.py](../scripts/run_qwen38_orthrus.py) | 脱离 vLLM 原生 runner 的独立参考执行器，支持小配置/手动按层放置等实验。 |
| [scripts/run_in_runtime.py](../scripts/run_in_runtime.py) | 在固定镜像中准备本地 Python 运行环境后转入目标脚本，不实现模型数学。 |
| [scripts/benchmark_qwen38_gqa.py](../scripts/benchmark_qwen38_gqa.py) | 比较 GQA Attention 的现成算子调用与布局/复制成本。 |
| [scripts/benchmark_qwen38_history.py](../scripts/benchmark_qwen38_history.py) | 度量历史缓存读取相关开销。 |
| [scripts/audit_qwen38_training_ops.py](../scripts/audit_qwen38_training_ops.py) | 训练算子数值、梯度与耗时审计。 |
| [scripts/audit_qwen38_device_training.py](../scripts/audit_qwen38_device_training.py) | 整模型设备驻留训练、阶段计时与内存验证。 |
| [scripts/audit_qwen38_tp_training.py](../scripts/audit_qwen38_tp_training.py) | TP 分片 loss、梯度与训练计算对照验证。 |
| [scripts/audit_qwen38_execution.py](../scripts/audit_qwen38_execution.py) | 执行路径现成算子和分阶段耗时审计。 |
| [scripts/probe_qwen38_npu_gdn.py](../scripts/probe_qwen38_npu_gdn.py) | 小规模直接调用 NPU GDN 算子，核对形状、状态和数值。 |
| [scripts/probe_qwen38_npu_fia.py](../scripts/probe_qwen38_npu_fia.py) | 小规模验证 NPU 融合/分页 Attention 的调用约束与数值。 |
| [scripts/probe_qwen38_npu_norm.py](../scripts/probe_qwen38_npu_norm.py) | 比较 norm 算子与原生舍入/数值行为。 |
| [scripts/probe_qwen38_npu_history.py](../scripts/probe_qwen38_npu_history.py) | 探查 NPU 原生历史缓存读取与状态布局。 |
| [scripts/probe_qwen38_layerwise.py](../scripts/probe_qwen38_layerwise.py) | 逐层对照模型执行结果，用于定位数值偏差。 |
| [scripts/probe_qwen38_module_replay.py](../scripts/probe_qwen38_module_replay.py) | 定位模块后缀重放及输入梯度兼容性。 |
| [scripts/probe_qwen38_native_batch.py](../scripts/probe_qwen38_native_batch.py) | 探查原生主模型不同 query/batch 执行行为。 |
| [scripts/probe_qwen38_native_width.py](../scripts/probe_qwen38_native_width.py) | 探查原生主模型不同验证块宽度行为。 |
| [scripts/probe_qwen38_execution_ops.py](../scripts/probe_qwen38_execution_ops.py) | 核对现成执行算子接口/行为，供实现选路。 |

## 自动化测试

测试文件验证模块行为，不会因为打开测试文件就启动真实 27B；真实设备验收通常由 scripts 下的程序显式启动。

| 文件 | 检查对象 |
| --- | --- |
| [test_qwen38.py](../tests/test_qwen38.py) | 双视图模型、混合层和独立参考算法行为 |
| [test_qwen38_native.py](../tests/test_qwen38_native.py) | 原生权重映射、TP、runner hooks 和 native proposer/learner 适配 |
| [test_qwen38_history.py](../tests/test_qwen38_history.py) | 分页 KV、接受边界和 GDN 状态读取 |
| [test_qwen38_ops.py](../tests/test_qwen38_ops.py) | NPU 算子包装和参考/可微路径契约 |
| [test_qwen38_loss.py](../tests/test_qwen38_loss.py) | 词表并行 KL/TV 与梯度 |
| [test_qwen38_validation.py](../tests/test_qwen38_validation.py) | 验收脚本及验证判定逻辑 |
| [test_qwen38_window.py](../tests/test_qwen38_window.py) | 资源窗口实验编排 |
| [test_prepare_qwen38_data.py](../tests/test_prepare_qwen38_data.py) | 数据解析、分词、过滤和来源记录 |

## 配置与其他目录

* `pyproject.toml`：Python 包依赖、CLI、测试配置、插件注册。
* 根目录 `README.md`：本学习分支先读入口。
* `docs/QWEN38_VLLM.md`：部署参数与接入配置。
* `docs/QWEN38_TRAINING_OPS_AUDIT.md`：训练算子/设备驻留实验记录，不是 forward 实现。
* `docs/validation/qwen38/`：原始验收 JSON、日志摘要与指标，需结合对应实验配置阅读。
* 其他模型文件（`dsv4*`、`glm53*` 等）不应混入本次 Qwen 调用链。
