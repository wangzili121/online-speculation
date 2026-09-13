# 运行与评测

本页说明 AR-LLM 与 dLLM 的训练、推理和评测入口。算法目标、概率修正、奖励定义和执行实现集中在
[算法文档](../methods/ALGORITHMS.md)；安装与解释器配置见[README](../../README.md#安装)。
已完成的实验分别汇总在[算法设计与准确率](../reports/GSM8K_3090_ALIGNED_RESULTS.md)和
[推理成本与执行效率](../reports/RTX3090_ROLLOUT_INFRA.md)中，各报告独立列出所用设置。

## 数据与配置

单方法通用入口默认使用 `arllm.toml`，通过 `--model` 指定因果语言模型，最大生成长度为 32,768 token。
上下文约束、思考范围、模型版本和分块评分的配置集中在
[README 的通用模型配置](../../README.md#通用模型与生成配置)与[实现说明](../methods/ALGORITHMS.md#alg-model-loading)。
以下 Qwen 复现命令显式选择历史配置；新加载器及思考范围功能的验证使用 CPU 微型模型和接口测试，尚未开展正式质量或性能实验。

配置位于 [`configs/`](../../configs/)。`gsm8k_quick.toml` 用于短预算检查，`gsm8k_3090_aligned.toml`
用于 Qwen2.5-1.5B，`gsm8k_llada_moe_3090.toml` 提供 LLaDA-MoE 的模型与算法参数。GRPO 训练单独读取
`gsm8k_grpo.toml`。模型路径、revision、数据子集、随机种子与生成长度均由配置和 CLI 参数决定。

准备脚本下载固定版本的公开 GSM8K 数据与模型。AR 使用 Qwen2.5-1.5B 作为基础模型，0.5B 模型可用于
候选或 rollout proposal；dLLM 使用 LLaDA-MoE，低层 proposal 由同一模型的分层执行提供。

数值参考值 verifier 会读取数据集标准答案，适合共享奖励目标诊断。自一致性、序列 logprob 和 Consilience
使用模型自身的输出或概率。自定义 verifier 通过 `--verifier-config` 选择；
`requires_reference = false` 时，入口向 verifier 提供提示与生成，不传入标准答案。

<a id="method-labels"></a>
## 方法标识

| 方法 | AR-LLM 标识 | dLLM 标识 |
| --- | --- | --- |
| 基础模型采样 | `base` | `base` |
| Beam search | `beam` | `block_beam` |
| 多次生成后选择 | `best_of_n` | `best_of_n` |
| 幂目标 MH | `mh` | `trajectory_power_mh` |
| 条件 IS | `conditional_is` | `conditional_is` |
| 低成本 rollout proposal | `conditional_is_small_proposal` | `conditional_is_reduced_layer_proposal` |
| 未校正 rollout 加权 | `conditional_is_small_proposal_uncorrected` | `conditional_is_reduced_layer_proposal_uncorrected` |
| RL 参数随机采样 | `rl_sample` | `vrpo_sample` |
| RL 参数贪心解码 | `rl_greedy` | `vrpo_greedy` |
| verifier 奖励 MH | `verifier_mh` | `verifier_mh` |
| verifier 奖励 IS | `verifier_conditional_is` | `verifier_conditional_is` |

RL 参数来自 GRPO 或 VRPO 训练；随机采样使用配置的采样策略，贪心解码每步取最大概率项。
完整标识与配对关系由 [`methods.py`](../../experiments/shared/methods.py) 维护，各入口的 `--help` 列出
可选值。迭代 IS、动态候选等研究方法需要显式选择，采用条件见[非默认方案记录](../methods/ALGORITHMS.md#alg-nondefault-notes)。

## 统一入口

以下命令从仓库根目录执行。默认运行 AR-LLM；解释器依次取 CLI 参数、`AR_PYTHON` / `DLLM_PYTHON`
环境变量和当前 Python。

低成本功能检查包含数据准备、一次 GRPO 更新和短预算推理：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile smoke --tag local-qwen
```

完整的 AR 训练与推理：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile full --tag qwen-full
```

使用已有模型，只运行指定方法与组件：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage inference --profile full --tag qwen-is-mh `
  --ar-methods base mh conditional_is `
  --components quality replay async --limit 32
```

LLaDA-MoE 在满足显存需求的机器上使用相同入口：

```powershell
python experiments\run_reproduction.py `
  --family dllm --stage all --profile full --tag llada-full `
  --dllm-python $env:DLLM_PYTHON
```

`--family both` 依次调度两侧。`--stage all` 将本次训练产生的适配器显式传给后续推理任务；只运行推理时，
RL 方法要求配置中的适配器已经存在。`--dry-run` 写出命令清单，供启动前检查路径、解释器与预算。

| 参数 | 作用 |
| --- | --- |
| `--limit` | 推理题目数量 |
| `--train-limit`、`--max-train-steps` | 训练样本与更新预算 |
| `--max-completion-length` | 训练补全长度 |
| `--passk-limit`、`--passk-draws` | pass@k 题目与独立重复数量 |
| `--ar-methods`、`--dllm-methods` | 两侧的具体推理方法 |
| `--ar-mh-suffix-schedule` | `uniform`、`inverse_length` 或 `multiscale`；统一入口默认 `multiscale` |
| `--verifier-config` | 独立奖励配置 |
| `--output-root` | 调度清单和组件汇总的输出目录 |

使用不同模型配置或后端时，可直接调用模型族入口：

```powershell
python -m experiments.arllm.run_arllm_suite `
  --config configs\gsm8k_3090_aligned.toml `
  --stage inference --profile full --methods base mh conditional_is `
  --components quality --backend transformers --tag custom-ar
```

<a id="infra-labels"></a>
<a id="replay-labels"></a>
## 组件与比较对象

| 组件 | 运行内容 | 比较对象 |
| --- | --- | --- |
| `quality` | 所选方法的单次生成评测 | 相同题目和生成长度的 Base |
| `matched_target` | 固定 verifier 下的 MH、IS 与 RL | 同一奖励定义与尺度 |
| `replay` | 纯新生成、已有历史与候选复用 | 相同候选与总 rollout 预算 |
| `async` | 连续批处理 | 相同请求和随机种子的逐提示执行 |
| `passk` | 每题独立重复生成 | 相同独立重复数 |
| `distribution` | 经验答案分布与累计计算量 | 共享奖励目标 |
| `dynamic_is` | 动态 proposal 与预算分配 | 固定候选 proposal 与固定分配 |
| `infra` | rollout 调度、流式奖励、MH 复用与 SMC | 同一算法的基础执行路径 |
| `ablations`、`budget_curve`、`length_ablation` | 参数、预算与长度扫描 | 每次固定其余参数 |
| `vllm` | AR 后端比较 | 相同模型、数据与数值类型的 Transformers |

`full` 默认包含前六项；其余组件通过 `--components` 显式选择。replay 最终估计记录预留后只使用一次；
候选重复出现时，共享该匹配键的库存约束。历史库构建、初始估计和在线推理成本分别记录。

## 统计与成本

质量统计包括单次生成准确率、Wilson 区间、题目级配对自助法区间与独立重复的 pass@k。经验答案分布使用
总变差距离（TV）和 Jensen–Shannon 散度。IS 记录权重 ESS 与复用数；MH 记录更新数、接受率和改变的 token 数。
统计实现位于 [`shared/statistics.py`](../../experiments/shared/statistics.py) 与
[`shared/metrics.py`](../../src/inference_scaling/shared/metrics.py)。

模型 $`j`$ 的前向 FLOPs 按参数量 $`N_j`$ 与实际参与前向计算的 token 位置数 $`S_j`$ 估算：

```math
\widehat F_j=2N_jS_j.
```

计数覆盖前缀预填充、逐 token 生成、序列重评分与草稿验证。基础模型和 proposal 模型分别记录后求和；该估算
省略注意力中随长度平方增长的项及逐元素算子。GRPO/VRPO 另外记录当前策略前向、反向、参考评分与适配器更新。

墙钟排除模型和数据加载，包含调度、评分及本次任务的收尾。比较时同时报告“优化路径成本 / 基线路径成本”
和采用的基线；小于 1 表示成本降低。后台预生成、缓存构建与首次查询的成本单列，避免把预先计算视作零成本。

<a id="consilience-protocol"></a>
## Consilience 评测设置

以下为原论文核对后的实验方案，正式运行尚未启动。分数定义、思考段目标和 off-policy 权重的简化见
[Consilience](../methods/ALGORITHMS.md#alg-consilience)。模型加载、生成分段和采样范围入口完成后，按此方案验证。

### 原论文设置与本地设置

[原论文第 4 节和附录 B](https://arxiv.org/html/2608.09898v1#S4) 的单轮实验使用每题 256 条独立生成的共同池，
从中抽取不同大小的子集，重复 10 次；主表比较 64 个候选上的 Top-1 选择。任务包括 LiveCodeBench-v6、
HMMT25-Feb 和 GPQA-Diamond，推理硬件为 4 张 48 GB L40S，最大生成长度为 130k。
原论文在所有选择方法与 Pass@1 计算前统一过滤截断生成。以下本地方案调整模型与预算，分别记录选择质量和执行成本。

| 项目 | 本地方案 | 依据 |
| --- | --- | --- |
| 主模型 | Qwen3-4B-Thinking-2507，BF16 | 明确的思考边界；3090 上优先单条长序列并发，容量需小规模验证 |
| 功能检查 | Qwen3-1.7B 的 thinking 模式或人工 token 后端 | 检查分段、评分与概率修正；与正式结果分开 |
| 最大新生成长度 | 32768；开发集检查截断后再决定是否提高 | 长度上限固定后用于测试；同时受提示长度、模型上下文和显存约束 |
| Consilience 范围 | 成功分段时仅 thinking；失败时使用全序列并记录原因 | 正常分段对齐思考段评分；回退样本单列统计 |
| top-logprob 数量 | 5 | 与论文一致；区别于生成时的 top-k 截断 |
| 窗口 | 跳过开头 5%，前后各 20% | 以思考 token 数为基准，首段系数固定为 3 |
| 分数概率 | 固定基础模型、温度 1、完整词表归一化 | 与 proposal 温度分开，保证奖励定义固定 |
| 严格 IS/MH 的生成策略 | 温度 0.6，top-p=1，无 top-k 截断 | 此策略定义比较中的基准分布；保留目标支持集 |
| 发布方推荐采样对照 | 温度 0.6、top-p=0.95、top-k=20 | 单列质量基线，采用不同的生成分布 |
| IS/MH 奖励强度 | 总尺度 1，初始候选值为温度 2，即有效强度 0.5 | 本库扩展的待验证起点，原论文未确定此参数 |
| 强度对照 | 有效强度 0、0.25、0.5、1、2 | 0 用于基准分布；其余对应奖励温度 4、2、1、0.5 |

[模型官方说明](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#best-practices) 建议一般生成上限为 32768，
竞赛数学和代码可达 81920；这些长度不代表单张 3090 能同时容纳多个长上下文。其 chat template 已包含起始
`<think>`，生成可能只有 `</think>`。分段应结合实际模板与原始 token，保留边界概率，并记录思考结束、
最终内容结束、长度截断和缺失边界等状态。

上述分数概率固定为温度 1 是本库的明确约定。论文通过推理服务收集 top-logprob，未完整说明所有服务版本下
返回值与采样温度、截断处理的对应关系；对齐公式需要核对实际返回的概率定义。仅保留 top-5 数值用于统计，
与把生成支持集截断到 5 个 token 是两种操作。

### 数据与比较对象

GSM8K 保留为已有结果的衔接设置。主要信号验证建议使用公开的
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)，代码任务采用
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 的 v6 新增题目，对齐论文的任务类型。
任务适配器需要对应的最终结果解析与评分规则；GSM8K 的数值解析器不适用于一般数学表达式或代码测试。
参考答案与测试用例只进入离线评测。开发集与测试集按题目分开，参数在测试前固定。

先比较同一生成池上的选择信号，再比较在线算法，分离奖励有效性与推理调度的影响：

| 比较 | 方法 | 固定条件与目的 |
| --- | --- | --- |
| 选择信号 | 随机选择、最终结果多数投票、平均 token logprob、平均 top-5 统计量、末段 20%、Consilience | 共用完整候选池；在支持多数投票的任务上比较其结果 |
| 有限候选重采样 | 相同池上的 Consilience Top-1 与按指数权重随机选择 | 候选数取 1、2、4、8、16、32，检验硬选择与 IS 的区别 |
| 在线条件 IS | 固定 Consilience 奖励的普通条件 IS、rollout replay、低成本 proposal | 固定目标与校正规则，按实测 token/FLOPs 和墙钟比较 |
| 奖励 MH | 基础模型、独立全序列提议、后缀提议及其缓存优化 | 使用相同 Consilience 目标；记录实际 MH 更新次数与有效状态变化 |
| 生成范围 | full 与 thinking | 成功分段时 Consilience 均只对 thinking 评分；后一设置在选定思考后生成最终内容，回退样本单列 |

代码任务的主要指标为每题最终选定一份代码的测试通过率；pass@N 仅作为候选覆盖率诊断。为区分论文中的
平均 top-5 置信度基线与全词表自确定度，两者使用不同方法标识。生成池子集重复 10 次用于估计选择波动；
置信区间按题目配对重采样，避免把同一题的重复选择当成独立题目。

MH 的迭代预算独立于最大生成长度设置；提高生成上限时，保持指定的更新次数。相同的候选数、rollout 数和
MH 更新数代表不同计算量，主要对比使用实际生成、评分 token 与估计 FLOPs；墙钟作为执行效率指标。
外部 proposal 的校正基线保留精确概率比，截断概率比和省略校正作为独立对照。

### 诊断与结果口径

- 质量：最终选定结果的准确率、候选覆盖率、题目级配对区间；评分与正确性的关系只作离线诊断。
- 分段：思考长度、最终内容长度、自然结束比例、截断比例、缺失边界与空思考比例；请求/实际范围及全序列回退比例。
- 信号：首尾窗口均值、原始分数、top-5 总概率，以及这些量与长度和正确性的关系。
- IS：内层与外层 ESS、最大归一化权重、复用数量和概率比范围。
- MH：接受率、实际改变的思考 token 数、重复状态比例；EOS 填充区的操作单独统计。
- 成本：生成、重评分、最终内容生成、历史库构建的 token 与 FLOPs，基础模型与 proposal 分列；另列峰值显存和墙钟。

完整率与准确率共同报告。生成预算耗尽且缺少可评测最终结果时，计入失败；仅在已完成样本上比较的结果作为
附加诊断，并使用所有方法相同的过滤集合。重试和被丢弃生成的成本计入总成本。截断思考的末窗口只是现有前缀的
末端。此类样本使用预先固定的全序列回退奖励，并保留截断标记；相关统计与完整思考段的窗口统计分开。

评分与 proposal 使用不同模型时，Consilience 始终由指定的基础模型计算；由小模型生成 rollout 仅减少其生成成本。
前缀置信度复用、生成时收集 top-logprob、长序列分块评分分别进行执行测试，核对分数和概率比后再测速度。
只有已在生成阶段收集所需统计量的路径，才能把选择阶段视为接近零额外模型前向。

## 输出与续跑

| 内容 | 默认位置 |
| --- | --- |
| 根级调度清单 | `results/reproduction/<tag>/reproduction_manifest.json` |
| AR 逐题记录 | `results/gsm8k/<profile>/` |
| AR 汇总与子清单 | 根级入口的 `<output-root>/arllm/`；独立入口的 `--summary-root` |
| dLLM 组件结果 | 根级入口的 `<output-root>/dllm/` |
| 模型与适配器 | `models/` 或训练参数指定的位置 |

输出目录由入口按需创建，`results/` 整体由 Git 忽略。仓库保留可重新运行的代码与配置，历史实验产物不作为
测试数据或运行前置条件。单元测试在临时目录构造所需样本。

逐题 JSONL 按配置标识续写，pass@k 使用独立任务分块。清单记录有效配置、数据行号、模型 revision 和实现哈希；
相同标签与配置可续跑。算法、模型或预算改变时使用新的 `--tag`。底层脚本和汇总器的参数可从统一入口的
`--dry-run` 输出取得。
