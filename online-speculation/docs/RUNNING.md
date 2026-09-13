# 运行与测量

运行位置为项目目录。模型、数据与结果位置由调用者传入。
命令中的环境变量表示本地选定的资源，个人配置保存在已忽略的 `local/` 或仓库外。

## 1. 环境与小模型检查

GPU 运行使用支持 CUDA 的 PyTorch；Windows 可在 WSL2 中运行。
安装与显卡驱动兼容的 PyTorch 后，执行：

```bash
python -m pip install -e '.[dev,text,hf,data]'
python -m pytest -q
python -m blockspec fit demo --device cpu --output local/demo.pt
```

合成闭环验证双向起草训练、张量映射及精确恢复。
输出检查点使用新文件名。主运行环境采用项目依赖，外部参照可使用独立虚拟环境。
分词器配置和外部模型源码各自需要兼容的 Transformers 版本，运行时选择对应的虚拟环境。

## 2. 输入资源

| 参数 | 内容 |
|---|---|
| `--model` | 双视图权重目录及对应 tokenizer |
| `--base` | 自回归基座目录及配置 |
| `--prompts`、`--learning-prompts` | 分离的评测、学习问题 JSONL |
| `--data`、`--validation` | 已分词的训练、验证 JSONL |
| `--output`、`--summary` | 调用者选定的结果或检查点位置 |

问题记录支持 `question` 或 `prompt` 字段；训练记录使用 `input_ids`。
模型维度、注意力头数、词表及特殊 token 从配置读取。
权重桥接校验受支持的架构、完整张量集合和形状；新增架构通过数值对齐后接入。

`corpus.py` 提供会话记录转换和按问题分组划分的工具；`parallel/fitting.py` 读取已分词序列。
同一问题的不同回答归入同一划分，学习与评测入口同时检查数据重叠。

## 3. 固定双向起草

```bash
python -m blockspec evaluate benchmark --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --requests 16 --tokens 2048 --blocks 32 \
  --empty-system --dtype bfloat16 --backend sdpa --temperature 0 \
  --repeats 2 --output "$RESULT_FILE"
```

`--thinking` 控制思考模板；省略时关闭。`--empty-system` 添加空 system 消息。
采样配置、聊天模板、EOS 规则与输出预算共同定义实验条件。
结果记录中的已有单轮测量使用 `--repeats 1`；独立复测宜增加重复轮次与问题数量。

三种概率执行的同轨迹对照：

```bash
python -m blockspec evaluate benchmark --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --requests 16 --tokens 2048 --blocks 32 \
  --empty-system --temperature 1 --top-k 0 --top-p 1 \
  --sampling-executions scalar tensor graph --repeats 2 --output "$RESULT_FILE"
```

随机采样评测默认选择 `tensor`，贪心评测使用直接最大值路径。
`scalar` 为逐项校正，`tensor` 为整块张量校正，`graph` 将相同张量操作录入 GPU 图。
输出给出各执行方式的 AR／投机 TPS、TPF、准备时间与配对区间。
剖析使用同一入口的 `profile` 模式，输出预算至多 256 token。

## 4. 在线学习

起草注意力后段续训：

```bash
python -m blockspec online --model "$DUAL_MODEL_DIR" \
  --prompts "$EVAL_PROMPTS" --learning-prompts "$LEARN_PROMPTS" \
  --empty-system --prompt-offset 32 --requests 16 --tokens 512 \
  --repeats 3 --shuffle-requests --seed 743 \
  --learn-requests 16 --learn-tokens 256 --last-layers 1 --stride 16 \
  --replay-blocks 1 --learning-rate 0.00001 --loss tv \
  --temperature 1 --top-k 0 --top-p 1 \
  --audit-requests 8 --audit-tokens 128 --output "$RESULT_FILE"
```

每轮校正后保存教师反馈，在更新间隔到达时重放起草后段、反向并发布参数。
评测组包括 AR、原始固定、原起点在线、预学习后固定及相同学习起点继续更新。
在线 TPS 包含反馈与更新成本；独立预学习、状态初始化及执行器准备单列。

此设置让训练与实际采样使用同一对完整词表分布。损失对照仅将 `--loss tv` 改为
`--loss forward_kl`，其余参数相同。每条重复流从各自的起始状态恢复，
`--shuffle-requests` 单独打乱顺序，报告同时给出合计及逐流吞吐比。
在线命令默认采用 TV、温度一及完整词表；模板和问题窗口由命令参数选择。

同前缀分布审计添加 `--audit-only --audit-requests 8 --audit-tokens 128`。

### 连续推理接口

给定已加载并冻结的双视图 `model`、同设备上的分词后 `prompt_stream` 与每请求输出预算 `output_budget`，
实际在线推理使用同一份学习状态：

```python
from blockspec import MaskedAttentionBranch, generate
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor

branch = MaskedAttentionBranch(model)
sampling = SamplingConfig(temperature=1., top_k=0, top_p=1.)
executor = SamplingExecutor(model.config.vocab_size, model.config.block_size, sampling,
                            device=next(model.parameters()).device)
sampler = ProposalSampler(sampling, executor=executor)
learner = SuffixLearner(model, SuffixConfig(last_layers=1, stride=16, loss="tv"))
for prompt in prompt_stream:
    result = generate(branch, prompt, output_budget, sampling=sampling, sampler=sampler,
                      eos_id=model.config.eos_token_id, feedback=OnlineFeedback(learner=learner))
    # result.tokens is this request's committed output.
```

每次请求结束后，`learner.state_dict()` 提供可本地保存的参数、优化器与更新计数；
相同模型和更新配置的学习器通过 `load_state_dict()` 恢复。
请求循环中的 TPS 已含在线成本，配对评测入口另提供固定起点等对照组。

## 5. 离线训练与恢复

```bash
python -m blockspec fit train --base "$BASE_DIR" \
  --data "$TRAIN_TOKENS" --validation "$VALIDATION_TOKENS" \
  --block-size 32 --mask-token-id "$MASK_TOKEN_ID" --device cuda \
  --precision bf16 --steps 200 --stop-after 100 --output "$CHECKPOINT_FILE"

python -m blockspec fit resume --checkpoint "$CHECKPOINT_FILE" \
  --data "$TRAIN_TOKENS" --validation "$VALIDATION_TOKENS" \
  --device cuda --output "$NEXT_CHECKPOINT_FILE"
```

随机锚点定义多个隔离的起草块，干净 AR 视图提供完整教师分布。
检查点保存参数、优化器、随机数、数据顺序及学习率进度。
恢复沿用保存的总步数和调度，`--stop-after` 表示中间停止边界。
`--optimizer-impl fused` 选择融合 AdamW；默认 `single` 保留逐张量执行。

### 冷启动的预算式完整块训练

冷启动实验位于可选的消融包，模型资源参数提供 AR 与双视图配置；
初始化时各层起草注意力均重新复制对应 AR 参数。
普通 AR 权重可通过 `--base "$AR_MODEL_DIR" --block-size "$BLOCK_SIZE" --mask-token-id "$MASK_TOKEN_ID"`
接入同一入口；掩码 token 从该模型的词表中显式指定。

```bash
python -m pip install -e ./ablation
python ablation/scripts/cold_start.py --model "$DUAL_MODEL_DIR" \
  --prompts "$TRAIN_PROMPTS" --heldout-prompts "$EVAL_PROMPTS" \
  --fraction .01 --requests 384 --tokens 256 --steps 128 --warmup-steps 8 \
  --block-size 4 --sequence-length 512 --anchors 42 --learning-rate .0001 \
  --probe-every 32 --probe-tokens 48 --gate-count 1 --heldout-count 8 \
  --heldout-offset 80 --heldout-tokens 128 --curve-every 16 --seed 857 --offline-control \
  --checkpoint "$COLD_STATE" --output "$RESULT_FILE"
```

真实请求提供训练序列，报告题和发布门控题彼此分离。
`--block-size` 指定训练与推理共用的块长，`--anchors` 控制每个窗口中的随机锚点数。
`--batch-size` 指定每个微批抽取的回答窗口数，较短窗口在右侧补齐，锚点取自原始有效范围。
每次更新的监督行数为 `accumulate × batch-size × anchors × (block-size - 1)`。
例如 `--batch-size 3 --anchors 14` 和 `--batch-size 1 --anchors 42` 使用相同数量的监督位置。
`--optimizer-impl fused` 与离线入口共用融合更新；对照实验和恢复沿用所保存的执行选项。
`--initial-probe-factor` 控制首次验证的耗时预留系数，默认 2；后续验证使用实测时间估计。
`--reuse-ar-prefix` 在 AR 服务期间记录当前回答的短前缀计时，复用为发布检查的 AR 参照。
回答交付后、进入重放区之前，只额外运行相同提示、种子和输出上限的候选分支。
前缀计时的新增开销记入 `validation_capture`，候选检查与参数切换记入 `validation`。
投机服务启用后，后续发布继续使用独立门控题上的 AR／服务版本／候选版本比较。
`--live-probe-requests` 指定同一候选版本在多少个实际请求上汇总 AR 前缀检查，默认 1。
取值大于 1 时，检查在请求开始之前根据已积累额度决定，候选参数在整组评估期间保持固定。
全部配对完成后，按两条路径各自的总 token 数和总耗时计算发布比值，随后恢复训练。
检查点保存尚待完成的配对统计；恢复沿用相同的评估数量与发布余量。
`--full-answer-screen` 选择完整回答的两级检查，与 `--reuse-ar-prefix` 分开使用。
`--screen-requests` 指定固定候选版本汇总的回答数，默认 2；`--screen-margin` 为筛选门槛，默认 1.02。
筛选通过后，后续另一条获准请求按实际输出上限完整生成候选，以 `--publish-margin` 的实测速度余量决定发布。
`--initial-screen-estimate` 提供首次评估的秒数预留，默认 0.4；后续按实测评估耗时预留。
筛选和确认均在请求开始前申请额度；通过筛选后的完整测速预留使用已完成评估的平均预测耗时，
另计候选临时切换成本并保留 20% 余量。超出预留的部分进入累计欠额。
该策略恢复时保留部分筛选统计和待确认版本；投机服务启用后的检查使用独立门控题。
增加 `--reuse-speculative-answers` 后，后续版本继续复用实际交付回答进行完整筛选。
候选与服务版本在整组检查中保持固定；通过筛选后，另一个获准请求额外生成候选与 AR，
分别比较候选／当前服务和候选／AR，两项达到发布余量后更新服务版本。
检查点同时保存服务版本号和用于估算确认成本的实测 AR 吞吐。
首次 AR 服务、训练目标和时间预算沿用完整回答策略。
性能记录第 9 节沿用上述命令，并设置
`--offset 768 --heldout-offset 176 --seed 1009 --optimizer-impl fused --initial-probe-factor 1.5 --reuse-ar-prefix`。
第 9.3 节的三请求对照改用 `--offset 1152 --heldout-offset 200 --seed 1061`，
并增加 `--live-probe-requests 3 --publish-margin 1.0504`。
第 11 节沿用冷启动命令，设置
`--offset 1536 --heldout-offset 256 --seed 1301 --optimizer-impl fused --initial-probe-factor 1.5`，
并采用 `--full-answer-screen --screen-requests 2 --screen-margin 1.02 --publish-margin 1.0504 --probe-tokens 256`。
该组使用完整回答策略，前缀复用选项保持默认值。
第 13 节沿用第 11 节设置，将 `--requests` 改为 768，并增加 `--reuse-speculative-answers`。
`--offline-control` 在流结束后从 AR 重新初始化，向离线训练提供全部已交付记录，
保持优化器配置、更新次数和监督行数相同，并在同一留出集比较学习质量。对照训练单独计时。
日常服务采用常规 GPU 执行设置。
模型导入与共同推理预热属于研究准备；在线账本从独立起草主权重的构造开始。
`--deterministic --offline-replay` 用于单独的逐元素审计：按在线实际抽取的批次离线重放，
检查最终参数一致。该配置固定矩阵乘工作区，并在更新内启用确定性算子。
直接调用 Python 审计接口时，在进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，
并显式传入 `deterministic_updates=True`。
报告中的 `stream.net_tps` 包含在线初始化、采集、训练、验证和发布；
`paired_run` 给出本次运行相对匹配 AR 的净吞吐。
AR 服务阶段复用该请求的实际生成时间；发布投机版本后，独立交错测量同提示、同输出预算的 AR 对照。
`curve` 为单独测量的学习质量，`research_measurement_seconds` 包含这部分测量和独立 AR 对照耗时。
研究测量期间的时间与提示均保持在服务账本和训练数据之外。
实验结束时的检查点写入耗时另列为 `shutdown_checkpoint_seconds`。

连续使用由 `ColdStartService.serve(prompt, output_budget, seed=...)` 提供。
它保留重放区、优化器、学习与服务参数版本和累计时间账本。
`state_dict()` 与 `load_state_dict()` 在请求边界保存／恢复；恢复新增的复制与校验成本进入账本。
恢复入口采用映射读取，随后访问权重、校验完整内容和装载优化器的时间一并计入恢复账本。
CPU 上的 FP32、BF16 参数通过分块指数域检查识别所有无穷与非数编码；其他类型使用张量有限值检查。
命令行通过 `--resume "$COLD_STATE"` 恢复，沿用原有训练、采样、控制器配置及相同的发布门控题；
`--offset` 指定接续的输入请求位置。训练步数表示整条学习流的总调度长度。

### 完整回答的条件工作量审计

```bash
python ablation/scripts/virtual_work.py --model "$DUAL_MODEL_DIR" \
  --checkpoint "$COLD_STATE" --prompts "$EVAL_PROMPTS" \
  --offset 232 --count 12 --tokens 256 --block-size 4 --seed 1249 --serial-audits 2
```

入口读取冷启动检查点的学习版本，独立比较普通 AR、候选生成与完整回答上的条件前向次数。
`--cold-copy` 替代检查点参数时，各层起草注意力从 AR 重新复制，可审计学习起点。
`--anchors-per-pass` 控制隔离掩码中同时计算的块数，默认 128；
`--calibration-points` 控制按访问质量选取的单轮计时点数，默认 3。
`--serial-audits` 对指定数量的完整回答逐起点重算起草概率，报告批量计算的数值误差。
`--prefix-compare` 交错测量同一候选的 48-token 前缀生成，与完整回答评估作额外成本对照。
输出为逐题统计及最终汇总，包含实际 TPS、预测速度、预期前向次数、概率评估和校准耗时。
这组测量属于独立研究评估，模型及学习状态保持原值。

## 6. 外部参照

独立运行读取权重张量。执行外部模型 Python 时，显式传入 `--reference-manifest "$REFERENCE_MANIFEST"`。
该本地文件提供被选中的代码、配置与权重的一致性校验，保存在仓库外或 `local/`。
结构为 `models.base`，包含 `weight_filename`、`weight_sha256` 和 `reference_lf_sha256`；
双视图参照另含 `entrypoint.file` 与 `entrypoint.class`。
`reference_transformers` 可指定外部环境版本。源码文件名及模型类均由本地配置选择。

```bash
python -m blockspec evaluate audit --model "$DUAL_MODEL_DIR" \
  --reference-manifest "$REFERENCE_MANIFEST" --blocks 4 32 --tokens 32
```

主线的数值参照核对 AR、起草、验证 logits 及历史 KV。
其他引擎与分支的运行入口见 [消融说明](../ablation/README.md)。

## 7. 测量与提交

同一问题交错运行各方法，采用相同模型精度、模板、采样规则和输出预算。
记录总输出 token、请求耗时、TPS、解码 TPF、更新次数与更新耗时；
启动、预热、预学习和状态切换单列。每种方法使用自身的匹配 AR 计算加速比。

实验入口通过统一报告层写出方法设置、指标与校验结论。
文件地址、资源标识、内容摘要和原始提示／生成文本留在本地输入与运行状态中。
`ALGORITHM.md` 保存推导，`RESULTS.md` 保存有效结果；其他设计的实现与取舍集中于 `ablation/README.md`。
提交前汇报性能和测试结果，提交说明聚焦方法变化与对应测量。
