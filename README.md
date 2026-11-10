# Orthrus-Qwen3.8 on Ascend

PyPTO Serving 支持真实 27B 的 K32 draft、共享 backbone/history、缓存恢复与 head 重启换入，已通过单卡 A2 检查。参数启动时准备，自动选择设备驻留或流式上传；当前为 TP1、一个活动请求。[运行与验证](online-speculation/docs/QWEN38_PYPTO_PORT.md)。下文介绍 vLLM-Ascend 路径。

PyPTO 提供 `native`、`fused` 和 `paged` 三种 Attention 后端。直接分页路径已通过真实 27B 的 K32 起草、六个历史边界、cache 只读及拒绝恢复检查；默认保留原生后端。[接入计划](online-speculation/docs/QWEN38_PYPTO_PLAN.md)、[kernel 任务](online-speculation/docs/QWEN38_PYPTO_KERNEL_TASK.md)和[分页实现验证](online-speculation/docs/QWEN38_PYPTO_PAGED_VALIDATION.md)说明接口与实测结果。

PyPTO 的优化配置为 `--attention-backend paged --gdn-backend vector --draft-group-size 4 --head-transfer packed`：直接读取分页历史，整块执行 GDN、投影和共享 MLP，K32 的 HOST 计算提交由 66 次降为 18 次。[GDN 与 K32 调优](online-speculation/docs/QWEN38_PYPTO_GDN_KERNEL_TASK.md)包含独立算子、整层、整轮起草及 NUMA 对照。

PyPTO 路径默认在 NPU 上整理 head 权重，主模型矩阵直接加载到最终设备布局。可用 `scripts/prepare_qwen38_pypto_head.py` 保存转换后的 head，之后通过 `--head-dir` 直接加载；文件读取与哈希校验仍由主机完成。[权重准备与复用](online-speculation/docs/QWEN38_PYPTO_PORT.md#权重准备与复用)说明命令和格式。

PyPTO 另提供 `--online --last-layers N`（N 为 1–4）：根据验证反馈，在 NPU 上重放后段、执行显式反向和 FP32 Adam，再在两轮起草之间切换整组 head 参数。支持 checkpoint 续训及导出为固定推理 head；64 GB A2 使用独立学习卡，推理仍为 TP1。[在线更新配置](online-speculation/docs/QWEN38_PYPTO_ONLINE_LASTN.md)说明启动、恢复和导出方式。

Qwen3.8-27B 的 Orthrus 投机推理与 draft 更新实现。draft 复用冻结的主模型 backbone，推理读取主模型历史缓存，候选由原生 vLLM-Ascend 验证。支持在线 last-N 更新和独立离线全层更新，均只更新 draft mixer，不训练主模型。

Full Attention 在块内采用双向注意力，GDN 使用 `orthrus_qwen35_reference` 的因果递推策略。

[PyPTO 推理适配与算子设计](orthrus_qwen38_pypto_requirements.md)说明模型结构、两类 mixer 的公式、现有实现基础、Torch-NPU 代码及引擎接入。
现成算子的选择与对照见[推理算子审计](online-speculation/docs/QWEN38_INFERENCE_OPS_AUDIT.md)和[训练算子审计](online-speculation/docs/QWEN38_TRAINING_OPS_AUDIT.md)。主要配置为四卡设备驻留，双卡 CPU offload 作为少卡选项保留。

[接受长度与推理开销测速](online-speculation/docs/QWEN38_SPEED_BENCHMARK.md)对比原生主模型、初始化 head 与保留真实起草计算的全接受控制组，提供独立测速入口和原始记录。

## 验证状态

本轮 TP 通信与 loss 更新（`72e5da6`）已在真实 27B、TP4 上完成复验；原始记录见[训练算子审计](online-speculation/docs/QWEN38_TRAINING_OPS_AUDIT.md#整模型复验)。

| 路径 | 当前状态 |
| --- | --- |
| 投机推理 | 真实 27B、TP2 已验证 K=2/8/17/32、多类请求边界、主模型历史缓存复用及 head 重载。 |
| 在线 last-N | TP4 独立重载离线导出的 head，last-1 / last-4 各完成 8 次设备驻留更新，覆盖 greedy 和采样请求；四 rank 一致，主模型参数未变、历史缓存只读检查通过。 |
| 离线全 draft | TP4 在 512 / 2048-token 真实语料窗口上分别完成 4 / 3 步全 64 层更新与 head 导出，所有层均有梯度、主模型哈希未变。另保留开启/关闭激活检查点的验证及 TP2 offload 路径。 |

未提供预训练 head 时，从主模型 mixer 复制初始化，仅用于验证流程；目前不承诺接受率或吞吐提升。

## 环境

使用下列固定镜像，在容器中挂载本仓库、BF16 模型和结果目录，并映射 Ascend 设备：

```text
quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a
```

```bash
cd online-speculation
python -m pip install --no-deps --no-build-isolation -e .
```

以下模型与数据路径按实际挂载修改。`mask-token-id` 必须与 head 训练配置一致，示例值 `1` 仅对应当前初始化测试。

## 推理与在线更新

```bash
python scripts/run_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 4 --mask-token-id 1 --block-size 32 \
  --online --last-layers 4 --update-stride 16 \
  --max-model-len 2048 --kv-cache-memory-bytes 4294967296 \
  --checkpoint-dir /data/online-heads --port 8000
```

去掉 `--online` 为固定 head 推理；`--last-layers N` 选择在线更新的末 N 层，使用真正验证到的教师分布做 TV 更新。服务沿用原生 OpenAI API，默认监听本机。

`--checkpoint-dir` 按版本保存 head；使用 `--head-dir /data/online-heads/version16` 重启加载。加载要求基模 config/权重索引哈希、TP、块长、mask ID 和混合层策略一致。

## 离线全层更新

输入为同一基模 tokenizer 生成的 JSONL，每行 `{"input_ids": [...]}`。可附加 `anchor_ranges` 限定预测块所在的半开区间；每条数据须包含完整训练序列。

原始文本可先用 `scripts/prepare_qwen38_data.py --input /data/raw.jsonl --output /data/train.jsonl --model /models/Qwen3.8-27B --source <数据集及版本>` 转换。支持 `text` 或 `messages` JSONL；Parquet 输入另需 `pyarrow`。`--min-tokens` 至少设为训练的 `--sequence-length`，输出同时记录数据来源与哈希。

卡充足时使用下面的设备驻留配置，FP32 master、梯度和 AdamW moment 都保留在 NPU。示例选择已有 FLA 训练后端：用 `pip install --no-deps --target /data/fla-deps fla-core==0.6.0 flash-linear-attention==0.6.0` 准备依赖，并设置 `ORTHRUS_QWEN38_FLA_PATH=/data/fla-deps`，该目录须在容器内可见。不使用 FLA 时改为 `--training-gdn-backend torch`。

```bash
python scripts/train_qwen38_draft.py \
  --model /models/Qwen3.8-27B --tp 4 --mask-token-id 1 --block-size 32 \
  --data /data/train.jsonl --output /data/draft-fit \
  --steps 1000 --sequence-length 128 --learning-rate 0.0001 \
  --optimizer-device model --no-checkpoint-layers --training-gdn-backend fla \
  --max-model-len 2048 --kv-cache-memory-bytes 4294967296
```

独立离线进程从文本随机选择 anchor，用冻结 AR 视图的完整 K-1 行概率做 forward KL 蒸馏，更新全部 draft 层，不提供生成服务。上述 128-token 配置关闭激活检查点，避免反向时重算；四卡实测预热后单步中位数约 2.91 s，开启时约 3.63 s。较长序列或显存不足时使用 `--checkpoint-layers`，用重算降低激活内存。

少卡配置保留：将上述命令改为 `--tp 2 --optimizer-device cpu --checkpoint-layers --kv-cache-memory-bytes 8589934592`。TP2 全 draft 的 FP32 参数、梯度与 AdamW 状态合计约 116 GB 主机内存，另需模型加载和临时缓冲；该 offload 路径的性能不作为设备驻留训练的提速依据。

入口不显式指定时，离线仍默认 CPU offload 和 checkpoint，在线默认设备驻留。FLA 由 NPU worker 延迟加载，不修改既有 Torch/Triton；只影响训练用 GDN，推理仍使用镜像已有算子。

训练默认使用 `--training-loss-backend gathered`。可选 `vocab_parallel` 在词表分片上精确计算同一 KL/TV，减少临时显存；当前四卡 K32 小测未获得更低延迟，因此不作为卡充足时的默认配置。两种方式使用相同的 head 格式，均不改变推理采样。

输出目录必须为新目录，包含 `loss.jsonl`、`summary.json` 和 `head/version<steps>`。在相同部署配置下，推理传 `--head-dir /data/draft-fit/head/version1000` 即可加载；离线也可用 `--head-dir` 从已有 head 继续学习，但不恢复 AdamW 状态，不能当作精确断点续训。
