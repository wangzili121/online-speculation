# Orthrus-Qwen3.8 on Ascend

Qwen3.8-27B 的 Orthrus 投机推理与 draft 更新实现。draft 复用冻结的主模型 backbone，推理读取主模型历史缓存，候选由原生 vLLM-Ascend 验证。支持在线 last-N 更新和独立离线全层更新，均只更新 draft mixer，不训练主模型。

Full Attention 在块内采用双向注意力，GDN 使用 `orthrus_qwen35_reference` 的因果递推策略。

## 验证状态

| 路径 | 当前状态 |
| --- | --- |
| 投机推理 | 真实 27B、TP2 已验证 K=2/8/17/32、多类请求边界、主模型历史缓存复用及 head 重载。 |
| 在线 last-N | last-1 / last-4 已在真实 27B 验证更新、冻结参数、只读历史和导出重载。固定反馈 32 步 TV loss 为 0.5044 → 0.4929，不代表业务加速收益。 |
| 离线全 draft | 真实 27B、TP2 已完成全 64 层更新、导出重载并继续 last-4 更新。4 步固定块 KL loss 为 7.16 → 3.06，主模型哈希未变。不是完整语料训练。 |

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
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --online --last-layers 4 --update-stride 16 \
  --max-model-len 4096 --kv-cache-memory-bytes 12884901888 \
  --checkpoint-dir /data/online-heads --port 8000
```

去掉 `--online` 为固定 head 推理；`--last-layers N` 选择在线更新的末 N 层，使用真正验证到的教师分布做 TV 更新。服务沿用原生 OpenAI API，默认监听本机。

`--checkpoint-dir` 按版本保存 head；使用 `--head-dir /data/online-heads/version16` 重启加载。加载要求基模 config/权重索引哈希、TP、块长、mask ID 和混合层策略一致。

## 离线全层更新

输入为同一基模 tokenizer 生成的 JSONL，每行 `{"input_ids": [...]}`。可附加 `anchor_ranges` 限定预测块所在的半开区间；每条数据须包含完整训练序列。

```bash
python scripts/train_qwen38_draft.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --data /data/train.jsonl --output /data/draft-fit \
  --steps 1000 --sequence-length 256 --learning-rate 0.0001 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

独立离线进程从文本随机选择 anchor，用冻结 AR 视图的完整 K-1 行概率做 forward KL 蒸馏，更新全部 draft 层，不提供生成服务。默认 CPU FP32 master/AdamW 和逐层 checkpoint，以搬运与重算降低显存需求。
TP2 全 draft 的 FP32 参数、梯度与 AdamW 状态合计约 116 GB 主机内存，另需模型加载和临时缓冲空间。

输出目录必须为新目录，包含 `loss.jsonl`、`summary.json` 和 `head/version<steps>`。在相同部署配置下，推理传 `--head-dir /data/draft-fit/head/version1000` 即可加载；离线也可用 `--head-dir` 从已有 head 继续学习，但不恢复 AdamW 状态，不能当作精确断点续训。

