# Orthrus-Qwen3.8 / Ascend

本分支：`feature/orthrus-qwen38`。提供 Qwen3.8-27B 的投机推理、在线 last-N 更新和离线全 draft 更新。主模型始终冻结；未提供预训练 draft 时，从 AR mixer 复制初始化，只用于打通流程，不保证加速。

## 环境

使用已验证镜像，挂载本仓库、模型和结果目录，并映射 Ascend 设备：

```text
quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a
```

在容器中进入 `online-speculation`，执行 `python -m pip install --no-deps --no-build-isolation -e .`。
下列 `/models/Qwen3.8-27B`、`/data` 路径按实际挂载修改。mask ID 必须与训练 head 一致。

## 推理与在线 Last-N

```bash
python scripts/run_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --online --last-layers 4 --update-stride 16 \
  --max-model-len 4096 --kv-cache-memory-bytes 12884901888 \
  --checkpoint-dir /data/online-heads --host 0.0.0.0 --port 8000
```

不传 `--online` 则固定 head；用 `--head-dir /data/head/version4` 加载离线或在线导出的 head。
只更新最后 N 层的 draft mixer，以实际验证到的教师概率计算 TV loss。可选 `--optimizer-device cpu --checkpoint-layers` 降低更新显存，代价是搬运与重算。`--all-draft-layers` 可显式选择在线全层，但不建议作为低延迟服务默认配置。

## 离线全 Draft

数据为 tokenized JSONL，每行 `{"input_ids": [...]}`；可附加 `anchor_ranges`，限定完整预测块所在的半开区间。数据须使用同一基模 tokenizer，每条至少满足序列长度。

```bash
python scripts/train_qwen38_draft.py \
  --model /models/Qwen3.8-27B --tp 4 --mask-token-id 1 --block-size 32 \
  --data /data/train.jsonl --output /data/draft-fit \
  --steps 1000 --sequence-length 256 --learning-rate 0.0001 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

此入口不提供生成服务。随机选择 anchor，用冻结 AR 教师强制计算完整 K-1 个位置的分布，以 forward KL 更新**所有层的 draft mixer**。默认 CPU FP32 master/AdamW、逐层 checkpoint；embedding、共享 norm、MLP 和 LM head 不更新。输出 `loss.jsonl`、`summary.json` 和 `head/version<steps>`；导出的 head 可在相同 TP、K、mask ID 和基模上直接加载。重新传 `--head-dir` 是权重续训，不是恢复 AdamW 状态的精确断点续训。

## 调用方 / Boom Agent

调用标准 OpenAI 接口即可，不需要在 agent 中接入 draft 或训练逻辑。示例配置：`base_url=http://<server>:8000/v1`，`model=/models/Qwen3.8-27B`，API key 按部署侧认证设置。服务不默认提供认证，跨主机访问应限制网络权限。
工具调用解析、chat template 和 agent 全流程需由调用方另行联调，不属于本分支已完成的验收。

当前范围：BF16、eager、同步单请求执行，TP2 已有推理/在线验证；队列可以接收多个客户端，但引擎不是并行 batch。暂不承诺量化、图模式、多模态或吞吐收益，K32 仍有已记录的数值一致性边界。主模型走原生 vLLM-Ascend；draft 复用镜像已有融合算子，反向不支持的步骤使用可微组合，不含自研 kernel。

详细配置和验收边界见 [Qwen3.8 接入说明](online-speculation/docs/QWEN38_VLLM.md)。
