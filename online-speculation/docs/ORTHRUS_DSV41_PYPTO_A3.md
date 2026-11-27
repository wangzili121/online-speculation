# Orthrus-DSV4.1 单机 16 卡验证

本分支的验证入口使用一台主机上的 16 张 Ascend A3 卡，拓扑为 DP4、TP4、EP16。Serving 基于 [A3 Serving PR #1](https://github.com/co-agent-serving/DeepSeek-V4.1-Flash-A3/pull/1) 的 `085fde1`。

## 代码版本

| 仓库 | 分支 | 配套提交 |
| --- | --- | --- |
| [pypto-lib](https://github.com/wangzili121/pypto-lib) | `feature/orthrus-dsv41-a3` | `c2396a52c2e78485399bfb1f4c9d6abef4ee3afb` |
| [DeepSeek-V4.1-Flash-A3](https://github.com/wangzili121/DeepSeek-V4.1-Flash-A3) | `feature/orthrus-dsv41-a3` | `b7117c261e61a953aacb18f75ffe2b227e2d1267` |
| [online-speculation](https://github.com/wangzili121/online-speculation) | `orthrus-dsv41-pypto-a3` | 本文所在提交 |

Lib 提供模型、起草、验证和在线更新。Serving 通过 `PYPTO_LIB_ROOT` 加载对应模型包，并处理请求和流式输出。本仓维护效果实验、数据集和报告。更新 Lib 或 Serving 时，在这里更新配套提交和受影响的验证命令。

## 准备

从本仓根目录执行。将 `LIB_ROOT` 和 `SERVING_ROOT` 指向上表的 checkout，`DEVICES` 填分配到的 16 张可见卡，按 rank 顺序排列。Python 环境使用 Serving 的 `pypto.pin` 对应的 PyPTO、Simpler、PTOAS 及 A3 CANN。

```bash
export LIB_ROOT=/path/to/pypto-lib
export SERVING_ROOT=/path/to/DeepSeek-V4.1-Flash-A3
export CHECKPOINT=/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8
export HEAD_DIR=/workspace/orthrus-dsv41-a3/head-k16
export RESULTS=/workspace/orthrus-dsv41-a3/results
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m pip install -r "$SERVING_ROOT/pypto-serving-deepseek-v4.1-flash-a3/requirements.txt"

# 已有与该 checkpoint 和 K16 匹配的 BF16 head 时，直接设置 HEAD_DIR。
PYTHONPATH="$LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python -m models.deepseek_v4_1_flash_a3.orthrus_draft.initialize \
    --checkpoint "$CHECKPOINT" --output "$HEAD_DIR" --mask-token-id 128803 --block 16
```

## 先跑 smoke

```bash
STUDY=online-speculation/scripts/run_dsv41_pypto_a3_study.py
python "$STUDY" --serving-root "$SERVING_ROOT" --lib-root "$LIB_ROOT" \
  --smoke --output "$RESULTS/smoke.json" \
  --model-root "$CHECKPOINT" --devices "$DEVICES" \
  --max-model-len 512 --chunk-size 128 --prefill-moe-slab-rows 64 \
  --orthrus-head "$HEAD_DIR" --online-last-layers 1 --update-every 1
```

通过条件是 greedy 输出与 AR 逐 token 一致，且完成至少一次在线更新。复用图时，取终端 `A3_SERVING_GRAPH_DIR` 后的目录。切换配套源码后首次执行让脚本重新编译。

## GSM8K 对照

```bash
export DATA_DIR=/workspace/orthrus-dsv41-a3/gsm8k
export GRAPH_DIR=/path/from/A3_SERVING_GRAPH_DIR
python online-speculation/scripts/prepare_dsv41_gsm8k.py --output "$DATA_DIR"
python "$STUDY" --serving-root "$SERVING_ROOT" --lib-root "$LIB_ROOT" \
  --data "$DATA_DIR" --output "$RESULTS/gsm8k.json" \
  --model-root "$CHECKPOINT" --devices "$DEVICES" --graph-dir "$GRAPH_DIR" \
  --max-model-len 512 --chunk-size 128 --prefill-moe-slab-rows 64 \
  --orthrus-head "$HEAD_DIR" --online-last-layers 1 \
  --train-count 64 --heldout-count 16 --tokens 64 \
  --online-loss tv --learning-rate 1e-5 --update-every 16 --teacher-rows 7
```

有现成的 GSM8K 原始文件时，数据脚本支持 `--source-directory`。当前实验并发为 1。在线阶段与 AR 使用相同输入，初始和冻结 head 使用相同的独立留出题。JSON 报告记录端到端吞吐、扣除同步更新时间的估算、每轮接受长度、阶段计时和更新次数。逐请求 token、对齐和更新明细在同名 JSONL。
