# Orthrus-DSV4.1 单机 16 卡验证

本分支的验证入口使用一台主机上的 16 张 Ascend A3 卡，拓扑为 DP4、TP4、EP16。Serving 基于 [A3 Serving PR #1](https://github.com/co-agent-serving/DeepSeek-V4.1-Flash-A3/pull/1) 的 `085fde1`。

## 代码版本

| 仓库 | 分支 | 配套提交 |
| --- | --- | --- |
| [pypto-lib](https://github.com/wangzili121/pypto-lib) | `feature/orthrus-dsv41-a3` | `5633e3fd40a6ae0c4c0fcbd826e47cbc2e8c1099` |
| [DeepSeek-V4.1-Flash-A3](https://github.com/wangzili121/DeepSeek-V4.1-Flash-A3) | `feature/orthrus-dsv41-a3` | `efe4e4a76b22b3ba32f4733c3aa32cfc27752ac7` |
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

在同一个 smoke 进程加 `--verify-gate` 可复用已加载权重，遍历当前 Verify 宽度的全部接受长度，并检查连续拒绝、跨页续算和教师行对齐。加 `--verify-width 16` 可用宽度 16 执行相同门槛。新版本的图清单会显示 `v41-a3-u8-paged128-prefix-commit-v2`。

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

`phase_seconds` 还分别记录 Draft 准备、上传与绑定、整图和回读，以及 Verify 准备与上传、整图、ID 回读和提交与教师回读。子项包含在对应的大阶段耗时中。

## K8 与 K16 冻结 head 对照

使用同一个初始化 head 和同一份留出集，将下面的命令在一个已分配的 16 卡组上执行一次。两组 Verify 都用 8 行，Draft 分别用 8 和 16 个有效位置。K8 的部分算子仍使用 16 行对齐的工作区。K8 当前只支持冻结推理，做在线实验时沿用上面的 K16 命令。按宽度训练的 head 要与导出的宽度一致。

```bash
for width in 8 16; do
  python "$STUDY" --serving-root "$SERVING_ROOT" --lib-root "$LIB_ROOT" \
    --data "$DATA_DIR" --frozen-only --output "$RESULTS/fixed-k$width.json" \
    --model-root "$CHECKPOINT" --devices "$DEVICES" --graph-dir "$GRAPH_DIR" \
    --max-model-len 512 --chunk-size 128 --prefill-moe-slab-rows 64 \
    --orthrus-head "$HEAD_DIR" --draft-block "$width" --verify-width 8 \
    --heldout-count 16 --tokens 64 || break
done
```

做短验收时可将 `--data "$DATA_DIR"` 换为 `--smoke --verify-gate`，输出用新文件名。正式对照看同题的 Draft 耗时、每轮接受长度、端到端吞吐和与 AR 的 greedy 一致性。

K16 的 embedding 和 Engram 两个输入字段为每 rank 140 KiB，之前为 1664 KiB。111 单卡用真实 checkpoint 的数据验证了 312 组 Engram 历史哈希，以及 K8/K16/K32 的主机准备和设备展开，结果逐位一致。K16 单 rank 这两个字段的上传中位数为 0.374 → 0.265 ms。该计时只覆盖输入传输。K8 的 SWA、C1A 和 C2A 在长历史、非连续物理页上通过拒绝后缀隔离检查。K8 SWA 与 Torch FP32 参考的输出最大绝对误差为 2.08e-6 以内。升级后让脚本用新目录编译，Draft 清单为 `a3-orthrus-draft-v2`。[输入验证数据](validation/dsv41/pypto-a3-compact-draft-seed.json) · [K8 缓存验证数据](validation/dsv41/pypto-a3-k8-cache-visibility.json)

在上一版 Lib `6d7dd1e` 上，使用真实 checkpoint 与同一份初始化 head，K8 冻结和 K16 last-1 均通过 Serving 的宿主启动预检，分别注册 3 张图的 3 个 callable 和 33 张图的 56 个 callable。

## 冻结 head 的多 DP 对照

在上一节的任一 `--frozen-only` 命令加 `--frozen-concurrency 4`。一个驻留 worker 内会用同一批题依次测 AR、单请求冻结 head 和四 DP 冻结 head。短验收用 `--smoke --verify-gate --frozen-only --frozen-concurrency 4`。报告比较逐 token greedy，并记录总体 tok/s、首 token 延迟、token 间隔 P95 和各阶段用时。HTTP 服务可用 `--max-num-seqs 4` 打开同样的逐轮批处理。在线更新仍按单活跃请求启动。

每个 DP 同时持有一个请求，空位可在一轮结束后立即补入新请求。需要回退到逐 token Decode 的请求，以一行输入和其他请求一起进入 Verify。Orthrus 的常规 Prefill 或单行 Decode 只回读 32 B 的采样记录。111 单卡单行回读与主机判定的中位数为 3.03 → 0.32 ms，输入与采样记录预先放在设备上，Python 侧使用一个 Torch 线程。在线更新的教师反馈按需回读全词表。[回读原始数据](validation/dsv41/pypto-a3-greedy-readback.json)

新版本的 Lib 和 Serving 通过了 34 个宿主测试，覆盖同轮四 DP、拒绝诊断、队列取消、活跃请求取消、slot 复用和由 Verify 转到逐 token Decode 后的教师反馈。四卡合成 logits 回放通过接受 0/3/7 个候选的算子对照。真实模型的 EP16 并行一致性和吞吐仍需实测。

## Greedy 采样算子

采样使用向量最大值与最早索引归约，并跳过未请求的 logits 行。111 单卡的 100 轮驻留对照中，1 个活跃行的设备窗口中位数为 307.77 → 78.75 μs，16 行为 307.84 → 81.32 μs。两个版本的阻塞调用中位数仍约为 16.5–16.7 ms。两组对比分开记录设备窗口与宿主提交，并通过并列、FP32 极值、NaN、Inf 及非连续采样行检查。[原始计时](validation/dsv41/pypto-a3-active-greedy.json)

用上一版 Lib 的 `6d7dd1e` 作基线。启用设备窗口记录时，将标准输出和错误输出写到 `--trace-log` 指定的文件。

```bash
git -C "$LIB_ROOT" show 6d7dd1e:models/deepseek_v4_1_flash_a3/lm_head.py > /tmp/dsv41-lm-head-baseline.py
SIMPLER_DEVICE_STRACE_ENABLE=1 PYTHONPATH="$LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python online-speculation/scripts/benchmark_dsv41_a3_greedy_sampler.py \
  --device 0 --baseline /tmp/dsv41-lm-head-baseline.py \
  --output "$RESULTS/greedy.json" --trace-log "$RESULTS/greedy.log" \
  > "$RESULTS/greedy.log" 2>&1
```

单卡和四卡的调度对照及 C2A 提交验证见 [调度实验](ORTHRUS_DSV41_PYPTO_A3_SCHEDULE.md)。

## 直接提交版本的验证

- 单卡 910B3 使用 W8A8 checkpoint 第 2 层验证 C2A。K8/K16 的全部有效前缀、跨逻辑页和非连续物理页共 29 组，检查保留的环形状态、FP4 数据、scale 和后续输出，结果逐位一致。单层首拒绝对照的中位数由 35.31 ms 降到 16.66 ms。[原始数据](validation/dsv41/pypto-a3-prefix-c2a.json)
- 单卡修改已提交前缀之外的数百个物理缓存行，Draft 的 SWA 输出及 C1A/C2A Indexer 的 Top-K 结果逐位一致。[原始数据](validation/dsv41/pypto-a3-prefix-cache-visibility.json)
- 单卡验证全词表 greedy 和设备有限值标志，覆盖跨词表分块并列及 NaN、正负无穷。原版与新版本的阻塞提交中位数均约 17.18 ms。[原始数据](validation/dsv41/pypto-a3-prefix-sampler.json)
- 单卡按 `1+3+3` 行收集真实路径所用的 TV 教师反馈。拒绝行填 NaN，前缀反馈与一次性输入同一批有效行的 GPU loss 和梯度逐位一致。相对 Torch 的梯度最大绝对误差为 `2.85e-8`。[原始数据](validation/dsv41/pypto-a3-prefix-online-teacher.json)
- Lib 和 Serving 联动的 27 个宿主测试通过。512 上下文下的 Prefill、Decode、Draft 和 last-1、last-2 在线图已通过编译。使用的是 111 的兼容工具链，上公司机时按 Serving 的 `pypto.pin` 构建。

该版本的 Draft 和 Verify 走紧凑回传，验证通过后直接发布有效长度。完整词表只为在线教师反馈和精度诊断按需读取。两个前向之间仍要在主机上按候选 token 查询 Engram，整轮吞吐和 EP16 精度由公司实机 smoke 验证。
