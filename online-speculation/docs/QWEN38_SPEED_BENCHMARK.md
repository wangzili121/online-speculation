# Qwen3.8 Orthrus 接受长度与推理开销

这组测试区分两个问题：当前 head 能接受多少候选，以及给定更长的接受前缀，现有推理实现能否获得加速。只测试固定 head 的推理，不运行在线更新。

## 统计口径

K32 每轮提交 31 个候选。以每轮接受的候选数为 A，主要报告平均接受长度；当全部验证轮都提交满块时，同时报告接受候选总数 / 提交候选总数。各 TP rank 执行同一请求，计数只取一份，不能累加。

已有 TP2 功能检查中，50 个满块验证轮共接受 63 个候选，即平均每轮 1.26 个、候选利用率约 4.1%。这是少量更新后的测试 head 与功能提示的结果，不代表正式 head 的业务接受率。功能检查的总运行时间含加载、审计等，不用于换算生成速度。

测速比较三条路径：

- **Target**：原生 vLLM-Ascend 自回归生成。
- **Real head**：执行真实 Orthrus draft，由原生 verifier 决定接受前缀。
- **Oracle**：先执行同一套真实 draft，再以 Target 预先生成的正确候选替换提案，仍由原生 verifier 正常验证。它衡量全接受时的计算上限，不代表 head 的预测能力。

三组使用相同的 TP、模型、提示、输出预算和 KV 配置。模型加载、预热及计数读取在计时区间外，关闭缓存审计和参数哈希检查；计时包含请求的 prefill、起草、验证、恢复及引擎调度。输出 token 数除以请求耗时得到单请求端到端 tok/s，与只统计首 token 之后的 TPOT 口径分开。

## 实测结果

2026-11-05，在同一组 2×64 GiB Ascend A2 上运行 BF16 Qwen3.8-27B，TP2、eager、同步单请求、上下文上限 2048、KV 预算每 rank 8 GiB。固定提示为 `Count the integers from 1 to 100:`，忽略 EOS，定长输出 128 token；各组独立预热一次，再取三次测速的中位数。

| 路径 | 平均接受候选 / 验证轮 | 接受候选 / 提交候选 | 端到端 tok/s | 相对 Target |
| --- | ---: | ---: | ---: | ---: |
| Target | — | — | 4.27 | 1.00× |
| AR 复制初始化的 head，真实起草 | 1.08 | 3.49% | 2.91 | 0.68× |
| Oracle，保留真实起草计算 | 31.00 | 100% | 34.71 | 8.14× |

每次真实 head 请求有 61 轮满块验证，接受 66 个候选；Oracle 有 4 轮满块验证，接受 124 个候选。Oracle 每次请求仍实际执行 5 次 draft，未进入验证的末次提案也计入耗时。所有重复的 greedy 输出均与 Target 的 128-token 续写一致，TP rank 计数一致，在线更新次数为 0。

这说明当前初始化 head 的接受长度不足以覆盖额外计算，而当前实现可以在理想接受条件下获得净加速。Oracle 不是训练效果，也不代表业务加速比；本组只测一个固定提示与当前 eager 配置，不代表 vLLM 生产配置或多请求吞吐。原始计时、计数、源码哈希及释放记录见 [验证目录](validation/qwen38/speed-20261105/manifest.json)。

## 复现

在已适配的固定 vLLM-Ascend 环境中分配同一组 NPU，以两个独立进程依次运行：

```bash
python scripts/run_in_runtime.py scripts/benchmark_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 \
  --block-size 32 --max-model-len 2048 --kv-cache-memory-bytes 8589934592 \
  --stage target --tokens 128 --repeats 3 --output /results/target.json

python scripts/run_in_runtime.py scripts/benchmark_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 \
  --block-size 32 --max-model-len 2048 --kv-cache-memory-bytes 8589934592 \
  --stage orthrus --reference /results/target.json \
  --tokens 128 --repeats 3 --output /results/orthrus.json
```

每条路径先预热一次，再计时三次，保留每次耗时、输出数量和 worker 计数。Target 保存真实参考续写；Orthrus 检查 greedy 输出与该续写相同，Oracle 另外检查全部候选被接受。输出文件必须使用新路径。

未传入 head 时由主模型 mixer 复制初始化；可使用匹配 TP 的 `--head-dir` 测试已有 head。测试固定为 greedy，Oracle 的候选和概率处理不能直接用于随机采样。

## 如何解读

忽略请求首尾，生成速度近似为 `(1 + 平均接受长度) / 每轮总耗时`。增加接受长度会摊薄起草和验证的成本；即使接受长度为 31，真实 draft 前向与模型、缓存、调度开销仍必须保留。

常博源仓库在约 1.7B、BF16、单卡 CUDA、预训练 head 的配置下，报告贪心 AR 73.134 tok/s、投机 255.500 tok/s，约 3.49 倍；其实现复用 backbone 和历史 KV，并使用 SDPA/FA2 及整块张量校正。该结果证明了对应配置的净收益，不直接代表 27B、混合 GDN 模型或 NPU 的表现。条件与记录见[源仓库性能报告](https://github.com/xchang1121/inference_scaling/blob/main/online-speculation/docs/RESULTS.md)。
