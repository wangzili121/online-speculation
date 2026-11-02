# Orthrus-Qwen3.8 on Ascend

Qwen3.8-27B 的 Orthrus 投机推理与 draft 更新实现。draft 复用冻结的主模型 backbone，推理读取主模型历史缓存，候选由原生 vLLM-Ascend 验证。支持在线 last-N 更新和独立离线全层更新，均只更新 draft mixer，不训练主模型。

## 验证状态

| 路径 | 当前状态 |
| --- | --- |
| 投机推理 | 真实 27B、TP2 已验证多种块长、缓存边界及 head 重载；K32 的部分数值一致性用例仍未通过，不能视为完整生产验收。 |
| 在线 last-N | last-1 / last-4 已在真实 27B 验证更新、冻结参数、只读历史和导出重载。固定反馈 32 步 TV loss 为 0.5044 → 0.4929，不代表业务加速收益。 |
| 离线全 draft | 实现了随机 anchor、完整块 KL 蒸馏、CPU 优化器与逐层 checkpoint；单元及双 TP 梯度测试已通过，真实 27B 短训与重载验收进行中。 |

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

`--checkpoint-dir` 按版本保存 head；使用 `--head-dir /data/online-heads/version16` 重启加载。加载要求基模哈希、TP、块长、mask ID 和混合层策略一致。

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

输出为 `loss.jsonl`、`summary.json` 和 `head/version<steps>`。在相同部署配置下，推理传 `--head-dir /data/draft-fit/head/version1000` 即可加载；离线也可用 `--head-dir` 从已有 head 继续学习，但不恢复 AdamW 状态，不能当作精确断点续训。

## 范围与文档

当前为 BF16、eager、同步单请求执行；排队不等于并行 batch。暂不支持量化、图模式、多模态或 PP/DP。Full Attention 块内双向，GDN 沿用显式的 `orthrus_qwen35_reference` 因果递推策略。

推理复用镜像已有融合 Attention、卷积、GDN 与 SwiGLU；反向不支持的融合步骤使用可微组合，不含自研 kernel，也不声称已达到最优性能。接入方式是插件加限定版本的内部兼容 hook，不是任意版本的纯配置插件。

- [接入与配置](online-speculation/docs/QWEN38_VLLM.md)
- [已有推理验收边界](online-speculation/docs/QWEN38_ACCEPTANCE_SUMMARY.md)
- [原仓库研究说明](README_RESEARCH.md)

模型分支分别为 `feature/orthrus-qwen38`、`feature/orthrus-dsv4`、`feature/orthrus-glm53`；保留公共代码与历史，后续按对应模型独立迭代。
