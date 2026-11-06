# Orthrus-Qwen3.8 逐层对照、head 重载与起草耗时

这组检查验证 PyPTO draft 能否正确读取现有训练链路导出的 head，以及替换 Full Attention 后的计算和开销。在同一个真实 Qwen3.8-27B worker 中，两种 Attention 后端使用相同的参数、输入和已提交历史。主模型仍使用固定 OneServing 的原生实现。

运行环境为 Ascend 910B3 64 GiB、CANN 9.1.1、驱动 26.1.1；依赖版本沿用[接入说明](QWEN38_PYPTO_PORT.md)。推理为 TP1、K32、单个活动请求；Full Attention 的临时块双向可见，GDN 按 head 契约使用 `orthrus_qwen35_reference` 因果递推。测试 head 来自少量离线更新，仅用于验证接入。

## 逐层计算

起草时保存每层实际收到的输入、输出，以及同一份目标历史。随后在 CPU 上用现有双视图层和 Transformers 5.17.0 回放：加载该层的共享 norm/MLP 和真实 draft 参数，执行相同的 Full Attention 或 GDN。每次只构造一层，不再加载第二份整网。

参考接口将原生卷积状态 `[tap, channel]` 转成 HF 的 `[batch, channel, tap]`，将递推矩阵 `[head, V, K]` 转成 `[batch, head, K, V]`；Full Attention 拼接已提交 KV 与当前临时块，关闭因果限制。RoPE 表按 HF 的约定转成 hidden dtype。所有历史都在副本上更新。

每个输出检查有限值、最大绝对误差和相对 L2 误差：

$$
e_{\mathrm{rel}}=\frac{\lVert O_{\mathrm{PyPTO}}-O_{\mathrm{HF}}\rVert_2}
{\max(\lVert O_{\mathrm{HF}}\rVert_2,10^{-12})}.
$$

误差门槛在运行前固定为 2%。两个后端各 64 层，共 128 项检查全部通过，最大相对 L2 误差约 0.285%；相关 CPU 回归共 357 项通过。这是每层使用相同输入的数值对照；完整 draft 的 31 × 248320 个有效 logits 另在两个 PyPTO 后端之间检查逐字节一致。

## head 加载与重启

将现有 TP4 head 精确重分片为 TP2 文件，再重启原生 TP1 worker 加载。转换按参数的行、列或 Q/K/V 分段规则切分，复制的 norm 保持一致，不改参数值；全部 rank 写完后才发布 manifest，拒绝覆盖已有目录。

重启前后，运行参数的 SHA256、初始目标历史、两个后端的最终 logits 和 31 个候选全部相同。两个后端都通过目标 cache 只读检查；验证后的输出与 target-only 一致，已提交历史与仅重放接受前缀一致。这里验证的是训练分片的换入与重启，执行仍为 TP1。

## 完整起草耗时

两种后端各预热一次，在同一 worker 中交错测量六轮，奇数轮反转顺序。计时包括参数上传、所有层计算、logits 回读和候选选择；启动、缓存校验、Torch 参考及逐层落盘均在计时外。

| 中位数 | 原生 Attention | 融合 Attention |
| --- | ---: | ---: |
| 完整 K32 起草 | 2.225 s | 2.311 s |
| 全部 16 层 Full Attention 入口 | 47.85 ms | 46.06 ms |
| 全部计算入口 | 338.90 ms | 339.00 ms |
| head 上传 | 1856.53 ms | 1940.14 ms |

计算入口的计时包含同步 worker 调用开销，不是设备纯计算时间。各行分别取中位数，不保证相加等于整轮中位数。

融合 Attention 的局部耗时较低，但本次完整起草没有取得收益。单卡 head 逐层上传占据主要时间，六轮样本不足以把整轮差异归因于 Attention。因此默认保留 `native`；`fused` 继续作为可选后端。完整 TP 和 head 驻留仍需随原生模型接口接通，不能由本次 TP1 结果推断多卡吞吐。

## 复现入口

在匹配的固定环境、项目目录中运行，替换模型、head、OneServing 和结果路径，设备编号使用已分配的逻辑编号：

```bash
# 同 worker 对照，并保存实际层输入供独立回放。
python scripts/compare_qwen38_pypto_draft.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B --head-dir /heads/tp4 \
  --device 0 --rounds 6 --capture --output-dir /results/original

# NPU worker 退出后执行，只使用 CPU。
python scripts/audit_qwen38_pypto_layers.py \
  --capture-dir /results/original/capture --model /models/Qwen3.8-27B \
  --head-dir /heads/tp4 --output /results/layers.json

# 导出相同参数的另一种训练分片布局，再重启检查。
python scripts/reshard_qwen38_pypto_head.py \
  --model /models/Qwen3.8-27B --head-dir /heads/tp4 --tp 2 --output /heads/tp2
python scripts/compare_qwen38_pypto_draft.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B --head-dir /heads/tp2 \
  --device 0 --rounds 2 --reload-reference /results/original/result.json \
  --output-dir /results/reloaded
```

原始数值、每轮耗时、运行版本及源码哈希见 `validation/pypto-qwen38/paired-20261106/`。该目录不包含模型权重或完整激活。两次 NPU 验证完成后，父进程和 worker 均退出，设备锁已释放；CPU 参考回放不初始化 NPU。
