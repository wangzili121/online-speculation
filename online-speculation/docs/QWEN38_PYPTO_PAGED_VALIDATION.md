# Orthrus 分页 Attention 验证

本次为 Orthrus 的 Full Attention 增加直接分页实现：通过页表将主模型历史与临时 draft 块搬入 L1，融合 QK、softmax、PV 和输出门控。历史与整个有效块参加同一个 softmax，目标 cache 保持只读。共享 backbone、GDN 策略及主模型源码不变。

## 独立算子

在一张 A2 上、同一 worker 中对照分阶段实现、融合实现、直接分页 16 行和 32 行四种配置。13 组输入覆盖空历史、部分有效块、非整页历史、128/256 容量切换及 512 位置上限，共 52 项检查通过；未使用的 cache 和块行填 NaN，所有输入回读逐字节不变。独立 Torch 参考及误差门槛保持不变。

各后端预热 5 次，交错测量 30 次。下表中墙钟列是 13 组输入各自中位数的范围，包含同步调度；设备窗口列是最长输入另一次开启时间线采集的单次记录，不是稳定性能中位数。

| 实现 | Resident dispatch 墙钟 | 单次设备任务窗口 |
| --- | --- | --- |
| 分阶段 `staged` | 0.963–1.060 ms | 203.28 μs |
| 融合 `fused` | 0.885–0.963 ms | 108.34 μs |
| 直接分页，16 行 | 0.827–0.911 ms | 64.08 μs |
| 直接分页，32 行 | 0.812–0.894 ms | 53.40 μs |

32 行方案减少了 query tile 数与重复历史搬运；用一个跨核流水槽满足 A2 的 Vector 缓冲限制。按有效长度选择 128/256/512 容量，避免短历史仍计算全长 512。这里的收益仅属于 Attention 入口，不能当成完整 draft 或请求吞吐收益。

原始交错样本、版本、形状及误差记录在 [component.json](validation/pypto-qwen38/paged-attention-20261107/component.json)，单次设备窗口及时间线源文件哈希在 [profile-summary.json](validation/pypto-qwen38/paged-attention-20261107/profile-summary.json)，完整时间线保存在 [device-profiles.zip](validation/pypto-qwen38/paged-attention-20261107/device-profiles.zip)。

## 完整模型

完整模型使用同一 worker、同一 head、同一段目标历史，先对照起草，再检查目标验证和接受前缀重放。直接分页实现另将 Full Attention 的 QKV 准备与该层后续计算合并为一次 HOST 提交，K32 的计算提交数由 82 降到 66；这不表示整网融合成了一个硬件 kernel。

选定的自适应 32 行配置已完成真实 27B 检查：六轮交错起草中，两条路径的有效 logits 和 31 个候选逐字节相同；全部目标 cache 在 draft 前后不变，目标验证和接受前缀重放通过。另用历史长度 15、17、96、97、224、225 的六个请求覆盖分页和容量切换，逐个检查 cache 字节及页表不变、校正结果、接受前缀重放、随后三步 AR 续写和请求释放，均通过。

完整起草中位数为原生 2.417 s、分页 2.401 s；计算提交累计耗时中位数为 346.6 / 323.6 ms，包含同步调用开销。约 2.05 s 用于 head 上传，不能将 Attention 局部收益直接当作服务收益。默认后端仍为 `native`。原始记录见 [paired.json](validation/pypto-qwen38/paged-attention-20261107/paired.json)，完整 logits 对照见 [logits-comparison.json](validation/pypto-qwen38/paged-attention-20261107/logits-comparison.json)。

## 复现

使用 [固定环境](QWEN38_PYPTO_PORT.md#固定基线)，在 `online-speculation/` 下运行：

```bash
python scripts/validate_qwen38_pypto_attention.py \
  --device 0 --backend all --tune-paged --profile --repeats 30 \
  --output /results/attention.json

python scripts/summarize_qwen38_pypto_profile.py \
  --component-result /results/attention.json --work-dir /runtime/work \
  --output /results/profile-summary.json

python scripts/compare_qwen38_pypto_draft.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --candidate-backend paged \
  --audit-boundaries --audit-verifier --capture --rounds 6 --output-dir /results/paired
```

目前完整服务仍是 TP1。单卡流式 head 上传仍占主要时间；完整 TP 与 head 驻留需要原生模型提供局部 backbone 和归约接口，不能由 Attention 核的提速代替。
