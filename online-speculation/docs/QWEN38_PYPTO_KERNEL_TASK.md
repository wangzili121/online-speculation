# Orthrus-Qwen3.8 Attention 开发与调优任务

本任务优化 draft 的分页双向 Attention。主模型的投影、norm、GDN、MLP 和 LM head 继续复用原生实现；完整投机流程由外层负责起草、验证、提交与恢复。

## 输入输出

| 输入 | 当前布局 | 含义 |
| --- | --- | --- |
| Q、gate | `[32, 6144]` | 24 个 query heads，head dimension 为 256；Q 已完成 norm/RoPE |
| 临时 K、V | `[32, 1024]` | draft 当前块的 4 个 KV heads |
| 历史 K、V pool | `[L, 4, P, 16, 256]` | 直接接收主模型的分页池，以 cache_layer 选择层，页表给出物理页号 |
| 页表 | `[32]` | 仅前 `ceil(past/16)` 项有效，未用项允许为 -1 |
| 可见性 bias | `[1, 512]` FP32 | 前 `past + width` 项为 0，其余为 -10000；由调用方生成 |
| past、width、cache_layer | 标量 | 已提交历史长度、当前块有效长度及缓存层号；`past + width <= 512` |
| 输出 | `[32, 6144]` | 完成 Attention 和 sigmoid 门控，尚未经过 O projection |

所有 query 读取相同的已提交历史及整个有效临时块，历史和临时块参加同一次 softmax。有效块外的位置不参与计算。所有输入保持只读，临时 draft KV 不提交给主模型。

## 实现与独立验证

- `src/blockspec/pypto_qwen38/attention.py`：Torch SDPA 数学参考，以及保留原生 BF16 中间舍入的兼容参考。
- `src/blockspec/pypto_qwen38/attention_programs.py`：`fused` 和 `staged` 两种实现，共用相同的分页读取及临时块准备。前者融合 QK、softmax、PV、门控；后者将 QK、softmax、PV 分阶段执行，保留全量分数和概率中间张量，供同口径比较。
- `src/blockspec/pypto_qwen38/paged_attention_programs.py`：`paged` 将页表指定的历史及临时块直接搬入 L1，不先生成连续 K/V 副本；保留融合 Attention 和门控，按 `past + width` 选择 128、256、512 容量。支持 16/32 行 query tile，32 行使用一个跨核流水槽以满足 A2 Vector 缓冲上限。长度选择在编排层完成，各混合核保持固定形状。
- `scripts/validate_qwen38_pypto_attention.py`：固定种子生成独立输入，覆盖 13 组历史/块边界，包括容量切换前后；无效 cache 与 inactive draft 行填入 NaN，检查误读；执行后回读所有输入，检查逐字节不变。

在 `online-speculation/` 目录、已匹配的固定 PyPTO 环境中运行，设备编号使用实际分配的逻辑编号：

```bash
python scripts/validate_qwen38_pypto_attention.py \
  --device 0 --backend all --tune-paged --profile \
  --repeats 30 --output /results/attention.json
```

主指标是同一 worker、相同输入下的 resident dispatch 墙钟中位数，各后端交错测量，包含分页准备、Attention、门控和同步调用开销；编译、上传、回读、Torch 计算在计时外。逐 case 保留全部样本；该指标不等于设备纯计算时间或整网吞吐。

`--profile` 在计时之外另采集设备时间线，使用 `scripts/summarize_qwen38_pypto_profile.py` 汇总最后一组输入的任务窗口，不能将并行 block 耗时相加。模拟器只用于功能检查，不提供性能结论；驱动显式检查运行模式，避免平台参数将 SIM 覆盖为 ONBOARD。本次真实 NPU 已通过；当前固定主机环境缺少模拟器要求的 `g++-15`，不报告模拟器通过。

实测与原始数据见 [分页 Attention 验证](QWEN38_PYPTO_PAGED_VALIDATION.md)。

验证和 Torch 参考独立于候选实现。调优不得修改输入、可见性、舍入约定、golden、误差门槛或跳过只读检查。可探索 query tile、K/V 搬运、混合核流水和中间存储；改变 kernel 数量不改变被测入口范围。

## 接回完整模型

服务、完整 draft 验证与 benchmark 均提供 `--attention-backend native|fused|paged`，默认保留 `native`。`fused` 和 `paged` 替换 Full Attention 核心，后者同时将该层的准备和计算合为一次 HOST 提交；GDN 策略和 head 参数契约相同。O projection、共享 MLP、验证与回滚仍使用现有计算。

`--draft-group-size 4 --head-transfer packed` 在 `paged` 后端上进一步将 3 个 GDN 层和 1 个 Full Attention 层合并提交；这是执行编排配置，K 仍为 32。独立核的输入与参考保持不变；整网分配、参数视图和缓存检查见 [执行与验证](QWEN38_PYPTO_PAGED_VALIDATION.md#四层合并执行)。

独立 kernel 通过后，运行 `validate_qwen38_pypto_neural.py --attention-backend fused --audit-block` 检查完整 K32 起草、目标 cache 不变、拒绝后的状态和 greedy 续写；再对同一 head 比较完整 draft 及请求耗时。独立 kernel 的收益不能直接当作服务收益。

该任务可作为 Goal Plus Playground 的 `pypto-kernel` 场景任务输入。Playground 提供工作区和优化流程；本项目提供算法、固定版本接口、独立参考与验证入口，不沿用示例任务的 DSpark 形状、设备号或性能数字。
