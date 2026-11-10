# PyPTO 在线 last-N 验证记录

环境沿用 [last-1 记录](../online-last1-20261110/README.md) 中固定的 PyPTO、Simpler、CANN 和 A2 运行时。模型采用因果 GDN、块内双向 Full Attention，主模型及共享 backbone 参数冻结；在线更新使用实际到达位置的教师分布和 TV 目标。

## 组件与梯度

| 检查 | 覆盖内容 | 记录 |
| --- | --- | --- |
| GDN 递推 | 前向、保存状态及解析反向；48 个 value heads、16 个 Q/K heads、维度 128，以及小尺寸对照 | `gdn-lastn-core-small-r8.json`、`gdn-lastn-core-real-r9.json` |
| GDN 前后处理 | 四 tap 因果卷积/SiLU、decay/beta、gated RMSNorm 的前向和反向，与独立 Torch autograd 对照 | `gdn-lastn-aux-small-r11.json`、`gdn-lastn-aux-real-r11.json` |
| 完整后段 | last-2/3/4 的 15/24/33 组参数梯度、冻结模块上的输入梯度、两次 Adam 更新及 checkpoint 精确恢复 | `suffix-last2-r11.json`、`suffix-last2-adam-r15.json`、`suffix-last3-adam-r24.json`、`suffix-last4-adam-r16.json` |
| CPU 回归 | 472 项通过，包含 last-2/3/4 的解析梯度与独立 autograd 对照 | `source-lastn-r22-cpu-output.json`、`source-lastn-r22-cpu-output.txt` |

GDN 组件覆盖有效块宽 1/17/32、零及非零历史；未使用行填入 NaN，输入和历史逐字节保持不变。合成 last-4 的一个 TV 比较位置因 BF16 前向微差改变正负号；以相同输入逐节点检查，最大局部相对误差约 `1.84e-4`，TV 导数一致；以设备实际 TV 梯度驱动独立 autograd，全部 33 组梯度通过，最大相对误差约 2.3%。没有更换目标或放宽原有梯度阈值。

## 真实 27B 闭环

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| last-2 更新 | 两块 K32 反馈联合更新，版本 0→4；新 head 下一轮生效，目标历史不变，greedy、scheduler、热恢复和 HTTP 更新通过 | `online-last2-fixed-r19.json` |
| last-4 更新 | 末四层联合更新，版本 0→4；上述流程全部通过 | `online-last4-fixed-r20.json` |
| 冷启动恢复 | last-2、last-4 分别由新进程恢复版本 1，再更新到版本 5，包含后续调度及 HTTP 更新 | `online-last2-fixed-resume-r20.json`、`online-last4-fixed-resume-r21.json` |
| 恢复连续性 | 从同一步出发，连续运行与冷恢复所得下一步的 FP32 参数、两组 Adam moment 及 manifest 逐字节相同；完整 payload 另行读取并核验 SHA256 | `last2-continuity-r23.json`、`last4-continuity-r23.json` |
| head 导出 | last-4 的 33 组执行参数与 checkpoint 精确一致，其他区域和源 head 保持原字节，文件 SHA256 通过 | `last4-final-export-check-r21.json` |
| 独立服务 | 导出 head 在不加载学习器的情况下启动；completion、chat、SSE 均成功，3 个请求无错误、cache 已回收，服务正常关闭 | `last4-final-http-result.json` |

last-2 重放与服务 logits 的最大相对 L2 约 `7.85e-5`，last-4 约 `0.00229`。表中测试使用真实模型和已有 head，但请求较短，不用于评价训练收敛、接受率或加速收益。当前范围为 BF16、主模型 TP1、独立学习卡、K32、512 个总位置、greedy、一个活动请求；last-3 已通过 CPU 梯度测试，整模型服务未单独实测。原 last-1 的验收记录保持不变。

大参数缓冲也单独检查：`flat-zero-real-r19.json` 覆盖约 1.8 GB FP32、0.91 GB BF16 缓冲各两次完整清零；`flat-zero-special-r19.json` 覆盖 704 元素的小缓冲。初值来自独立主机字节填充，检查会读取所有字节。`flat-zero-legacy-r19.json` 是不适用于大扁平缓冲的旧入口的负向复现；正式更新使用新的有界任务网格，并在每次清梯度后检查整个缓冲的范数为零。

## 复现入口

在固定运行环境中执行，启动、恢复和导出配置见 [last-N 使用说明](../../../QWEN38_PYPTO_ONLINE_LASTN.md)。

```bash
python scripts/validate_qwen38_pypto_gdn_training.py --device 0 --real-shape --output core.json
python scripts/validate_qwen38_pypto_gdn_training_aux.py --device 0 --real-shape --output aux.json
python scripts/validate_qwen38_pypto_suffix_training.py --device 0 --last-layers 4 --output suffix.json

# 同一初始 head、相同反馈，在连续运行与冷恢复中各生成 step 2 后比较。
python scripts/validate_qwen38_pypto_checkpoint_continuity.py \
  --continuous /data/continuous/step-00000002 \
  --resumed /data/resumed/step-00000002 --output continuity.json

# 将导出的完整 head 与更新前 head、对应 checkpoint 一起对照。
python scripts/validate_qwen38_pypto_suffix_export.py \
  --model /models/Qwen3.8-27B --head-dir /models/initial-head \
  --exported-head /models/updated-head --checkpoint /data/step-00000003 \
  --output export.json
```
