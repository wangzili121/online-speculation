# Qwen3.8 原生 vLLM 验证记录

## 2026-11-01 推理与在线更新功能复核

本轮区分三件事：推理与在线更新功能、不同执行形状的数值一致性、head 的训练效果。功能验收要求实际起草分布 q 正确交给原生 sampler，缓存和接受位置正确，教师来自原生主模型且监督位置对齐，只更新指定后缀的 draft 参数；不要求初始化 head 已经带来加速。

代码复核确认，q 在起草时保存，原生验证和采样使用主模型实际 logits；教师在采样约束原地修改 logits 前按 `target_logits_indices` 复制。接受 a 个候选后拒绝，监督 a+1 行；全接受时只监督候选对应的行，不把 bonus 或拒绝后的行加入训练。
新增 K32 的 31 个拒绝位置及全接受分支测试，同时检查非连续教师索引、原始 logits 被修改后的教师独立性，以及验证后 pending 释放。共享 backbone 仍传递输入梯度，但优化器及发布过程只处理所选 draft 参数。

同一生产代码在 167 的 4/5 卡重新加载真实 27B 和 version254，BF16、TP2、K32、8K、eager、同步单请求，更新最后四层、stride=1。六类请求覆盖 greedy、随机采样、top-k/top-p、32-token 输出、JSON 和跨页输入；两个 rank 各完成 75 次更新、97 次起草与历史读取、75 个完整验证块、66 次缓存只读检查，最终 TV loss 均为 `0.26185691`。
主模型每 rank 的全部加载参数约 26.90 GB，运行前后内容哈希不变，两个 head 分片哈希均改变。16 种子、32 个投机位置的采样校准通过。四个 greedy 用例仍保留 JSON 不同这一项，其他三个与同后端原基线一致；该次程序退出码 0 表示功能检查通过，不表示全部数值对照通过。没有开启附加 AR 重放诊断，也没有改动原有门槛。
最终源码的相关 CPU 回归为 220 passed、6 个 CUDA 专用测试 skipped。部署代码及镜像框架源文件哈希与上一轮一致；本轮仅增加诊断工具、回归测试和记录。更新后的临时 head 未另行导出，原 version254 保留。

[功能与数值检查汇总](validation/qwen38/acceptance-1101-functional-audit-summary.json) · [真实模型在线结果](validation/qwen38/acceptance-1101-functional-audit-online.json) · [运行日志](validation/qwen38/acceptance-1101-functional-audit-online.log.gz) · [CPU 回归](validation/qwen38/acceptance-1101-functional-final-cpu.xml)

### JSON 分叉的独立对照

`probe_qwen38_native_batch.py` 只启动单请求、无投机的原生主模型。运行时检查 drafter 为空，Orthrus 的 runner、Attention、GDN recurrent hook 均未安装；prefill 使用与原基线相同的 CANN 后端。
诊断时只把 decode 的原生线性投影及 LM head 的输入复制为多行，执行原有算子后取第一行；Attention、GDN 递推和正式缓存始终只处理原来的一行。复制行之间的线性计算互相独立，数学结果应相同，这样能单独观察矩阵形状造成的低精度数值变化。

| 原生线性计算行数 | JSON 第二个 token | 16-token 输出 |
| --- | ---: | --- |
| 不补齐、2、8、16 | 71093 | 与原无投机基线相同 |
| 32 | 90 | 与 Orthrus K32 的 JSON 输出相同 |
| 恢复不补齐 | 71093 | 恢复原输出，前两个位置的原始 logits 哈希也恢复 |

32 行时，71093、90 和 248068 的目标 logits 同为 16.625，原生 greedy 选择 90。该对照没有 draft、候选验证或拒绝回滚，因此确认原生矩阵形状变化本身足以复现这条分叉；不是依赖猜测把差异归给舍入。补齐没有进入正式代码，也不是性能建议，更不证明所有其他数值误差已完成归因。

[完整行数扫描](validation/qwen38/acceptance-1101-native-linear-width.json) · [最终源码重跑的 0/32/0 对照](validation/qwen38/acceptance-1101-native-linear-final.json) · [运行版本与源码哈希](validation/qwen38/acceptance-1101-functional-audit-runtime.json)

探索阶段还尝试了真正的多请求 batch。8/16 请求出现非有限 logits，32 请求出现设备错误；这超出当前单请求适配范围，不能用它证明 JSON 分叉的原因，也未将其列为通过。最终工具只保留上述单请求矩阵形状对照。[探索原始输出](validation/qwen38/acceptance-1101-native-batch.json.gz)

### AR 诊断的含义

在线 TV loss 比较 draft 分布与原生主模型教师分布；`0.085345` 则来自另一项比较：将 AR 权重映射到 Torch 执行器后，与原生 verifier 比较。生产起草不调用这条 AR 重放路径，在线教师也不来自它。
该附加诊断的 `0.05` 门槛仍未通过，保留原记录及非零退出状态，不放宽门槛，也不把诊断误差当作在线 loss 或接入故障。不同形状下 greedy 逐 token 不一致的记录也照实保留，不宣称所有数值检查通过。

验证进程全部退出后停止专用容器，2/3/4/5 卡均无本次 NPU 进程，原模型和 version254 保留，其他用户任务未操作。[资源释放记录](validation/qwen38/acceptance-1101-functional-audit-release.json)

## 2026-11-01 GQA 与受控数值对照

draft 的 Full Attention 直接将原始 KV 头传给现成 FAS。真实 27B 的 TP2 配置中，12 个 Q 头共享 2 个 KV 头，原先会先将整段 KV 复制成 12 头；现在取消这一步，不改变 attention mask 或 head 参数。
同时移除无梯度 RoPE 分支未使用的系数展开。未开发新 kernel，主模型及原生 verifier 的执行路径不变。

NPU 对照覆盖 0/11/491/4096 个历史位置、K32、因果和双向 mask，共八组：原始 GQA 与显式复制 KV 的前向逐元素一致，Q/K/V 梯度相对 L2 误差上限为 `0.00276`，在预设 `0.01` 容差内。
单层 Full Attention 对 CPU FP32 的输出及投影、Q/K norm 梯度检查通过；原生融合推理和可微分分支输出仍逐元素一致，历史缓存不变。相关 CPU 回归为 215 passed、7 个 CUDA 专用测试 skipped。

单卡微基准使用真实的头数及维度、K32 双向块，每种情况预热后交替测量五轮，每轮 100 次。下表为每次调用的中位耗时，不是整网 TPOT：

| 历史位置数 | 复制 KV 后计算 | 直接 GQA |
| ---: | ---: | ---: |
| 0 | 0.160 ms | 0.094 ms |
| 491 | 0.148 ms | 0.084 ms |
| 4096 | 0.165 ms | 0.141 ms |
| 8160 | 0.280 ms | 0.237 ms |

8160 个历史位置加 32 个当前位置时，Torch allocator 记录的单次临时分配峰值由 `127.2 MiB` 降为 `31.2 MiB`。这不是模型总 HBM，也不是业务吞吐提升；微基准在共享机器上运行，保留全部测量值。

数值定位继续固定同一组候选。以下对照中，原生 verifier logits 均逐元素不变；仅替换映射 AR 对照的计算，不修改正式 draft：

| 对照路径 | 最大 TV |
| --- | ---: |
| 原执行路径 | 0.059093 |
| Full Attention 使用整块投影形状 | 0.167937 |
| 再加入 GDN 整块投影 | 0.249569 |
| 再加入原生 GDN 门控 | 0.311801 |
| 仅保留两个残差分量并使用原生融合 norm | 0.147055 |
| 整块投影、门控及融合残差组合 | 0.178292 |

这些替换没有改善全网数值误差，没有作为默认实现。另在真实 5120 维、32 行上调用原生残差 norm：当前先舍入残差和的路径与它逐元素一致，改为保留未舍入 FP32 和的表达式则产生约 `0.00291` 的相对 L2 误差。
因此不能只凭某个 kernel 源码分支推断实际执行的舍入顺序，现有残差路径保持不变；这些反例也不能代替剩余误差的完整归因。

GQA 正式路径又在真实 27B、TP2、K32、8K 上加载 version254，两个 rank 各完成 195 次最后四层更新、230 次历史读取、194 个完整验证块和 101 次缓存只读检查。最终 TV loss `0.23841257` 在两 rank 一致，head 哈希改变，主模型加载参数哈希不变；16 种子、32 个投机位置的采样校准通过。
对照同镜像、同 CANN 后端的已保存无投机基线，16 个 greedy 用例共 400 token，仍为 15 个逐 token 一致，仅 JSON 在第二个 token 分叉。

**严格 AR 对照仍未通过：** 四个前缀的最大 TV 为 `0.085345`、`0.025647`、`0.021200`、`0.035092`，程序返回非零，保留 `completed=false`，门槛未放宽。
固定同一组旧候选后，新 GQA 路径的最大 TV 仍为 `0.059093`，与复制 KV 的路径完全相同；不能把在线更新后候选发生变化时某项 TV 的降低，解释成数值问题已经修复。

另用 `ORTHRUS_MODULE_PROMPT=Hello`、同一 version254 固定 head 重放真正未通过的短历史请求，两个 rank 均复现最大 TV `0.085345`。
96 个 GDN 初始递推状态逐元素一致；第 0 层的输入、norm、MLP 和输出在全部 32 行上一致，第 1 层输入 norm 仍一致，随后在 GDN 投影和前处理中出现最早局部差异。
该层 GDN 输出的同输入相对 L2 误差为 `1.38e-4`；之后全网差异逐层累积。这一请求没有发现初始递推状态偏移，不能把剩余误差全部归因于缓存边界；投影等数值路径仍需进一步对齐和归因。

[GQA 前后向与残差 norm](validation/qwen38/acceptance-1101-gqa-final-ops.json) · [微基准](validation/qwen38/acceptance-1101-gqa-bench.json) · [CPU 回归](validation/qwen38/acceptance-1101-gqa-cpu.xml) · [投影与门控对照](validation/qwen38/acceptance-1101-packed-projections.json.gz) · [融合残差对照](validation/qwen38/acceptance-1101-residual-only.json.gz)

[真实模型在线回归](validation/qwen38/acceptance-1101-gqa-last4.json) · [greedy 对照](validation/qwen38/acceptance-1101-gqa-greedy-comparison.json) · [固定旧候选复测](validation/qwen38/acceptance-1101-gqa-fixed-block.json.gz) · [短历史定位](validation/qwen38/acceptance-1101-hello-stage.json.gz) · [运行版本与源码哈希](validation/qwen38/acceptance-1101-gqa-runtime-lock.json)

全部实验结束后停止专用容器，使用过的 2、3、4、5 卡均无本次 NPU 任务，模型和 head 保留。
[在线回归日志](validation/qwen38/acceptance-1101-gqa-last4.log.gz) · [资源释放](validation/qwen38/acceptance-1101-gqa-resource-release.json)

## 2026-10-31 原生算子执行路径与在线反向

将逐模块定位得到的改动接入实际 draft 执行器，不再仅通过诊断开关替换。共享 MLP 使用 packed gate/up 投影和 `npu_swiglu`；GDN 使用现成卷积、L2 norm、递推及门控 norm，状态始终为自有临时存储。
在线更新需要反向时，保留可微分表达式，并对齐融合实现的中间精度。没有新增 kernel，也没有修改安装目录内的框架源码。

进一步发现，镜像的 `patch_qwen3_5.py` 已将 Full Attention 的 QKV 拆分、Q/K norm 和 RoPE 融为一个算子，实际不会经过独立的 Q/K norm 模块。
其 gamma 先以 BF16 计算 `1 + weight`，norm 与 RoPE 之间则保留 FP32 中间结果；将两步分开并提前转回 BF16 会产生额外舍入。
无梯度路径现复用该原生融合算子，在线反向按相同顺序计算，再调用支持反向的 `npu_fusion_attention`。RoPE 仅共享只读系数表，不共享有状态的执行模块。

固定同一组旧候选块，接入卷积、MLP 和 GDN 适配后最大 AR TV 为 `0.150247`；接入上述 Full Attention 融合路径后为 `0.059093`，原路径为 `0.298912`。
Full Attention 同输入 Q/K 的最大相对 L2 误差分别降至 `9.17e-5`、`1.70e-4`，V 逐元素一致。
将完全相同的原生 Q/K/V 交给 FAS，Attention 输出与原生验证器的最大相对 L2 误差为 `3.90e-5`。剩余全网差异仍不能据此全部归因于某个算子，`0.05` 门槛不变。

NPU 单算子验证覆盖：

- 卷积长度 1/4/17/32/36/65，包含超过一个分片的输入；融合前向与 FP32 卷积加 SiLU 参考逐元素一致，输入和权重梯度有限且非零。
- GDN 长度 1/8/17/32，包含非零历史状态；前向相对 L2 误差不超过 `1.82e-5`，最终状态不超过 `8.37e-7`，Q/K/V、g、beta 均有有效梯度，历史未被改写。
- Full Attention 同时覆盖因果和块内双向 mask、11 个历史位置、32 个当前位置、GQA 和部分 RoPE。原生融合前向与可微分分支输出逐元素一致；对 CPU FP32 参考，输出相对 L2 误差不超过 `0.00474`，各投影及 Q/K norm 权重梯度不超过 `0.00891`，历史未被改写。
- SwiGLU、L2 norm、门控 norm 覆盖 1/16/32 行；前向对照及反向检查通过。

相关 CPU 回归为 214 passed、7 个 CUDA 专用测试 skipped，不与其他批次累加。
head 参数名和交付格式不变；上述数值对照不是业务接受率、吞吐收益或最优 kernel 的证明。

真实 27B 再次加载 version254，更新最后四层，TP2、K32、8K、eager，覆盖短请求、跨页、分块 prefill、接近上下文上限、128-token decode 和 1/2/31/32/33-token 输出。
两个 rank 各完成 193 次更新、228 次历史读取、192 个完整验证块、101 次缓存只读检查；计数及最终 TV loss `0.36623648` 一致。
head 内容哈希改变，主模型全部加载参数的哈希不变。16 种子、32 个投机输出位置的条件概率频数检查通过。
新跑的无投机基线与在线测试对照，16 个 greedy 用例共 400 token，15 个逐 token 相同，仍只有 JSON 在第二个 token 分叉。

**严格 AR 对照仍未通过，运行返回非零且 `completed=false`。** 四个前缀的最大 TV 分别为 `0.085345`、`0.025647`、`0.021200`、`0.067469`；实际到达行均不超过 `0.028181`，但不能用到达行结果替代全部候选行的 `0.05` 门槛。
此项比较的是映射 AR 与原生 verifier 的执行数值；实际验证、采样及在线监督的教师 logits 都来自原生主模型，不使用映射 AR 对照的分布。
剩余差异尚未完全定位，不将它全部归因于主模型；JSON 的原生不同宽度问题也仍单独保留。

同一代码重新启动 OpenAI API，加载原 head、最后四层更新、stride=16。15 项接口检查通过，覆盖普通生成、top-k/top-p、logprobs、chat、SSE、双客户端排队、1804-token 跨页输入，以及五轮取消、非法请求后恢复。
服务期间两个 rank 各实际更新 4 次，计数、loss 和梯度范数一致；完成后向本次服务发送 SIGTERM，主动停服退出码为 0。关闭日志仍出现原生异步输出任务的 `EngineDeadError` 和 resource tracker 提示，不将退出码为 0 描述为零报错。

[融合 Q/K 前的固定块](validation/qwen38/acceptance-1031-adapted-fixed-candidates.json.gz) · [融合后的固定块](validation/qwen38/acceptance-1031-adapted-fused-qk.json.gz) · [NPU 前向与反向](validation/qwen38/acceptance-1031-fused-execution-ops.json) · [CPU 回归](validation/qwen38/acceptance-1031-fused-final-cpu.xml)

[新无投机基线](validation/qwen38/acceptance-1031-fused-final-target.json) · [最后四层更新及失败门槛](validation/qwen38/acceptance-1031-fused-final-last4.json) · [greedy 逐用例对照](validation/qwen38/acceptance-1031-fused-greedy-comparison.json) · [API](validation/qwen38/acceptance-1031-fused-api.json) · [运行版本和源码哈希](validation/qwen38/acceptance-1031-fused-runtime-lock.json)

本轮测试后已停止专用容器，实查使用过的 2、3、4、5 卡无 NPU 进程，模型及 head 保留，未停止其他用户任务。
[在线测试日志](validation/qwen38/acceptance-1031-fused-final-last4.log.gz) · [API 与停服日志](validation/qwen38/acceptance-1031-fused-api-server.log.gz) · [资源释放](validation/qwen38/acceptance-1031-fused-resource-release.json)

## 2026-10-31 同操作数逐模块定位

使用 167 的 2、3 卡运行真实 27B、TP2、K32、8K、eager 对照，不进行在线更新。
先记录原生验证器每一层的输入，再让映射模块直接使用这些输入和同一份历史；这样局部误差不会混入前面各层累积的偏差。
两 rank 共 96 个 GDN 层的初始递推状态逐元素一致。相同真实 Q/K/V、门控和初始状态下，原生递推输出与 PyTorch 递推/分块公式输出的最大相对 L2 误差分别为 `4.58e-5`、`5.09e-5`。
这批输入没有发现历史状态偏移或 GDN 递推公式、K32 分片边界的明显错误；不据此排除所有请求和状态组合的问题。

更大的局部误差出现在前后处理：Q/K L2 归一化约 `0.0033`，GDN 门控输出 norm 约 `0.0047`，MLP 约 `0.0042`，均为同输入下的相对 L2 误差上限。
HF 组合路径会在部分中间量上先舍入到 BF16；原生融合实现的计算和舍入顺序不同。
packed/split projection 的局部误差约 `1e-4` 或更小，也不是全网偏差的唯一来源。

随后固定同一组 31 个候选及 32 行输入，分别替换映射 AR 的计算路径。各组原生 verifier logits 逐元素相同，最大 TV 如下：

| 映射 AR 对照方式 | 最大 TV |
| --- | ---: |
| 当前执行路径 | 0.298912 |
| 仅 Q/K 使用 FP32 归一化中间量 | 0.297735 |
| 仅 GDN 门控输出 norm 保留 FP32 中间量 | 0.251562 |
| 仅 MLP 使用 packed projection 和现成 `npu_swiglu` | 0.291641 |
| 仅卷积与 SiLU 保留 FP32 中间量 | 0.305654 |
| 上述四项同时使用 | 0.058304 |
| 四项中不替换卷积路径 | 0.341542 |
| 四项再复用原生 Full Attention 投影、Q/K norm 和 RoPE | 0.055403 |
| 使用现成融合卷积、L2 norm、GDN 输出 norm 及原生 Full Attention 投影路径 | 0.055403 |
| 上一项再改为跨层保留两个残差分量并调用原生融合 norm | 0.233497 |

现成融合卷积仅使用自有临时状态，不借用主模型的持久 cache；单算子随机输入还与 FP32 卷积加 SiLU 参考结果逐元素一致。
这些结果证明前后处理的计算精度和融合方式是明确的偏差来源，但还未完全解释剩余误差。
单项替换可能变差，组合后的改善也不是各项贡献的简单相加；融合残差这组反例同样保留。
`0.05` 门槛没有放宽，以上均为诊断选项，没有设为生产默认，也没有验证这些无梯度诊断组合的在线反向或性能收益。

各组仍复现 JSON 的原生 greedy 分叉：decode 输出 `[271, 71093]`，K32 输出 `[271, 90]`。
原生不同 batch 宽度的数值变化与映射 AR 对照偏差是两个问题；对齐后者不能宣称修复前者。
本轮未改动 vLLM/vLLM-Ascend 源码及生产 proposer，未开发新 kernel。

复现入口：`scripts/probe_qwen38_module_replay.py`，参数与下节相同。
设置 `ORTHRUS_MODULE_REPLAY_OUTPUT` 指定结果文件；`ORTHRUS_MODULE_VARIANTS=1` 运行逐项替换，
`ORTHRUS_MODULE_FOLLOWUP=1`、`ORTHRUS_MODULE_NATIVE_OPS=1`、`ORTHRUS_MODULE_RESIDUAL=1` 分别运行后续三组对照。
JSON 同时保存候选 token、逐层局部误差、原生 logits 不变检查及诊断源码 SHA256；局部模块的 `flattened_rows` 指展平后的行，早期文件中同一字段名为 `rows`，不能当作 token 行号。
相关回归为 230 passed、7 个 CUDA 专用测试 skipped；最后新增的诊断 bookkeeping 检查随工具用例另验 37 passed，两组存在重叠，不相加计数。

[逐项替换](validation/qwen38/acceptance-1031-module-precision.json.gz) · [卷积及投影对照](validation/qwen38/acceptance-1031-module-followup.json.gz) · [现成融合算子](validation/qwen38/acceptance-1031-module-native-ops.json.gz) · [融合残差反例](validation/qwen38/acceptance-1031-module-residual.json.gz)

[相关回归](validation/qwen38/acceptance-1031-module-complete-cpu.xml) · [诊断工具回归](validation/qwen38/acceptance-1031-module-tools-cpu.xml)

本次定位结束后停止专用容器，2、3 卡均无 NPU 进程，未停止其他用户的任务，模型和 head 保留。
[资源释放](validation/qwen38/acceptance-1031-module-resource-release.json) · [融合卷积单算子前向](validation/qwen38/acceptance-1031-module-conv-op.json)

## 2026-10-31 同宽度数值复查

在 167 的 0、4 卡恢复真实 27B、TP2、K32、8K、eager 单请求验证。固定复制初始化的 head，不进行在线更新。
同模型宽度对照再次复现：0/1/7/15/16 个候选保持原输出，31 个候选在同一 JSON 位置分叉。
逐层记录表明，普通 decode 与 K32 的首行 embedding、第一处 input norm 输出相同；第 0 层 GDN 后的 post norm 首次出现可见差异，最大绝对误差为 0.000732421875。
这比单独比较最终 logits 更接近差异来源，但还不能据此把全部最终误差归因于某一个 kernel。

修正 AR 对照的输入宽度：原脚本只重放 anchor 和前 30 个候选，现包含全部 31 个候选，与原生 verifier 同为 32 行，再比较前 31 行 logits。
同宽度最大 TV=0.11111507，首行 TV=0.03195545；预设的 0.05 门槛仍未通过，没有用放宽阈值处理。
诊断还覆盖第 15、16、26 行，不再只观察首行。旧的 0.09108947 记录保留为不同输入宽度下的历史结果。

在相同的原生输入上分别控制 gamma 和归一化中间结果的舍入，128 处 norm 中有 121 处与“两处均先舍入到 BF16”的表达式逐元素一致；第一处非残差 norm 则与 HF 表达式一致。
这说明原生 fused 分支与参考表达式确有数值路径区别，不代表直接把所有 norm 换成某个 fused 调用就能消除全网误差。

固定修复前 head 提出的候选块，四种归一化实验的原生 verifier logits 完全相同。只对齐残差 norm 的舍入时，AR 最大 TV 从 0.111115 降到 0.047429；
直接对所有 norm 使用 `npu_rms_norm(x, 1+w)` 则为 0.052653。这是同一候选块的受控实验，不是通用通过结论。
把候选全部改为另一组 token，原生前一位置的 320 处逐层记录仍逐元素相同；该用例未发现候选未来信息泄漏。

随后将共享残差 norm 接到现成的 `torch_npu.npu_rms_norm(x, ones)`，再乘 BF16 的 `1+w`；第一层输入及 Q/K norm 保留原路径。
1/16/32 行、隐藏维度 5120 的单算子前向与目标舍入表达式逐元素一致，输入梯度相对 L2 误差为 0.00266--0.00311，在预设容差内。
只增加非持久的单位 gamma buffer，不改变共享主模型参数、head tensor 格式、Attention、cache 或 rejection sampler；未开发新 kernel。

**扩大验证后，AR 数值门槛仍未通过。** 新执行路径会改变 draft 提出的候选，不能与前一段固定候选实验直接比较最大 TV。
复制初始化 head 的 JSON K32 对照最大 TV=0.298912；加载 version254 的扩展测试中，四个前缀的最大 TV 分别为 0.176190、0.030569、0.032436、0.290049。
不能把单个受控块的 0.047429 登记为整个执行器通过，也不能把所有误差都归因于 norm；后续仍需隔离 GDN 与 packed/split projection 的数值路径。

加载 version254 后，两个 rank 各完成 195 次最后四层更新、194 个完整验证块和 101 次缓存只读检查，计数及最终 loss 一致。
16 个 greedy 用例共 400 token，仍只有同一 JSON 用例与 target-only 不同；其他 15 个用例逐 token 一致。
第一次扩展运行的 JSON 在采样校准前落盘，因此其中计数为 163；195 次更新和最终失败门槛见原始日志，不把中间快照当作最终结果。
验收工具现会在最终断言前保存采样统计及权重哈希，另有回归确认失败文件仍包含这些字段。
修正保存逻辑后再次加载 version254，四个请求加 16 种子采样校准，两 rank 各完成 53 次更新、53 个完整验证块和 60 次缓存只读检查。
主模型两分片合计 53,793,351,168 字节的参数哈希不变，head 内容哈希改变；32 个投机输出位置的条件概率频数检查通过。
这四个请求中，实际到达行的最大 AR TV 为 0.028735；但全部候选行的最大 TV 仍为 0.176190，因此程序保持非零退出和 `completed=false`，没有用到达行结果替代原门槛。
相关回归为 225 passed、7 个 CUDA 专用测试 skipped；Ruff 与 `git diff --check` 通过。

复现入口（与部署环境和版本锁定文件配合使用）：

```bash
ORTHRUS_LAYERWISE_VARIANTS=1 ORTHRUS_LAYERWISE_OUTPUT=/work/evidence/layerwise.json \
python scripts/probe_qwen38_layerwise.py \
  --model /models/Qwen3.8-27B --mask-token-id 1 --tp 2 --block-size 32 \
  --max-model-len 8192 --kv-cache-memory-bytes 12884901888
```

[宽度复测](validation/qwen38/acceptance-1031-resume-width.json) · [修复前逐层记录](validation/qwen38/acceptance-1031-layerwise-aligned.json.gz) · [固定候选受控实验](validation/qwen38/acceptance-1031-layerwise-variants-v3.json.gz) · [现成 RMSNorm 前后向](validation/qwen38/acceptance-1031-norm-op.log.gz)

[新路径逐层记录](validation/qwen38/acceptance-1031-norm-layerwise.json.gz) · [195 次更新及失败门槛的原始日志](validation/qwen38/acceptance-1031-norm-last4.log.gz) · [最终回归](validation/qwen38/acceptance-1031-norm-release-cpu.xml) · [版本锁定](validation/qwen38/acceptance-1031-norm-runtime-lock.json)

[53 次更新、冻结哈希、采样及未通过门槛的完整结果](validation/qwen38/acceptance-1031-norm-final-last4.json)

本轮结束后已停止专用容器；最终使用的 167 的 4、5 卡实查没有 NPU 进程，只剩基础显存占用，模型及 version254 保留。
验证脚本因数值门槛返回 1；容器的 137 则来自随后主动停止 `sleep infinity`，`OOMKilled=false`，不将二者混为模型 OOM。
[资源释放记录](validation/qwen38/acceptance-1031-norm-resource-release.json)

## 2026-10-31 扩展验收

**扩展验收尚未全部通过，不能用下方较早的通过记录代替新用例的结果。**
真实 27B、TP2、8K 上下文的 K32 检查中，16 个 greedy 用例有 15 个逐 token 一致，JSON 提示在第二个输出 token 处分叉。
同一模型、同一提示、同一 draft head，只改变实际提交的候选数：0/1/7/15/16 个候选均与无投机一致，31 个候选出现分叉。
无投机该位置的前两名目标 logits 为 16.625 和 16.5；32 行验证时出现三个 16.625 的并列值，原生 argmax 选择发生变化。
进一步固定第一层原生投影的输入和权重，直接调用 `torch.nn.functional.linear`，仅改变 batch 行数，也复现了最大 0.03125 的输出变化。
实测各块长度进入第一层投影的输入按字节相同；原生调用与独立 linear 检查均出现该变化。
这确认了目标后端的 batch 形状相关数值变化是一个明确来源，不涉及 draft mask 或历史缓存读取；尚未逐层归因全部最终 logit 误差。

同一 K32 运行的 64 种子、128 个投机输出位置的采样统计通过；参考概率直接取自原生采样前 logits。
固定 head 的相同 seed 请求复现检查通过；两个 TP rank 的计数一致，每 rank 248 次缓存只读抽查通过。
两 rank 全量主模型参数运行前后 SHA256 相同，覆盖合计 53,793,351,168 字节。
映射 AR 视图与原生验证的额外数值对照未达到预设 TV<=0.05：31 行最大 TV=0.09108947，首行 TV=0.02189109。
这项诊断不能登记为通过，也不能仅凭阈值超出便认定参数映射有误；需与目标后端的块长度数值差异一同分析。

相关回归为 224 passed、16 个 CUDA 专用测试 skipped。
原始记录：[K32 严格对照](validation/qwen38/acceptance-1031-k32-fixed-strict.json)、[同模型块宽度对照](validation/qwen38/acceptance-1031-json-width.json)、[原生投影与独立 linear](validation/qwen38/acceptance-1031-json-projection.json)、[最新回归结果](validation/qwen38/acceptance-1031-final-v4-cpu.xml)。

另有不加载模型的单卡数值复现：`python scripts/probe_npu_linear_batch.py --device 0`。
本机随机矩阵的 batch 差异最大为 2.86e-6，见 [单算子记录](validation/qwen38/acceptance-1031-linear-synthetic.json)；真实模型投影的 0.03125 来自另一组权重和输入，不把二者当作同一误差指标。
新增回归覆盖固定反馈失败仍返回诊断结果；验证入口先保存 checkpoint，再执行最终门槛，保留诊断失败时的状态。

镜像内原生 sampler 的 31 个拒绝位置、全接受 bonus 和 32768 次分布抽样通过；另一个混合 batch 的零候选 RNG 检查失败，未登记为通过。
该失败由直接调用原生 sampler 复现，不经过本插件，且不属于当前 `max_num_seqs=1` 的声明范围；本轮未修改它。
`logprob_token_ids` 与投机组合、受限词表的 processed-logprobs 也未纳入已支持范围，镜像内实测出现过 bonus shape 错误或返回项不完整。
last-1、stride=16 的在线边界检查已完成：14 个 greedy 请求合计 215 token 与基线一致，另含 3 个 top-k/top-p 请求。
两个 rank 各验证 111 轮、更新 6 次，计数及最终 loss 一致；head 哈希改变，主模型全量权重哈希不变，每 rank 52 次缓存只读抽查通过。
五类请求边界各运行三轮，第二、三轮活跃内存的最大增长为 4,136,448 字节，低于 64 MiB 门槛；这是有界复测，不是长时间无泄漏证明。
两个分片及 manifest 已导出到 version6。见 [last-1 原始记录](validation/qwen38/acceptance-1031-last1-online.json)。
last-4 的第一次扩展运行在固定反馈验收失败：沿用已有 AdamW 状态重复更新 8 次，TV loss 从 0.51410162 升至 0.51602209，两 rank 分别有 25、26 个执行参数变化，缓存仍按字节不变。
这是未通过项，不将“参数能更新”当成“loss 已下降”；完整记录见 [失败摘要](validation/qwen38/acceptance-1031-replay8-failed.json)及 [原始日志](validation/qwen38/acceptance-1031-replay8-failed.log.gz)。
32 步复测完成：两 rank 各有 222 次请求触发更新和 32 次固定反馈更新，TV loss 从 0.51410162 降至 0.50858486；两 rank loss 序列一致，缓存不变。
8 步处的 0.51602209 在复测曲线中精确复现，后续 loss 仍有波动，不将单样本改善称为业务收敛。
该运行的主模型全量参数哈希不变，每 rank 83 次缓存只读抽查通过；24 个 greedy 请求中，除同一 JSON 数值边界外其余 23 个对齐。
见 [last-4 引擎执行记录](validation/qwen38/acceptance-1031-last4-online.json)及 [分项判定](validation/qwen38/acceptance-1031-functional-matrix.json)。引擎执行完成不代表外部数值或内存门槛全部通过。

此运行的重复请求内存门槛未过：最大增长 184,753,152 字节。定位到采样适配每轮定义局部类，其闭包让完整 q 表等待循环 GC。
改用模块级代理后，关闭循环 GC 的弱引用回归通过；临时换回旧实现可稳定复现失败，没有靠强制 GC 或放宽 64 MiB 门槛处理。
新进程加载 version254，两个 TP 分片内容哈希与导出时完全相同，再执行 85 次在线更新、五轮交替请求及采样校验；主模型哈希不变，每 rank 78 次缓存只读抽查通过。
最后两轮同类请求边界的活跃内存增长为 0；暖机后最大波动为 564,224 字节。
见 [修复后真机记录](validation/qwen38/acceptance-1031-memory-fixed-online.json)和 [内存判定](validation/qwen38/acceptance-1031-memory-fixed-check.json)。

受控全接受及拒绝分支均通过：全接受 4 轮、124 个候选，四次从接受 32 个输入的位置继续；连续拒绝 127 轮、接受 0 个候选，均从接受 1 个输入的位置继续。
两组各 128 个输出 token 均与无投机基线一致，见 [全接受](validation/qwen38/acceptance-1031-branch-oracle.json)、[连续拒绝](validation/qwen38/acceptance-1031-branch-reject.json)。这些候选是验收夹具，不是训练 head 的接受率证据。

原生 OpenAI API 的 15 项检查通过，包含 chat、普通 logprobs、SSE、两个客户端排队、1804-token 跨页输入及五轮取消/非法请求后恢复。
加载 version254 后使用 last-1、stride=16，两 rank 各实际更新 4 次，计数、loss、梯度范数一致。
最终服务的 JSON 数值探针仍与 target-only 不同，保留为未通过项；停服退出码为 0，关闭期出现原生异步 output-handler 错误和 resource-tracker 警告，不声称停服零报错。
本次专用服务及容器已停止，实查 167 的 0、4 卡没有运行进程；61、111 没有本次仍在运行的验收容器。临时 pre-replay version222 已清理，最终 version254 和 last-1 version6 保留。
见 [API 检查](validation/qwen38/acceptance-1031-api-167.json)、[服务在线更新](validation/qwen38/acceptance-1031-api-online-updates.json)、[最终 JSON 探针](validation/qwen38/acceptance-1031-api-json-diagnostic.json)及 [资源释放检查](validation/qwen38/acceptance-1031-resource-release.json)。
最终代码、框架 commit、镜像和 67 个源文件的 SHA256 见 [版本锁定](validation/qwen38/acceptance-1031-runtime-lock-final.json)；实际部署源码与本地逐文件一致，三个关键框架源文件哈希未改变。

以下为 2026-10-30 及之前已完成的验证范围。

日期：2026-10-30。对象为真实 Qwen3.8-27B BF16 checkpoint，不是缩小模型。

## 当前判断

**真实 27B 在 BF16、TP2、K32、eager、同步单请求调度下，原生 vLLM 起草、验证、拒绝恢复、在线更新和 head 保存重载已通过功能验证。**
不是将独立 Torch 执行器包装成 HTTP：主模型、正式缓存和 rejection sampler 均由原生 vLLM-Ascend 执行。
原生 OpenAI API 的 8 类接口检查也全部通过。本记录不代表生产性能验收。

## 扩展边界与修复

2026-10-30 的扩展检查使用 61 的 4、5、6、7 卡运行两组 TP2，不是 TP4。
新增 128-token 输出、4092-token 输入加 4-token 输出、停止后重新请求，以及固定反馈的重复更新检查。

修复后的完整 last-4 在线回归通过：7 个 greedy 用例共 220 个输出 token 与 target-only 逐 token 相同，另含 temperature=1 和 top-k/top-p 采样请求。
每 rank 起草及读取历史 120 次，验证 110 轮，其中 109 轮为完整 K32；实际请求触发 110 次更新，固定反馈再更新 8 次，共 118 次，两个 rank 的计数及 loss 一致。
每 rank 的 28 次抽查均确认主模型 cache 按字节不变；固定反馈更新前后的 cache 检查也通过。
同一反馈的 TV loss 从 0.5937114 降至 0.5093935，选中的 draft 参数发生变化；中间 loss 并不单调，这只是固定样本更新检查，不是业务收敛或吞吐收益证据。
更新后的两个 head 分片与 manifest 已保存为 `heads-final-61/version118`。
原始结果见 [完整边界基线](validation/qwen38/acceptance-soak-target-61.json)和 [last-4 在线回归](validation/qwen38/acceptance-final-last4-61.json)。

重启服务加载 version47，使用最后 1 层、默认 `stride=16`，两个 rank 各实际更新 3 次，计数、loss 和梯度范数一致。
[API 8 类检查](validation/qwen38/acceptance-api-final-61.json)全部通过：health/model、greedy、top-k/top-p、SSE、两个客户端排队、取消后恢复、1804-token 跨页输入及后续请求隔离。
两个客户端仍是在 `max_num_seqs=1` 下排队，不代表动态批处理已实现。

4K 边界检查发现并修复了 runner 适配问题：发生拒绝后，下一轮可能不再有候选，只有一个普通 decode token。
原适配把它与 prefill 一起标为 -1；原生 metadata 调用又因本轮没有候选而不传上轮接受位置，导致 GDN 从错误的状态槽继续。
修复后，prefill 保持 -1，无候选 decode 使用 0，并继续向 GDN builder 传递接受位置；没有修改 Attention、卷积或 GDN kernel。

固定 head、关闭在线更新时，修复前的边界输出为 `[3,3,3,3]`，修复后为 `[3,3,3,271]`，与 target-only 和原生 n-gram 对照一致。
最后一步目标 logits 的首选 token 也恢复为 271；因此这次差异不能归因于在线训练或普通舍入误差。
对应原始结果为 [修复前](validation/qwen38/acceptance-context-fixed-61.json)、[原生 n-gram 对照](validation/qwen38/acceptance-context-ngram-61.json)和 [修复后](validation/qwen38/acceptance-context-fixed-repaired-v2-61.json)。

一次服务启动的自动 KV 预算为 -5.14 GiB，尚未进入请求执行。该估值的根因未独立确认；扩展部署改用 vLLM 原有的显式 KV 预算，每 rank 12 GiB，仍执行 profiling forward，不修改预算算法。
此配置已完成固定 head 重载和服务启动。它仅适用于本次 TP2/4K/单请求配置，不能直接外推到其他规模。

修复后的 CPU 回归为 182 passed、7 个 CUDA 专用测试 skipped，Ruff 与 `git diff --check` 通过。
新增回归覆盖无候选 decode 的 GDN 标记及 metadata 传递、head 缺失/多余 tensor、形状、精度、非有限值和 TP 不匹配的拒绝，并确认加载失败前不改动任何 draft 权重。
部署边界仍为 eager、同步、`max_num_seqs=1`；未将有界测试描述为长时间稳定性或生产性能验收。

## 主模型缓存复用验证

原生入口现使用 `shared_target_snapshot`，直接读取 target 的分页 KV 和 GDN 接受前缀状态，不再调用 AR forward 重建历史。
现有 draft 执行器接收连续布局，因此分页 KV 仍需 gather，GDN 使用独立工作状态；这不是零拷贝分页 Attention。

本轮重新运行无投机基线，并在 61 的双卡上验证新路径，配置为 TP=2、K=32、eager、同步 batch-one。
固定 head 使用 max_model_len=2048，最后四层在线更新使用 1024；两者均与同后端基线对照。

| 检查 | 结果 |
| --- | --- |
| 固定 head、原生缓存 | 每 rank 55 次缓存读取与起草、50 轮完整 K32 验证、46 个接受候选；greedy 三个用例共 72 个输出 token 与基线完全一致 |
| 最后 4 层在线更新 | 每 rank 52 次读取与起草、47 轮完整 K32 验证、47 次更新、50 个接受候选；两个 rank 计数与 loss 一致，greedy 再次对齐 |
| 主模型缓存只读 | 在线测试每 rank 抽查 15 次，覆盖各请求的首次有效 prefill 起草、投机起草及在线更新；参与本轮的 KV 物理页、卷积行和候选递推状态前后按字节相同 |
| 原生 GDN 接受位置 | 单卡调用真实卷积及 recurrent 算子，连续三轮从上轮接受 1/17/32 个输入的位置继续；本轮全部 32 个接受位置均正确，合计 96 次选择，FP32 状态最大绝对误差 1.2e-7，读取前后原缓存不变 |
| 在线 head 导出与重载 | 两个 TP 分片和 manifest 保存到 `heads-shared-cache-61/version47`，重启服务成功加载，不复制 backbone |
| API 与最后 1 层在线更新 | 加载 version47 后每 rank 完成 52 次更新；8 项接口检查全部通过，包括 SSE、两个客户端排队、断开流后恢复、1804-token 跨页 prefill 及后续请求隔离 |

本轮最后四层测试的最终 TV loss 为 0.4607100487。这里只证明更新链路与冻结约束，不据此宣称收敛或加速。
API 测试使用 max_model_len=2048，最后一层测试的最终日志为 TV loss=0.5171158、梯度范数=0.9754444，两个 rank 一致。
CPU 回归为 154 passed、17 个 CUDA 专用测试 skipped；新增覆盖乱序页、跨页、接受边界、拒绝与全接受、缓存复用后的快照生命周期，以及只读审计能够检出故意写入。
proposer 单元测试明确禁止 AR forward，避免静默回退到历史重放。

一次在线启动在 profiling 阶段得到负 KV 预算，未进入请求执行；清理本次 worker 后重试通过，profiling 波动的根因未单独确认。

本轮记录为 `shared-cache-target-61.json`、`shared-cache-fixed-61.json`、`shared-cache-last4-61.json`、`shared-cache-api-61.json` 和 `shared-cache-gdn-61.log`。
[无投机基线](validation/qwen38/shared-cache-target-61.json)、[固定 head](validation/qwen38/shared-cache-fixed-61.json)、[共享缓存在线更新](validation/qwen38/shared-cache-last4-61.json)、[API](validation/qwen38/shared-cache-api-61.json) 的原始 JSON 随代码提供。
日志已保存在实验机和本地；结束后已停止本次 61、167 的专用容器，实查 61 的 2、3 卡无 NPU 进程，没有停止其他用户任务。安装目录的三个原生源码文件哈希与文末锁定值一致。
以下记录保留为缓存改造前的功能及底层算子证据，不将它们冒充新缓存路径的测试。

## 缓存改造前的验证

实验机 61 的两张 Ascend 910B3，TP=2、K=32、eager、同步调度、max_num_seqs=1。
离线 LLM 接口测试使用 max_model_len=1024，HTTP 服务使用 2048。
基线和 Orthrus 均使用相同 CANN GDN prefill 后端，KV 预算比例为 0.65。

| 检查 | 结果 |
| --- | --- |
| 无投机基线 | 完成 greedy、temperature=1、top-k/top-p、48-token 输出和 484-token 分块 prefill；逐层审计未发现非有限值 |
| Orthrus 更新最后 1 层 | 两 rank 各 47 个完整 K32 验证块、47 次在线更新、48 个接受候选；计数及 loss 一致 |
| greedy 对照 | 短文本、连续拒绝、分块 prefill 三个用例的 72 个输出 token 均与基线相同 |
| head 导出 | 两个 TP 分片及 manifest 成功保存到 version47；不包含第二份 backbone |
| head 重载并更新最后 4 层 | 重启加载 version47 后，两 rank 各完成 40 次更新，覆盖 3 个 GDN 和 1 个 Full Attention；greedy 再次逐 token 对齐 |
| 原生 OpenAI API | health/model、greedy、top-k/top-p、SSE、两个客户端排队、取消后恢复、1804-token 跨页 prefill、长请求后历史重置全部通过 |

两次在线测试的教师、loss、梯度均为有限值，TP 更新计数一致。最后 1 层结束时 TV loss 为 0.4872228503，最后 4 层为 0.3578712344。
这些数值来自不同请求和不同更新序列，不能用来宣称 loss 单调下降或最后 4 层收益更高。
HTTP 验收同样更新最后 4 层，两 rank 各完成 58 次更新，日志中的最终 TV loss 为 0.1591993，梯度范数为 3.5297961。
两个客户端是在 max_num_seqs=1 下排队，不是动态批处理；跨页用例验证可运行及后续历史隔离，不将它描述为长上下文的全量数值验收。

每 rank 的模型加载阶段内存约为 32.11 GiB（last-1）或 32.76 GiB（last-4），包含共享主模型、新增 draft mixer 和训练 master，不是完整运行峰值。
head 由 AR 参数复制初始化；此次没有完成离线 head 训练，也没有证明吞吐提升。

## 算子与采样验证

- GDN recurrent：17/31/32 行分段，以及上轮接受 27 行、本轮只算 8 行的恢复均在 NPU 通过。FP32 状态对 Torch 参考的最大绝对误差约 1.2e-7；输出约 1.82e-4。
- GDN prefill：1/8/32 行均有限；输出最大绝对误差分别为 6.10e-5、2.44e-4、2.44e-4。CANN 状态接口为 BF16，对 FP32 参考的状态最大误差为 0.00503409，含精度转换影响。
- 分页 FIA：覆盖 0/127/129/255 个历史位置、17/32 个 query、乱序物理页及跨页读取；KV 内容前后完全相同。对 FP32 SDPA 采用逐元素 rtol=0.025、atol=0.005，四组通过，整体最大绝对误差 0.00974393。
- 原生 rejection sampler：greedy/random/mixed 模式全部 31 个拒绝位置、全接受 bonus、固定种子批次独立性通过。32768 次抽样的目标概率为 [0.1,0.3,0.6]，观测为 [0.10229492,0.29669189,0.60101318]，通过六倍标准差阈值。

采样模式验证分布，不能要求开启与关闭投机时同种子输出序列相同。
对应脚本为 probe_qwen38_npu_gdn.py、probe_qwen38_npu_fia.py；采样检查直接调用镜像内的原生实现，没有重写采样或 Attention kernel。

## 回归与部署边界

缓存改造前 CPU 回归 130 passed、17 skipped，跳过项为 CUDA 专用测试，不是 NPU 通过记录。
覆盖权重共享、AR/HF 对照、原生参数复制、冻结权重输入反向、TP 前向与梯度、全局裁剪、last-1/last-4、到达行监督、请求切换、K32 元数据和 head 交付。
PIP_NO_INDEX=1 下 editable 安装通过；Ruff 检查通过。

当前不覆盖图模式、动态批处理、量化、PP/DP、超长上下文或生产 goodput。
当前插件复用 native cache 的内容，但仍有临时布局快照，不宣称已完成零拷贝或生产吞吐优化。
head 文件交付不含 AdamW 状态，重启不保证精确续接优化器轨迹。

本镜像原生 Triton prefill 在 target-only 中也出现过首层 GDN NaN，因而显式选择已有 CANN 后端；并非修改该算子的内核。
GDN 和分页 FIA 的 16-query 限制由现有算子的分段调用解决，K 仍为 32。forward 不依赖临时配置上下文，配置在 worker 初始化时绑定。
KV 预算需给 CANN 工作空间留余量，不能将原有 prefill profiling 数字直接当作新后端的完整峰值。

运行入口和限制见 [部署说明](QWEN38_VLLM.md)。
缓存改造前的测试结束时曾停止三台机器的专用容器，当时实查 61 八卡均无 NPU 进程；这不是当前共享机器的占用情况。

## 版本标识

```text
image: quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a
vLLM: 0.26.0+empty
vLLM-Ascend: 0.19.1rc2.dev1398+g85625bd77
Transformers: 5.14.1
PyTorch: 2.10.0+cpu + torch_npu

installed source SHA256:
model_runner_v1.py: c99d5172383c0bc3ef0f8b5722e87f558aa9e3d5f2e523b273bb1ff11e88a2e4
attention_v1.py: 4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076
ops/gdn.py: 4e0663d014503b49ce68e36b95a4dd728e33671983d4d8788db0dbfb9cea64f6

base config.json SHA256:
191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab
base model.safetensors.index.json SHA256:
77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df
```

完整运行日志在实验机 `/data/disk/wangzili/orthrus-qwen38-native-20261030/evidence/`，本地副本在工作区 `debug-20261030/qwen38-native-61/`。
关键结果文件为 target-cann-61.json、orthrus-cann-61.json、orthrus-reload-last4-61.json 和 api-61.json。
head 权重仅保留在实验机的 evidence/heads-native-61/version47，不随代码或本地日志副本发布。
权重索引哈希不等于全部权重内容哈希；正式部署仍需完整模型文件锁定。
