# Orthrus-DSV4.1 PyPTO 在线更新实现边界

## 当前可用部分

双机 PyPTO 已经完成 DSV4.1 W8A8 的 target 推理、K32 draft、整块 verify、接受前缀提交和拒绝回滚；初始化 head 下，双机 16 卡的 greedy 输出与 target-only 逐 token 相同。最后一层的 Torch 参考学习器也已具备：它从同一 anchor 重放 C1A Reuse 层，冻结 W8A8 主模型、mHC、MoE、Norm 和 LM Head，只更新 diffusion Attention，并支持 TV/KL、梯度裁剪和 head 导出。

last-N 不再限定为最后一层。`DSV41SuffixPlan` 可选择连续的最后 1～40 层，首轮扫描固定为 1/2/4/8；`DSV41SuffixReplay` 从所选后缀之前的唯一边界开始，按顺序重新执行每一层，避免更新早期层后继续使用旧的下游隐藏状态。线上只更新所选层的 diffusion Attention，C1A indexer 因 Top-K 离散而保持冻结。

GSM8K 数据固定、延迟教师反馈状态机、两阶段更新频率、学习率 warmup、发布边界和结果汇总已经独立实现并通过单元测试。Qwen3.8 的参数只作为候选起点，DSV4.1 会在独立 tuning 集扫描教师行数、学习率和更新频率。

## PyPTO 反向链路

这里缺的不是普通 Attention 前向，而是最后一层从教师 logits 回到 diffusion Attention 参数的完整反向。模型前向已经能复用现有算子；在线更新需要以下设备计算按顺序连起来：

| 计算 | 输入与输出 | 可复用部分 | DSV4.1 专属工作 |
| --- | --- | --- | --- |
| 完整词表 TV/KL | draft/teacher logits → loss、`dlogits` | Qwen 在线实现的分块 softmax、TV/KL 公式 | 改为 DSV4.1 词表和 TP4 分片，保留真实教师行掩码 |
| LM Head 与末层共享模块 VJP | `dlogits` → 最后一层 Attention 输出梯度 | 长矩阵乘、RMSNorm、SwiGLU 基础梯度 | 冻结 W8A8 LM Head、mHC 和 MoE，只计算输入梯度；Top-K 路由按前向固定 |
| C1A Reuse Attention 反向 | 输出梯度 → `dQ/dKV/dO` | matmul、softmax、RoPE、norm 基础梯度 | 同时覆盖 SWA 历史、C1A 压缩历史、sink/LSE 合并和 K32 临时双向块；历史 cache 只读 |
| 参数梯度与 AdamW | `dQ/dKV/dO` → FP32 master/moments | Qwen 的统计、clip、AdamW、BF16 发布公式 | 按 DSV4.1 TP4 的 Q/KV/O shard 布局生成参数梯度 |
| 双机版本发布 | 新 head → 16 个执行 rank | 已有双机控制链路和驻留权重句柄 | DP4 的同一 verifier 边界切换同一版本；失败时所有 rank 保留旧版本 |

其中模型无关的完整词表 TV/KL、线性梯度、RMSNorm 前反向、SiLU 梯度、AdamW、梯度统计/裁剪和 BF16 发布 PyPTO kernel 已经放在 `pypto_dsv41_online_programs.py` 与 `pypto_dsv41_optimizer_programs.py`。它们不是占位接口：上述路径已在 910B3 上与 Torch 参考对照通过；词表分块额外适配了 DSV4.1 的 129280 维词表。真实 TP4 shard 下的 LM Head、WQ-A、WQ-B、WKV、WO-A 和 WO-B 梯度形状也已完成 PyPTO 编译。

Attention 反向的 Torch golden 在 `dsv41_attention_vjp.py`，显式给出含 sink 和特殊 block mask 的 `dQ/dK/dV/dSink`。对应 PyPTO 实现在 `pypto_dsv41_attention_vjp.py`：它完成 softmax VJP、sink 归并、Q/K/V 矩阵梯度、16 个本地 query head 对共享 latent K/V 的梯度归并，并只将临时 K32 行交给 draft 参数；目标历史 KV 不产生梯度。Torch 公式已与 autograd 对照，生产 TP4/K32 形状已完成 PyPTO 编译。

冻结尾部的反向顺序固定为 LM Head、final norm、最终 mHC collapse、FFN mHC post、所选 MoE、FFN norm、FFN mHC pre、Attention mHC post。`dsv41_tail_vjp.py` 给出精确 Torch 契约，`pypto_dsv41_tail_vjp.py` 已实现最占带宽的 mHC post 数据梯度，并以 K32×5120 生产形状编译。还需在真实 session 中把前向保存的路由与 mHC 小状态接给这些叶子；Top-K 路由本身不重新求导。

双机切换协议已固化在 `dsv41_publication.py`：两个节点先把同一 SHA256、同一版本写入 16 个 rank 的备用槽，确认 rank 不重不漏后再进行可逆切换；双方都确认激活后才 commit。任一步失败都恢复旧 slot 和旧版本，避免两个节点混用不同 head。`ResidentDraftAttention.enable_online(last_layers)` 会为完整后缀建立两个可变驻留槽；激活只发生在 verifier 安全边界。

因此，当前不能把 Torch/CPU 学习器称为“PyPTO 在线训练端到端已完成”。算子、last-N 重放契约和双槽发布已准备；剩余集成点是从真实 verifier 捕获修正前缀的教师 logits 与后缀重放状态，并将 mHC/MoE 的冻结输入反向按上述顺序编排进一次更新。普通权重加载、数据下载和主模型推理不需要重写成训练算子。

## 明日验收门槛

先用 4～16 道 GSM8K 完成 fixed-head smoke，确认真实文本解码、greedy 一致、接受统计和吞吐。在线版本只有同时满足以下条件才算跑通：

1. 教师 logits 来自主模型沿拒绝后修正的真实前缀，不能来自错误候选分支。
2. 只有所选 last-N diffusion Attention 参数变化；主模型、indexer 和共享模块的参数哈希不变。
3. loss、梯度范数和版本号有限，发布前后 16 个 rank 的 head 哈希一致。
4. 开启在线更新后仍与 target-only greedy 输出逐 token 相同。
5. 报告接受长度、更新/发布时间和端到端吞吐，不能只报告 loss 下降。
