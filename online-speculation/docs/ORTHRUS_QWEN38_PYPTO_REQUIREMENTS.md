# Orthrus-Qwen3.8：PyPTO 推理算子需求

更新日期：2026-11-03。对象为 Qwen3.8-27B 文本模型，仅讨论 PyPTO 推理，不包含训练或反向算子。

## 1. 模型与适配目标

Orthrus 在原模型每层增加一套独立的 draft mixer，复用 embedding、层外 norm、MLP 和 LM head。draft 一次输入 `anchor + (K-1) 个 MASK`，生成候选，再由主模型验证。anchor 是已知的最后一个 token；可复用的主模型历史截止到 anchor 之前。

Qwen3.8 的 64 层由两类 mixer 交替组成：

```text
Embedding -> [GDN -> GDN -> GDN -> Full Attention] × 16 -> Norm -> LM head
                每层均含层外 Norm、残差与 Dense MLP
```

**PyPTO 需要适配两条核心计算路径，不是固定新增四个 kernel。** Full Attention 读取历史 KV，并在当前 draft 块内双向计算；GDN 读取主模型卷积和递推状态，在临时状态上继续计算。
当前采用 `orthrus_qwen35_reference`：Full Attention 块内双向，GDN 保持因果递推。这是 Orthrus Qwen3.5 混合模型的扩展策略，原始普通 Qwen3 算法本身没有 GDN；head 必须与该策略匹配，不能静默换成另一种双向 GDN。

当前 K=32；K 是配置，不是算法常数。改变 K 不改变数学定义，但需要相应的编译规格、有效长度处理和测试。

### PyPTO 已有基础

除主线的普通 Qwen3 实现外，已有尚未合入的 [pypto-lib PR #1503](https://github.com/hw-native-sys/pypto-lib/pull/1503)，固定源码为 `1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9`。
该 PR 提供 Qwen3.8 整网前向；作者报告的验证范围为单卡、单请求、最多 128 个位置，未包含投机解码。配套 [pypto-serving PR #281](https://github.com/hw-native-sys/pypto-serving/pull/281) 也仍为 Draft，不能等同于主线已支持 Orthrus。

下文以这个 Qwen3.8 PR 为直接基础，不再把 GLM KDA 当成 Qwen GDN 的替代实现。模块路径均相对于该 PR 的 `models/qwen3_8_27b/`。

| 计算模块 | 现有实现 | Orthrus 所需增量 |
| --- | --- | --- |
| Full Attention | `attention_state.py` 有 Q/K norm、64 维部分 RoPE、GQA 和输出门控；整网的 `full_network_core.py::attention_context_full` 读取连续 KV，只计算一个有效 query，并按其位置限制可见范围。 | 增加 K 行 query 的 draft 入口：每行读取相同历史前缀及整个临时块，不执行主模型 KV append；扩展块长与历史容量。接 vLLM 时另适配分页布局。 |
| GDN | `delta_state.py` 有宽度 4 卷积、head 分组、标量衰减、Delta Rule 和 SiLU 门控 norm；`decode_layer.py::delta_decode_into` 接收旧状态，但一次只执行一个有效 token。 | 保留这些公式和权重语义，提供非零 AR 初始状态下的 K 行连续计算；卷积和递推输出写入 draft 工作区，不覆盖 AR。可先编排已有单步实现，再评估 chunk 融合。 |

Embedding、MLP、残差等不因 Orthrus 改变数学定义，复用现有 Qwen3.8 实现。独立 draft 参数加载、TP 权重布局和跨运行时张量互通仍需模型/引擎适配，不应列作新的 Attention 算法。
该 PR 使用单卡完整规格；当前 vLLM 验证采用 TP2，每个 rank 的 Full Attention 为 12 个 Q head、2 个 KV head，GDN 为 8 个 Q/K head、24 个 V head。接入时须参数化这些尺寸、映射分片权重，并对齐原引擎的 TP 通信，不能把 PR 中的单卡常量直接用于 TP2。

## 2. Full Attention：历史前缀与双向块

设 anchor 位于绝对位置 \(a\)，块内索引 \(i=0,\ldots,K-1\)。\(K_H,V_H\) 表示主模型在 \([0,a)\) 的历史；\(Q_D,K_D,V_D\) 由本层 draft 参数产生。按单个 query head 书写，KV head 根据 GQA 分组选择，\(d=256\)：

$$
\begin{aligned}
o_i^{\mathrm{AR}}
&=\operatorname{softmax}\left(
q_i^{\mathrm{AR}}[K_H;K_{\mathrm{AR},0:i}]^\top/\sqrt d
\right)[V_H;V_{\mathrm{AR},0:i}],\\
O_D
&=\operatorname{softmax}\left(
Q_D[K_H;K_D]^\top/\sqrt d
\right)[V_H;V_D].
\end{aligned}
$$

分号表示序列维拼接，\(0:i\) 包含位置 i。区别在于 AR 每行的可见边界随位置推进，draft 每行都能看整个临时块。历史本身仍由主模型因果计算产生，不重新变成双向。

该模型共有 24 个 Q head、4 个 KV head，GQA 比例为 6；Q/K 先做 norm，仅前 64 维做 RoPE。Attention 输出再乘 query 投影产生的 sigmoid gate，并做 O projection。这些前后处理在 Qwen3.8 PR 中已有基础，不需要作为 Orthrus 新算法重写。
Q/K norm 的缩放参数按 `1 + weight` 解释；对接时仍须对齐 Torch 路径的舍入边界，公式相同不代表不同融合实现自动数值一致。

### 从现有实现到 draft 入口

`attention_context_full` 当前只消费第一个有效 query，并由当前位置生成因果可见范围；`append_kv_full` 将当前 AR 的 K/V 放入连续缓存。Orthrus 不能原样调用整条 AR step，因为临时 MASK 的 K/V 不是主模型历史。

需要把 Attention 的输入分成历史与临时块，计算 K 行输出；历史截止位置统一为 a，块内不加因果限制。Q/K norm、RoPE、GQA、门控和投影继续复用。历史和块内可以在同一 kernel 中遍历，也可分段计算后按 LSE 精确合并，不要求单独开发一个合并 kernel。

### 现有 Torch/NPU 组合

下面是现有实现中 Attention 核心的独立等价函数。输入 Q/K 已完成 norm 与 RoPE，张量布局为 `[batch, head, token, dim]`；`blocked` 为可选的禁止访问 mask，当前单请求、无 padding 的 draft 块全可见。

```python
import torch
import torch_npu

def draft_full_attention(q, kd, vd, kh, vh, blocked=None):
    # Orthrus：目标历史只作输入，MASK 块的 KV 只参与本次计算。
    k = torch.cat((kh, kd), dim=2).contiguous()
    v = torch.cat((vh, vd), dim=2).contiguous()
    return torch_npu.npu_fusion_attention(
        q, k, v, q.shape[1], input_layout="BNSD",
        atten_mask=blocked, scale=q.shape[-1] ** -0.5,
        keep_prob=1.0, sparse_mode=0,
    )[0]
```

该融合算子直接接收较少的 KV head，不需要把整段历史复制成 24 个 head。完整前后处理在 `src/blockspec/vllm_ascend/qwen38_ops.py::full_attention`。
主模型分页 KV 由 `qwen38_history.py` 转成独立快照；当前不是零拷贝分页直读。PyPTO 首版可以接同样的连续输入，后续再用 block table 直接读取，避免把“先实现正确前向”和“必须一次完成分页融合”绑在一起。

## 3. GDN：带历史初值的块递推

GDN 不计算 softmax Attention，没有一张 mask 可以把它改成所需状态计算。其输入先经过宽度为 4 的因果逐通道卷积，再执行门控 Delta Rule。Q/K 各有 16 个 head，V 有 48 个 head，维度均为 128；每组 Q/K 对应三个 V head。

省略 batch/head 下标，令 q、k 为 L2 归一化后的列向量，v 为 value，状态取原生布局 \(S\in\mathbb R^{d_v\times d_k}\)。单步公式为：

$$
\begin{aligned}
\beta_i&=\sigma(b_i),&
g_i&=-\exp(A_{\log})\operatorname{softplus}(u_i+\Delta t),\\
\overline S_i&=\exp(g_i)S_{i-1},&
\delta_i&=\beta_i(v_i-\overline S_i k_i),\\
S_i&=\overline S_i+\delta_i k_i^\top,&
o_i&=S_iq_i/\sqrt{d_k}.
\end{aligned}
$$

其中 sigmoid 记为 \(\sigma\)，b、u 来自对应 mixer 的投影；\(A_{\log}\)、\(\Delta t\) 是每个 V head 的参数，g 是标量衰减。AR 和 draft 使用相同公式，但参数和状态归属不同：

| 路径 | 初始状态 | 本块参数与输出状态 |
| --- | --- | --- |
| AR | 上一轮已接受的 AR 状态 | 用主模型参数计算，验证后由引擎保留接受前缀对应状态。 |
| Draft | anchor 之前的 AR 状态及三个卷积历史输入 | 用 draft 参数计算，在独立工作区推进；结果不提交给 AR。 |

递推状态按 FP32 累积；输出再经过带独立 weight 的 RMSNorm、`SiLU(z)` 门控和输出投影。这里的 norm 不能套用 Full Attention 的 `1 + weight` 解释。
公式采用原生 cache 的 `[V, K]` 状态布局，HF 及该 PyPTO PR 的入口采用 `[K, V]`，适配时需要转置；两维恰好都为 128，不能只按 shape 判断兼容。卷积历史保留的是卷积前的三个 QKV 投影输入，不是卷积输出。

### 从现有实现到 draft 入口

Qwen3.8 PR 已有这里的 GDN，而非仅有公式相近的 GLM KDA。`delta_decode_into` 接收 `previous_conv` 和 `previous_recurrent`，把一个新输入接入卷积历史，再调用 `delta_step`；因此它可作为带初值块计算的基础。

需要将这个入口推广到 K 个位置：卷积按顺序使用历史尾部及块输入，递推将每步状态传给下一步，并输出全部位置。PR 中的短 prefill 卷积带有起始位置和零历史假设，不能代替这个非零历史入口。其 16 行 padded tile 也不代表已经支持 K32。
首版可以循环编排已有单步组件，之后按实测选择 chunk 实现；不需要改变 GDN 公式，也不能把同一请求的 K 个位置当成互不相关的 batch 行。

### 现有 Torch/NPU 组合

下面展示状态交接和递推核心。Q/K 已归一化；为对接现有适配函数，块内 Q/K 按 V head 分组展开。只展开短块，不复制长历史。

```python
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from blockspec.vllm_ascend.qwen38_ops import delta_rule

def draft_gdn(q, k, v, log_decay, beta, ar_state):
    # 输入布局 [batch, token, head, dim]；ar_state 使用 HF 的 [K, V] 布局。
    groups = v.shape[2] // k.shape[2]
    q = q.repeat_interleave(groups, dim=2)
    k = k.repeat_interleave(groups, dim=2)
    output, _ = delta_rule(
        q, k, v, g=log_decay, beta=beta,
        initial_state=ar_state, output_final_state=False,
        use_qk_l2norm_in_kernel=False, groups=groups,
        fallback=torch_chunk_gated_delta_rule,
    )
    return output  # draft 最终状态不转为主模型历史。
```

完整路径还包含投影、宽度 4 卷积、Q/K norm、门控 norm 和输出投影。NPU 推理复用镜像已有的融合卷积、L2 norm、递推及门控 norm；K32 递推在当前 A2 接口限制下分成最多 16 行的片段，并传递同一临时状态。这不是 K 次完整模型推理。
适配层将 `ar_state` 复制到独立工作区，分段前只打包一次输入，并保留逐位置 checkpoint；调用方不需要重复复制历史初态。上述执行调整不增加 PyPTO 的算子种类或改变状态契约。
完整模型执行见 `src/blockspec/parallel/qwen38.py`、`src/blockspec/vllm_ascend/qwen38_ops.py` 和 `qwen38_gdn_backend.py`。可选 FLA 反向仅属于训练实现，不列入本次 PyPTO 推理需求。

## 4. 完整层与推理引擎

一层的 draft 计算仍是 `共享 input norm -> draft mixer -> 残差 -> 共享 norm/MLP -> 残差`。共享的是权重，不是本轮计算结果；执行器应绑定主模型权重并加载独立 draft 参数，不复制第二套 backbone。

仅将 draft 放到 PyPTO 时，vLLM-Ascend 仍负责主模型前向、验证采样及正式 cache 的提交/回收。接入方传入只读历史和独立工作区，并约定张量布局、设备地址、有效长度和 stream 同步；不能仅凭两边都在 NPU 就假设可以直接共享张量。
第一步也可以仅替换两个 mixer 的计算入口，继续沿用现有 Torch 执行器的共享 MLP、权重映射和 TP 通信；整段 draft 前向迁入 PyPTO 时，再接齐这些模块的调用，不必同时重写其算子。

若主模型也放到 PyPTO，则额外需要块验证与接受前缀恢复。验证了 K 行 AR 输入后只保留前 A 行时，Full Attention 限制后续可见 KV 长度；GDN 必须取得第 A 行结束的递推状态和卷积尾部，不能把第 K 行状态“截短”。可以保存 checkpoint 或重算接受前缀，选择方式属于引擎与算子输出接口设计。
无论采用哪种方式，**正式历史来自 AR 验证，不来自 draft 状态转正**。

### 整体 Torch/NPU 运行入口

上面的两个函数说明算子接口；完整模型、共享权重、历史读取及 propose/verify 循环已经在本仓库实现，可在固定 vLLM-Ascend 环境中运行：

```bash
cd online-speculation
python scripts/run_in_runtime.py scripts/validate_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

模型路径按实际挂载修改。该入口使用真实 27B、镜像已有融合算子和原生 verifier，不启用在线更新；未提供 head 时从主模型 mixer 复制初始化，只验证执行链路。使用训练 head 时加 `--head-dir`，mask ID 须与训练配置一致。

## 5. 实现与验收顺序

1. 从固定版本的 Qwen3.8 PR 复用前后处理，分别接通连续历史的双向 Full Attention 和非零历史初值的块 GDN。
2. 用现有 Torch/NPU 实现逐模块对照，覆盖 K=2/8/17/32、非零状态、短尾块、只读历史；Full Attention 另测乱序分页与跨页边界。
3. 接整层与整网 draft，核对 head 权重布局、位置编码和主模型共享参数；再接 vLLM 验证和任意接受位置的状态恢复。
4. 对实际瓶颈做分页直读、缓冲复用及融合；测 draft 前向和搬运各自耗时，不把一个小算子提速等同于整网加速。

现有 Torch/NPU + vLLM-Ascend 已在真实 27B、TP2 验证起草、greedy/采样、拒绝恢复、缓存隔离及 head 导出重载；在线 last-N 与离线全 draft 更新也有独立验证记录。这是 PyPTO 的可运行参照，不代表 PyPTO Orthrus 已完成验收。
本轮不要求为训练实现反向，也不涉及 DSV4 的压缩 KV、indexer 或 SWA/HCA/CSA。

## 参考实现

- [Qwen3.8 PyPTO 整网入口](https://github.com/hw-native-sys/pypto-lib/blob/1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9/models/qwen3_8_27b/full_network_core.py)、[GDN 状态计算](https://github.com/hw-native-sys/pypto-lib/blob/1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9/models/qwen3_8_27b/delta_state.py)、[带历史的 GDN decode](https://github.com/hw-native-sys/pypto-lib/blob/1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9/models/qwen3_8_27b/decode_layer.py)。
- [Orthrus Qwen3.5 混合模型参考](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py)。
- [运行入口与配置](https://github.com/wangzili121/online-speculation/blob/feature/orthrus-qwen38/online-speculation/docs/QWEN38_VLLM.md)、[推理验证记录](https://github.com/wangzili121/online-speculation/blob/feature/orthrus-qwen38/online-speculation/docs/QWEN38_VLLM_VALIDATION.md)、[真实语料与更新验证](https://github.com/wangzili121/online-speculation/blob/feature/orthrus-qwen38/online-speculation/docs/QWEN38_TRAINING_OPS_AUDIT.md)。
