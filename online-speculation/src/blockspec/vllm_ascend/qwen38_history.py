# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：历史读取层：从原生分页 KV / GDN checkpoint 取得已提交前缀的快照。共享历史来源，不代表所有数据零拷贝。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Read committed history from the v0.26 native hybrid cache, without AR replay.

The functional draft needs dense KV and HF-shaped GDN states. These are owned
per-proposal snapshots of the target cache, not a second persistent AR cache.
Paged KV is gathered, so this adapter does not claim zero-copy attention.
"""

# 依赖：导入 torch。
import torch
# 依赖：从 torch.nn 导入 functional；供本模块调用。
from torch.nn import functional as F

# 依赖：从 ..parallel.qwen38 导入 HybridHistory；供本模块调用。
from ..parallel.qwen38 import HybridHistory


# ── paged_history ──
# 职责：把原生 [page,token,KV_head,dim] 按 block_table 收集为 [1,KV_head,history,dim]；发生 gather/重排拷贝。
# 输入：原生 cache [page,page_size,kv_heads,dim]，当前请求 block_table [1,num_pages]，严格历史长度 length。
# 输出：[1,kv_heads,length,dim] 的连续张量；index_select/contiguous 可能复制，不承诺零拷贝。
def paged_history(cache, block_table, length):
    """Native [page, token, KV head, dim] -> owned [1, KV head, token, dim]."""
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：batch-one native paged KV and a positive prefix required。
    if (cache.ndim != 4 or block_table.ndim != 2 or block_table.shape[0] != 1
            or length < 1):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("batch-one native paged KV and a positive prefix required")
    pages = (length + cache.shape[1] - 1) // cache.shape[1]
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：committed prefix exceeds native block table。
    if pages > block_table.shape[1]:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("committed prefix exceeds native block table")
    # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
    indices = block_table[0, :pages].to(device=cache.device, dtype=torch.long)
    return cache.index_select(0, indices).flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0).contiguous()


# ── gdn_history ──
# 职责：从原生 speculative checkpoint 选取接受前缀对应的卷积/递推状态，避免把拒绝后缀当历史。
# 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定；accepted：本轮已提交边界对应的位置数；具体包含哪些输入由调用处的 native metadata 定义；query_length：本轮原生执行的输入行数；与已接受候选数不是同一个量。
# 输出：(conv_state,recurrent_state)，是 draft 工作初态，不是候选 token 的最终输出。
# 注意：accepted 对应原生已接受输入的状态边界；原生 sampler 的输出数量还包含校正/bonus 的语义。
def gdn_history(cache, metadata, *, accepted, query_length, kernel_width):
    """Select the state after accepted AR inputs, never the rejected suffix.

    Ascend stores SSM as [slot, head, V, K]. Its speculative convolution row
    contains two retained inputs followed by the current inputs (width=4).
    The next accepted prefix's last three inputs start at accepted - 1.
    """
    conv, recurrent = cache
    speculative = metadata.spec_sequence_masks is not None
    if speculative:
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native speculative GDN requires width=4 and a valid accepted prefix。
        if kernel_width != 4 or not 1 <= accepted <= query_length:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("native speculative GDN requires width=4 and a valid accepted prefix")
        indices = metadata.spec_state_indices_tensor
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：missing native speculative GDN state indices。
        if indices.ndim != 2 or indices.shape[0] != 1 or indices.shape[1] < accepted:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("missing native speculative GDN state indices")
        state_index = indices[0, accepted - 1:accepted]
        conv_index = metadata.spec_decode_metadata.spec_causal_conv1d.cache_indices[0, :1]
        # 数据：从原生卷积工作缓存选择接受前缀尾部的起点。
        offset = accepted - 1
    else:
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：non-speculative GDN cache cannot roll back a partial query。
        if accepted != query_length:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("non-speculative GDN cache cannot roll back a partial query")
        state_index = metadata.non_spec_state_indices_tensor[:1]
        conv_meta = (metadata.non_spec_prefill_metadata.causal_conv1d if metadata.num_prefills
                     else metadata.non_spec_decode_metadata.causal_conv1d)
        # 计算：重排维度表达，元素不变；可能在必要时产生拷贝。
        conv_index = conv_meta.cache_indices.reshape(1, -1)[0, :1]
        # 数据：从原生卷积工作缓存选择接受前缀尾部的起点。
        offset = 0
    # 数据：卷积后续递推必须保留的历史输入数，通常为 kernel_width−1。
    keep = kernel_width - 1
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unsupported native GDN cache layout。
    if (conv.ndim != 3 or recurrent.ndim != 4 or offset + keep > conv.shape[1]):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("unsupported native GDN cache layout")
    # 仅复制选中的工作状态；draft 的递推/卷积更新不能写回主模型。
    # 计算：按索引收集行/页/状态，返回新张量；对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
    tail = conv.index_select(0, conv_index.to(conv.device, torch.long))[:, offset:offset + keep]
    # 数据：转换后的 GDN 卷积初态；不能与主模型可写状态混用。 此处交换张量维度以适配 head/token 或 K/V 布局。
    conv_state = F.pad(tail.transpose(1, 2), (1, 0))
    # 数据：从原生布局转换后的 GDN 递推矩阵，供临时 draft 初态使用。 此处使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    recurrent_state = recurrent.index_select(0, state_index.to(recurrent.device, torch.long)).transpose(-1, -2).contiguous()
    # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
    return conv_state, recurrent_state


# ── NativeHybridHistory ──
# 职责：类型 `NativeHybridHistory`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class NativeHybridHistory:
    """Version-pinned cache reader, consumed immediately after sync bookkeeping."""

    backend = "shared_target_snapshot"

    # ── NativeHybridHistory.__init__ ──
    # 职责：把每层原生缓存模块与 static_forward_context 名称对应起来；之后用名称取得本轮 metadata。
    def __init__(self, target, runner, layer_types):
        # 数据：原生模型主体，后续共享参数/缓存来自这里。
        native = getattr(target, "language_model", target).model
        names = {id(module): name for name, module in runner.compilation_config.static_forward_context.items()}
        self.layers = []
        for layer, kind in zip(native.layers, layer_types, strict=True):
            module = layer.self_attn.attn if kind == "full_attention" else layer.linear_attn
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native cache module is missing from the forward context。
            if id(module) not in names:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("native cache module is missing from the forward context")
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            self.layers.append((kind, module, names[id(module)]))
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：hybrid cache reader requires a full-attention layer。
        if not any(kind == "full_attention" for kind in layer_types):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("hybrid cache reader requires a full-attention layer")
        # 数据：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定；当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
        self.metadata = self.request_id = None
        self.reads = 0

    # ── NativeHybridHistory.capture ──
    # 职责：保存当前请求的逐层 metadata；这里只记录索引信息，真正张量读取发生在 read。
    # 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定；request_id：当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
    def capture(self, metadata, request_id):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：eager per-layer attention metadata required。
        if not isinstance(metadata, dict):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("eager per-layer attention metadata required")
        # 数据：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定；当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
        self.metadata, self.request_id = metadata, request_id

    # ── NativeHybridHistory.read ──
    # 职责：校对请求和已提交长度，逐层读取 Full Attention KV 或 GDN 状态，返回 HybridHistory。
    # 输入提示：request_id：当前请求身份，用于拒绝使用其他请求遗留的 pending/history；sampled_count：原生 sampler 本轮输出的有效 token 个数，包含校正/bonus token 的情况；draft_count：原生 metadata 记录的本轮候选个数，用于分辨普通 decode 和投机验证。
    # 调用者：proposer.propose_batch，在原生同步 token 记账之后。
    # 输出：HybridHistory(states,length)，每层一对状态；整组历史由该次起草持有，不能跨请求复用。
    def read(self, request_id, length, *, sampled_count, draft_count):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：missing current-request target cache metadata。
        if self.metadata is None or request_id != self.request_id:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("missing current-request target cache metadata")
        full_name = next(name for kind, _, name in self.layers if kind == "full_attention")
        full_meta = self.metadata[full_name]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：only unpadded batch-one native cache metadata is supported。
        if len(full_meta.seq_lens_list) != 1 or len(full_meta.actual_seq_lengths_q) != 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("only unpadded batch-one native cache metadata is supported")
        # 数据：本轮原生执行的输入行数；与已接受候选数不是同一个量。
        query_length = full_meta.actual_seq_lengths_q[0]
        # 数据：本轮已提交边界对应的位置数；具体包含哪些输入由调用处的 native metadata 定义。
        accepted = sampled_count if draft_count else query_length
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：committed token ledger does not match the native cache boundary。
        if (not 1 <= accepted <= query_length or draft_count and query_length != draft_count + 1
                or length != full_meta.seq_lens_list[0] - query_length + accepted):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("committed token ledger does not match the native cache boundary")
        # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
        states = []
        for kind, module, name in self.layers:
            meta = self.metadata[name]
            if kind == "full_attention":
                # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native full-attention layers disagree on the history boundary。
                if (meta.seq_lens_list != full_meta.seq_lens_list
                        or meta.actual_seq_lengths_q != full_meta.actual_seq_lengths_q):
                    # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                    raise RuntimeError("native full-attention layers disagree on the history boundary")
                # 数据：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
                key, value = module.kv_cache
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                states.append((paged_history(key, meta.block_tables, length),
                               paged_history(value, meta.block_tables, length)))
            else:
                # 约束检查：下方条件表示不支持的输入/状态；触发时报错：speculative verification has no GDN state checkpoints。
                if draft_count and meta.spec_sequence_masks is None:
                    # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                    raise RuntimeError("speculative verification has no GDN state checkpoints")
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                states.append(gdn_history(module.kv_cache, meta, accepted=accepted,
                                          query_length=query_length, kernel_width=module.conv_kernel_size))
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.reads += 1
        return HybridHistory(states, length)
