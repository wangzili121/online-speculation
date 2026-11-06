# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：主模型 verifier 的分页 Attention 适配：K32 保持不变，内部按 query 上限切片并保持因果可见边界。不是 draft 双向 Attention 的实现文件。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Preserve K32 while honoring the A2 paged FIA query limit of 16."""

# 依赖：导入 copy。
import copy
# 依赖：导入 functools。
import functools
# 依赖：导入 inspect。
import inspect


# ── _enabled ──
# 职责：限定 hook 仅作用于本 Qwen proposer 配置，避免改变其他模型执行路径。
def _enabled(config):
    spec = config.speculative_config
    return (spec is not None and spec.method == "custom_class"
            and spec.model == "blockspec.vllm_ascend.qwen38_proposer.Qwen38OrthrusProposer")


# ── verification_chunks ──
# 职责：把原生因果 verifier query 分为最多 16 行的片段；每片 key 末端随 query 末端推进，保持因果语义。
# 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
def verification_chunks(metadata, limit=16):
    """Each query slice retains exactly the same causal key visibility."""
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：causal batch-one verification required。
    if len(metadata.actual_seq_lengths_q) != 1 or not metadata.causal:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("causal batch-one verification required")
    total = metadata.actual_seq_lengths_q[0]
    # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
    past = metadata.seq_lens_list[0] - total
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：negative cached history length。
    if past < 0:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("negative cached history length")
    for start in range(0, total, limit):
        end = min(start + limit, total)
        part = copy.copy(metadata)
        part.actual_seq_lengths_q = [end - start]
        part.seq_lens_list = [past + end]
        part.seq_lens = metadata.seq_lens.new_tensor(part.seq_lens_list)
        part.seq_lens_cpu = metadata.seq_lens_cpu.new_tensor(part.seq_lens_list)
        part.query_start_loc = metadata.query_start_loc.new_tensor([0, end - start])
        part.num_actual_tokens = part.num_decode_tokens = end - start
        part.max_query_len = end - start
        part.num_decodes, part.num_prefills = 1, 0
        # 逐项产出：调用者迭代时才取出这条记录/结果，不一次性装入整个数据集。
        yield slice(start, end), part


# ── install_attention_hooks ──
# 职责：为指定版本 Ascend Attention 安装 builder 和调用切片适配；不把 verifier 改成双向。
def install_attention_hooks(builder_class=None, impl_class=None, spec_state=None):
    if builder_class is None:
        # 依赖：从 vllm_ascend.attention.attention_v1 导入 AscendAttentionBackendImpl, AscendAttentionMetadataBuilder, AscendAttentionState；供本模块调用。
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackendImpl, AscendAttentionMetadataBuilder, AscendAttentionState,
        )
        builder_class, impl_class = AscendAttentionMetadataBuilder, AscendAttentionBackendImpl
        spec_state = AscendAttentionState.SpecDecoding
    if getattr(builder_class, "_qwen38_query_limit_hook", False):
        # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
        return
    initialize = builder_class.__init__
    forward = impl_class.forward_fused_infer_attention
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unsupported Ascend attention builder signature。
    if not {"vllm_config", "kv_cache_spec", "layer_names", "device"} <= inspect.signature(initialize).parameters.keys():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise RuntimeError("unsupported Ascend attention builder signature")
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unsupported Ascend FIA signature。
    if not {"attn_metadata", "kv_cache", "output"} <= inspect.signature(forward).parameters.keys():
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise RuntimeError("unsupported Ascend FIA signature")

    @functools.wraps(initialize)
    # ── install_attention_hooks.bounded ──
    # 职责：用受限候选宽度建立兼容 builder，再恢复真实 K32 配置与 decode 阈值。
    def bounded(self, kv_cache_spec, layer_names, vllm_config, device):
        if not _enabled(vllm_config) or vllm_config.speculative_config.num_speculative_tokens < 16:
            return initialize(self, kv_cache_spec, layer_names, vllm_config, device)
        bounded_config = copy.copy(vllm_config)
        bounded_spec = copy.copy(vllm_config.speculative_config)
        bounded_spec.num_speculative_tokens = 15
        bounded_config.speculative_config = bounded_spec
        initialize(self, kv_cache_spec, layer_names, bounded_config, device)
        self.vllm_config, self.speculative_config = vllm_config, vllm_config.speculative_config
        self.decode_threshold = self.reorder_batch_threshold = self.speculative_config.num_speculative_tokens + 1

    @functools.wraps(forward)
    # ── install_attention_hooks.sliced ──
    # 职责：对超过设备 query 限制的 verifier 逐片调用已有 FIA；整块 KV 已在外层写入，不重复提交 draft KV。
    # 输入提示：query：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；key：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def sliced(self, query, key, value, attn_metadata, output, kv_cache=None):
        if (not _enabled(self.vllm_config) or attn_metadata.attn_state != spec_state
                or attn_metadata.actual_seq_lengths_q[-1] <= 16):
            return forward(self, query, key, value, attn_metadata, output, kv_cache)
        # 外层已一次写入整块 AR KV；这里只切 query，第二片仍能读取第一片。
        # 每片的 KV 末端随 query 末端推进，因此与整块因果 Attention 等价。
        for rows, metadata in verification_chunks(attn_metadata):
            forward(self, query[rows], None if key is None else key[rows],
                    None if value is None else value[rows], metadata, output[rows], kv_cache)
        # 输出：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
        return output

    builder_class.__init__ = bounded
    impl_class.forward_fused_infer_attention = sliced
    builder_class._qwen38_query_limit_hook = True
