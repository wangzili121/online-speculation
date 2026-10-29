"""Preserve K32 while honoring the A2 paged FIA query limit of 16."""

import copy
import functools
import inspect


def _enabled(config):
    spec = config.speculative_config
    return (spec is not None and spec.method == "custom_class"
            and spec.model == "blockspec.vllm_ascend.qwen38_proposer.Qwen38OrthrusProposer")


def verification_chunks(metadata, limit=16):
    """Each query slice retains exactly the same causal key visibility."""
    if len(metadata.actual_seq_lengths_q) != 1 or not metadata.causal:
        raise ValueError("causal batch-one verification required")
    total = metadata.actual_seq_lengths_q[0]
    past = metadata.seq_lens_list[0] - total
    if past < 0:
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
        yield slice(start, end), part


def install_attention_hooks(builder_class=None, impl_class=None, spec_state=None):
    if builder_class is None:
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackendImpl, AscendAttentionMetadataBuilder, AscendAttentionState,
        )
        builder_class, impl_class = AscendAttentionMetadataBuilder, AscendAttentionBackendImpl
        spec_state = AscendAttentionState.SpecDecoding
    if getattr(builder_class, "_qwen38_query_limit_hook", False):
        return
    initialize = builder_class.__init__
    forward = impl_class.forward_fused_infer_attention
    if not {"vllm_config", "kv_cache_spec", "layer_names", "device"} <= inspect.signature(initialize).parameters.keys():
        raise RuntimeError("unsupported Ascend attention builder signature")
    if not {"attn_metadata", "kv_cache", "output"} <= inspect.signature(forward).parameters.keys():
        raise RuntimeError("unsupported Ascend FIA signature")

    @functools.wraps(initialize)
    def bounded(self, kv_cache_spec, layer_names, vllm_config, device):
        if not _enabled(vllm_config) or vllm_config.speculative_config.num_speculative_tokens < 16:
            return initialize(self, kv_cache_spec, layer_names, vllm_config, device)
        bounded_config = copy.copy(vllm_config)
        bounded_spec = copy.copy(vllm_config.speculative_config)
        bounded_spec.num_speculative_tokens = 15
        bounded_config.speculative_config = bounded_spec
        initialize(self, kv_cache_spec, layer_names, bounded_config, device)
        self.vllm_config, self.speculative_config = vllm_config, vllm_config.speculative_config

    @functools.wraps(forward)
    def sliced(self, query, key, value, attn_metadata, output, kv_cache=None):
        if (not _enabled(self.vllm_config) or attn_metadata.attn_state != spec_state
                or attn_metadata.actual_seq_lengths_q[-1] <= 16):
            return forward(self, query, key, value, attn_metadata, output, kv_cache)
        # 外层已一次写入整块 AR KV；这里只切 query，第二片仍能读取第一片。
        # 每片的 KV 末端随 query 末端推进，因此与整块因果 Attention 等价。
        for rows, metadata in verification_chunks(attn_metadata):
            forward(self, query[rows], None if key is None else key[rows],
                    None if value is None else value[rows], metadata, output[rows], kv_cache)
        return output

    builder_class.__init__ = bounded
    impl_class.forward_fused_infer_attention = sliced
    builder_class._qwen38_query_limit_hook = True
