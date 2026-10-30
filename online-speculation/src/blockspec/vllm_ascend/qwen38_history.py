"""Read committed history from the v0.26 native hybrid cache, without AR replay.

The functional draft needs dense KV and HF-shaped GDN states. These are owned
per-proposal snapshots of the target cache, not a second persistent AR cache.
Paged KV is gathered, so this adapter does not claim zero-copy attention.
"""

import torch
from torch.nn import functional as F

from ..parallel.qwen38 import HybridHistory


def paged_history(cache, block_table, length):
    """Native [page, token, KV head, dim] -> owned [1, KV head, token, dim]."""
    if (cache.ndim != 4 or block_table.ndim != 2 or block_table.shape[0] != 1
            or length < 1):
        raise ValueError("batch-one native paged KV and a positive prefix required")
    pages = (length + cache.shape[1] - 1) // cache.shape[1]
    if pages > block_table.shape[1]:
        raise ValueError("committed prefix exceeds native block table")
    indices = block_table[0, :pages].to(device=cache.device, dtype=torch.long)
    return cache.index_select(0, indices).flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0).contiguous()


def gdn_history(cache, metadata, *, accepted, query_length, kernel_width):
    """Select the state after accepted AR inputs, never the rejected suffix.

    Ascend stores SSM as [slot, head, V, K]. Its speculative convolution row
    contains two retained inputs followed by the current inputs (width=4).
    The next accepted prefix's last three inputs start at accepted - 1.
    """
    conv, recurrent = cache
    speculative = metadata.spec_sequence_masks is not None
    if speculative:
        if kernel_width != 4 or not 1 <= accepted <= query_length:
            raise ValueError("native speculative GDN requires width=4 and a valid accepted prefix")
        indices = metadata.spec_state_indices_tensor
        if indices.ndim != 2 or indices.shape[0] != 1 or indices.shape[1] < accepted:
            raise ValueError("missing native speculative GDN state indices")
        state_index = indices[0, accepted - 1:accepted]
        conv_index = metadata.spec_decode_metadata.spec_causal_conv1d.cache_indices[0, :1]
        offset = accepted - 1
    else:
        if accepted != query_length:
            raise ValueError("non-speculative GDN cache cannot roll back a partial query")
        state_index = metadata.non_spec_state_indices_tensor[:1]
        conv_meta = (metadata.non_spec_prefill_metadata.causal_conv1d if metadata.num_prefills
                     else metadata.non_spec_decode_metadata.causal_conv1d)
        conv_index = conv_meta.cache_indices.reshape(1, -1)[0, :1]
        offset = 0
    keep = kernel_width - 1
    if (conv.ndim != 3 or recurrent.ndim != 4 or offset + keep > conv.shape[1]):
        raise ValueError("unsupported native GDN cache layout")
    # 仅复制选中的工作状态；draft 的递推/卷积更新不能写回主模型。
    tail = conv.index_select(0, conv_index.to(conv.device, torch.long))[:, offset:offset + keep]
    conv_state = F.pad(tail.transpose(1, 2), (1, 0))
    recurrent_state = recurrent.index_select(0, state_index.to(recurrent.device, torch.long)).transpose(-1, -2).contiguous()
    return conv_state, recurrent_state


class NativeHybridHistory:
    """Version-pinned cache reader, consumed immediately after sync bookkeeping."""

    backend = "shared_target_snapshot"

    def __init__(self, target, runner, layer_types):
        native = getattr(target, "language_model", target).model
        names = {id(module): name for name, module in runner.compilation_config.static_forward_context.items()}
        self.layers = []
        for layer, kind in zip(native.layers, layer_types, strict=True):
            module = layer.self_attn.attn if kind == "full_attention" else layer.linear_attn
            if id(module) not in names:
                raise ValueError("native cache module is missing from the forward context")
            self.layers.append((kind, module, names[id(module)]))
        if not any(kind == "full_attention" for kind in layer_types):
            raise ValueError("hybrid cache reader requires a full-attention layer")
        self.metadata = self.request_id = None
        self.reads = 0

    def capture(self, metadata, request_id):
        if not isinstance(metadata, dict):
            raise ValueError("eager per-layer attention metadata required")
        self.metadata, self.request_id = metadata, request_id

    def read(self, request_id, length, *, sampled_count, draft_count):
        if self.metadata is None or request_id != self.request_id:
            raise RuntimeError("missing current-request target cache metadata")
        full_name = next(name for kind, _, name in self.layers if kind == "full_attention")
        full_meta = self.metadata[full_name]
        if len(full_meta.seq_lens_list) != 1 or len(full_meta.actual_seq_lengths_q) != 1:
            raise RuntimeError("only unpadded batch-one native cache metadata is supported")
        query_length = full_meta.actual_seq_lengths_q[0]
        accepted = sampled_count if draft_count else query_length
        if (not 1 <= accepted <= query_length or draft_count and query_length != draft_count + 1
                or length != full_meta.seq_lens_list[0] - query_length + accepted):
            raise RuntimeError("committed token ledger does not match the native cache boundary")
        states = []
        for kind, module, name in self.layers:
            meta = self.metadata[name]
            if kind == "full_attention":
                if (meta.seq_lens_list != full_meta.seq_lens_list
                        or meta.actual_seq_lengths_q != full_meta.actual_seq_lengths_q):
                    raise RuntimeError("native full-attention layers disagree on the history boundary")
                key, value = module.kv_cache
                states.append((paged_history(key, meta.block_tables, length),
                               paged_history(value, meta.block_tables, length)))
            else:
                if draft_count and meta.spec_sequence_masks is None:
                    raise RuntimeError("speculative verification has no GDN state checkpoints")
                states.append(gdn_history(module.kv_cache, meta, accepted=accepted,
                                          query_length=query_length, kernel_width=module.conv_kernel_size))
        self.reads += 1
        return HybridHistory(states, length)
