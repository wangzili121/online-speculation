#!/usr/bin/env python3
"""Check the Orthrus V4.1 MRV2 history boundary and write barriers.

This smoke runs in the pinned vLLM-Ascend image without model weights.  It
constructs the state immediately after DFlash-style parallel input preparation
and intercepts the metadata builder, so an upstream ABI change cannot silently
make draft rows visible as committed target history or writable cache slots.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from blockspec.vllm_ascend import dsv41_speculator as module


def main() -> None:
    speculator = object.__new__(module.AscendOrthrusV41Speculator)
    committed = torch.tensor([42, 9], dtype=torch.int32)
    block = 32
    num_tokens = 2 * block
    speculator.input_buffers = SimpleNamespace(
        seq_lens=committed + block,
        query_start_loc=torch.tensor([0, block, 2 * block], dtype=torch.int32),
        positions=torch.cat((torch.arange(42, 74), torch.arange(9, 41))).to(torch.int64),
    )
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.arange(2 * num_tokens, dtype=torch.int32).view(2, num_tokens),
        input_block_tables=[
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32),
        ],
    )
    speculator.attn_groups = [[object()], [object()]]
    speculator.kv_cache_config = SimpleNamespace()
    speculator.draft_attn_layer_names = {"swa", "long"}

    captured: dict[str, object] = {}
    original = module.build_attn_metadata

    def capture(**kwargs):
        captured.update(kwargs)
        return {"captured": kwargs}

    module.build_attn_metadata = capture
    try:
        result = speculator._build_attn_metadata(
            num_reqs=2,
            batch_desc=SimpleNamespace(num_reqs=2),
            query_start_loc_np=np.array([0, block, 2 * block], dtype=np.int32),
            seq_lens_cpu_upper_bound=committed.clone(),
            step=block,
        )
    finally:
        module.build_attn_metadata = original

    assert "captured" in result
    assert torch.equal(captured["seq_lens"], committed)
    assert torch.equal(captured["seq_lens_cpu_upper_bound"], committed)
    assert captured["max_seq_len"] == 42
    assert captured["causal"] is False
    assert captured["skip_ring_state_update"] is True
    assert captured["slot_mappings"].eq(PAD_SLOT_ID).all()
    assert [table.tolist() for table in captured["block_tables"]] == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
    print(
        json.dumps(
            {
                "draft_seq_lens_after_input_prep": (committed + block).tolist(),
                "target_history_seq_lens": captured["seq_lens"].tolist(),
                "query_start_loc": captured["query_start_loc_cpu"].tolist(),
                "cache_slots_all_pad": True,
                "causal": captured["causal"],
                "skip_ring_state_update": captured["skip_ring_state_update"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
