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
from unittest.mock import patch

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
    speculator.max_num_tokens = num_tokens
    speculator.device = torch.device("cpu")
    speculator.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))

    def make_group(name):
        builder = object.__new__(module.AscendDSAV41MetadataBuilder)
        builder._build_query_metadata = True
        builder._build_compressor_metadata = True
        return SimpleNamespace(layer_names=[name], get_metadata_builder=lambda _: builder)

    draft_groups = [[make_group("swa")], [make_group("long")]]
    target_groups = [[make_group("swa")], [make_group("long")]]
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[name]) for name in ("swa", "long")],
    )

    def init_draft_groups(*args):
        speculator.attn_groups = draft_groups

    with (
        patch.object(module.DraftModelSpeculator, "set_attn", init_draft_groups),
        patch.object(module, "prepare_dflash_inputs_factory"),
        patch.object(module.dflash_speculator, "prepare_dflash_inputs"),
    ):
        speculator.set_attn(None, speculator.kv_cache_config, speculator.block_tables, None, target_groups)
    for groups, enabled in ((draft_groups, False), (target_groups, True)):
        for (group,) in groups:
            builder = group.get_metadata_builder(0)
            assert builder._build_query_metadata is enabled
            assert builder._build_compressor_metadata is enabled

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
    assert captured["causal"] is True  # Obtain target cache coordinate views only.
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
                "target_coordinate_metadata_causal": captured["causal"],
                "draft_builds_ar_operator_metadata": False,
                "target_builds_ar_operator_metadata": True,
                "skip_ring_state_update": captured["skip_ring_state_update"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
