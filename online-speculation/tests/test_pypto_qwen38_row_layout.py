"""截掉不可见的尾行，不改变有效块的 Attention 或对外 K32 契约。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.attention import AttentionShape, attention_golden
from blockspec.pypto_qwen38.tp2_row_layout import row_geometry


@pytest.mark.parametrize("batch,rows,expected", [(1, 32, (32, 32)), (2, 32, (64, 64)), (2, 16, (64, 32))])
def test_storage_contract_and_computation_are_distinct(batch, rows, expected):
    assert row_geometry(batch, rows) == expected


@pytest.mark.parametrize("batch,rows", [(0, 32), (3, 32), (True, 32), (2, True), (2, 8), (1, 16)])
def test_unsupported_layouts_fail_without_importing_pypto(batch, rows):
    with pytest.raises(ValueError):
        row_geometry(batch, rows)


@pytest.mark.parametrize("flag,width,batch", [(True, 32, 2), (True, 8, 1), (1, 8, 2)])
def test_runner_rejects_incompatible_layout_before_loading_any_weights(flag, width, batch):
    from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner
    config = SimpleNamespace(speculative_tokens=0, prefix_cache=False, max_model_len=512,
        long_prefill_token_threshold=4, max_num_batched_tokens=128, max_num_seqs=2)
    with pytest.raises(ValueError, match="compact paired rows"):
        OrthrusTP2Runner(config, "nonexistent-head", draft_width=width, output_candidates=width-1,
                         draft_batch_size=batch, compact_paired_rows=flag)


@pytest.mark.parametrize("past,width", [(0, 8), (127, 8), (496, 16)])
def test_active_attention_depends_on_logical_width_not_padded_query_count(past, width):
    torch.manual_seed(231 + past)
    original = AttentionShape(q_heads=4, kv_heads=2, head_dim=64, page_size=128, query_tile=32)
    compact = replace(original, block_size=16, query_tile=16)
    pools = [torch.randn((2, 8, 128, 64)).mul_(.1).bfloat16() for _ in range(2)]
    table = torch.tensor([5, 1, 7, 2], dtype=torch.int32)
    for _ in range(2):  # 两个请求各自的临时块，均不能看见自己的无效尾行。
        q, k, v, gate = [torch.randn((32, n)).mul_(.1).bfloat16() for n in (256, 128, 128, 256)]
        reference = attention_golden(original, q, k, v, gate, *pools, table, past, width,
                                     native_rounding=True)[:width]
        actual = attention_golden(compact, q[:16], k[:16], v[:16], gate[:16],
                                  *pools, table, past, width, native_rounding=True)[:width]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
