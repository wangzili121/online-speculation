import pytest
import torch

from blockspec.vllm_ascend.sparse_draft import (
    compact_verifier_queries,
    recent_page_plan,
    reuse_vegas_page_plan,
    vegas_page_plan,
    verifier_guided_page_plan,
)


def test_compact_verifier_queries_keeps_request_boundaries():
    query = torch.arange(7, dtype=torch.float32).view(7, 1, 1)
    compact = compact_verifier_queries(query, torch.tensor([0, 2, 5, 7]))
    assert compact[:, :, 0, 0].tolist() == [[0, 1], [2, 4], [5, 6]]


def test_recent_page_plan_keeps_partial_page_last():
    table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, -1]],
                         dtype=torch.int32)
    plan = recent_page_plan(
        table, [400, 300], 128, budget_ratio=0.5,
        min_tokens=128, max_tokens=1024, dense_below_tokens=0)
    assert plan.block_table.tolist() == [[12, 13], [21, 22]]
    assert plan.sequence_lengths == [144, 172]
    assert plan.selected_blocks == [2, 2]
    assert plan.total_blocks == [4, 3]


def test_verifier_guided_plan_selects_best_old_page_and_newest_page():
    key = torch.zeros(6, 4, 1, 2)
    key[0, -1, 0] = torch.tensor([1.0, 0.0])
    key[1, -1, 0] = torch.tensor([3.0, 0.0])
    key[2, -1, 0] = torch.tensor([2.0, 0.0])
    key[3, -1, 0] = torch.tensor([100.0, 0.0])
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    query = torch.tensor([[[1.0, 0.0]]])
    plan = verifier_guided_page_plan(
        query, key, table, [16], budget_ratio=0.5,
        min_tokens=4, max_tokens=16, dense_below_tokens=0)
    # The newest page is retained regardless of its score; page 1 wins top-k.
    assert plan.block_table.tolist() == [[1, 3]]
    assert plan.sequence_lengths == [8]


def test_verifier_guided_plan_uses_dense_budget_below_threshold():
    key = torch.randn(3, 4, 1, 2)
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    plan = verifier_guided_page_plan(
        torch.randn(1, 1, 2), key, table, [10], budget_ratio=0.1,
        min_tokens=4, max_tokens=16, dense_below_tokens=16)
    assert set(plan.block_table[0].tolist()) == {0, 1, 2}
    assert plan.sequence_lengths == [10]


def test_vegas_plan_finds_important_token_inside_page():
    key = torch.zeros(4, 4, 1, 2)
    key[0, -1, 0] = torch.tensor([2.0, 0.0])
    key[1, 1, 0] = torch.tensor([9.0, 0.0])
    key[2, -1, 0] = torch.tensor([3.0, 0.0])
    key[3, -1, 0] = torch.tensor([100.0, 0.0])
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    queries = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    plan = vegas_page_plan(
        queries, key, table, [16], budget_ratio=0.5,
        min_tokens=4, max_tokens=16, dense_below_tokens=0,
        page_chunk=2)
    assert plan.block_table.tolist() == [[1, 3]]
    assert plan.sequence_lengths == [8]


def test_vegas_plan_is_layer_and_gqa_aware():
    key = torch.zeros(3, 2, 1, 2)
    key[0, 0, 0] = torch.tensor([1.0, 0.0])
    key[1, 0, 0] = torch.tensor([0.0, 2.0])
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    # Two query heads share one KV head. Their mean points to page 1.
    queries = torch.tensor([[
        [[0.0, 1.0], [0.0, 3.0]],
        [[0.0, 1.0], [0.0, 3.0]],
    ]])
    plan = vegas_page_plan(
        queries, key, table, [6], budget_ratio=2 / 3,
        min_tokens=2, max_tokens=6, dense_below_tokens=0)
    assert plan.block_table.tolist() == [[1, 2]]


def test_reuse_vegas_plan_keeps_old_selection_and_advances_newest_page():
    table = torch.tensor([[0, 1, 2, 9]], dtype=torch.int32)
    plan = reuse_vegas_page_plan(
        [torch.tensor([1], dtype=torch.int32)], table, [16], 4,
        budget_ratio=0.5, min_tokens=4, max_tokens=16,
        dense_below_tokens=0)
    assert plan.block_table.tolist() == [[1, 9]]
    assert plan.sequence_lengths == [8]


def test_reuse_vegas_plan_ignores_cache_for_dense_rows():
    table = torch.tensor([[4, 5, 6]], dtype=torch.int32)
    plan = reuse_vegas_page_plan(
        [None], table, [10], 4, budget_ratio=0.1,
        min_tokens=4, max_tokens=16, dense_below_tokens=16)
    assert plan.block_table.tolist() == [[4, 5, 6]]


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1])
def test_page_plan_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="budget_ratio"):
        recent_page_plan(
            torch.tensor([[0]], dtype=torch.int32), [1], 1,
            budget_ratio=ratio, min_tokens=1, max_tokens=1,
            dense_below_tokens=0)
