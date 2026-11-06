import pytest
import torch
from torch.nn import functional as F

from blockspec.pypto_qwen38.attention import AttentionShape, attention_golden, validate_inputs


def fixture(shape, past, width):
    generator = torch.Generator().manual_seed(173)
    rand = lambda dims: torch.randn(dims, generator=generator)
    q, gate = [rand((shape.block_size, shape.query_width)) for _ in range(2)]
    k, v = [rand((shape.block_size, shape.kv_width)) for _ in range(2)]
    pages = shape.context // shape.page_size + 3
    pools = [rand((shape.kv_heads, pages, shape.page_size, shape.head_dim)) for _ in range(2)]
    table = torch.randperm(pages, generator=generator)[:shape.context // shape.page_size].int()
    return q, k, v, gate, *pools, table, past, width


@pytest.mark.parametrize("past,width", [(0, 8), (1, 1), (3, 8), (4, 7), (5, 2), (24, 8)])
def test_paged_golden_matches_dense_bidirectional_sdpa_and_preserves_inputs(past, width):
    shape = AttentionShape(block_size=8, q_heads=6, kv_heads=2, head_dim=16,
                           context=32, page_size=4, query_tile=4)
    args = fixture(shape, past, width)
    before = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    actual = attention_golden(shape, *args)
    q, k, v, gate, kp, vp, table, _, _ = args
    history = [torch.cat([pool[:, table[i]] for i in range((past + 3) // 4)], 1)[:, :past]
               if past else pool[:, 0, :0] for pool in (kp, vp)]
    keys, values = [torch.cat((old, new[:width].view(width, 2, 16).transpose(0, 1)), 1)
                    .repeat_interleave(3, 0) for old, new in zip(history, (k, v))]
    expected = F.scaled_dot_product_attention(q.view(8, 6, 16).transpose(0, 1), keys, values,
                                              is_causal=False).transpose(0, 1).reshape(8, 96) * gate.sigmoid()
    torch.testing.assert_close(actual, expected)
    assert all(torch.equal(old, new) for old, new in zip(before, (x for x in args if isinstance(x, torch.Tensor))))


def test_future_block_positions_are_visible_and_unused_pages_are_not_read():
    shape = AttentionShape(block_size=8, q_heads=2, kv_heads=1, head_dim=16,
                           context=32, page_size=4, query_tile=4)
    args = list(fixture(shape, 0, 8))
    args[0].zero_()
    args[2].zero_()
    args[2][-1].fill_(8)
    args[3].zero_()
    args[4].fill_(float("nan"))
    args[5].fill_(float("nan"))
    args[6].fill_(-1)
    output = attention_golden(shape, *args)
    torch.testing.assert_close(output, torch.full_like(output, .5))


def test_only_committed_history_and_active_draft_rows_contribute():
    shape = AttentionShape(block_size=8, q_heads=2, kv_heads=1, head_dim=16,
                           context=32, page_size=4, query_tile=4)
    args = list(fixture(shape, 5, 3))
    expected = attention_golden(shape, *args)
    for pool in args[4:6]:
        pool[:, args[6][1], 1:].fill_(float("nan"))
    args[1][3:].fill_(float("nan"))
    args[2][3:].fill_(float("nan"))
    args[6][2:] = -1
    torch.testing.assert_close(attention_golden(shape, *args), expected)


@pytest.mark.parametrize("past,width", [(-1, 8), (25, 8), (0, 0), (0, 9)])
def test_invalid_boundaries_are_rejected(past, width):
    shape = AttentionShape(block_size=8, q_heads=2, kv_heads=1, head_dim=16,
                           context=32, page_size=4, query_tile=4)
    with pytest.raises(ValueError):
        validate_inputs(shape, *fixture(shape, past, width))


def test_used_physical_page_must_exist():
    shape = AttentionShape(block_size=8, q_heads=2, kv_heads=1, head_dim=16,
                           context=32, page_size=4, query_tile=4)
    args = list(fixture(shape, 5, 3))
    args[6][0] = -1
    with pytest.raises(ValueError, match="out of range"):
        attention_golden(shape, *args)


def test_unknown_backend_is_rejected_before_importing_pypto():
    from blockspec.pypto_qwen38.programs import make_programs
    from blockspec.pypto_qwen38.draft import NeuralDraft
    with pytest.raises(ValueError, match="backend"):
        make_programs(None, attention_backend="unknown")
    with pytest.raises(ValueError, match="backend"):
        NeuralDraft(None, None, attention_backend="unknown")


@pytest.mark.parametrize("kwargs", [{"physical_pages": 0}, {"pool_layers": -1},
                                    {"capacity": 64}, {"capacity": True}])
def test_paged_backend_rejects_incompatible_storage_or_capacity(kwargs):
    from blockspec.pypto_qwen38.paged_attention_programs import make_paged_attention
    with pytest.raises(ValueError):
        make_paged_attention(**kwargs)


@pytest.mark.parametrize("tile", [4, 8])
def test_paged_backend_does_not_advertise_unvalidated_cube_or_vector_tiles(tile):
    from blockspec.pypto_qwen38.paged_attention_programs import make_paged_attention
    with pytest.raises(ValueError, match="query_tile=16"):
        make_paged_attention(AttentionShape(query_tile=tile))
