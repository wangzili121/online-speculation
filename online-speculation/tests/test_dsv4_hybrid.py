import pytest


torch = pytest.importorskip("torch")

from blockspec.vllm_ascend.dsv4_hybrid import (
    compressed_logical_indices,
    front_pack_indices,
    gather_paged_cache,
    merge_attention_segments,
    tail_logical_indices,
)


def test_attention_segment_merge_matches_one_softmax():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 4, 5)
    keys = [torch.randn(2, length, 4, 5) for length in (6, 9, 2)]
    values = [torch.randn(2, length, 4, 7) for length in (6, 9, 2)]
    outputs = []
    lses = []
    scores = []
    for key, value in zip(keys, values, strict=True):
        score = torch.einsum("bqhd,bkhd->bqhk", query, key)
        scores.append(score)
        lses.append(torch.logsumexp(score, dim=-1, keepdim=True))
        outputs.append(
            torch.einsum("bqhk,bkhd->bqhd", score.softmax(-1), value)
        )
    merged, merged_lse = merge_attention_segments(outputs, lses)
    all_scores = torch.cat(scores, dim=-1)
    all_values = torch.cat(values, dim=1)
    expected = torch.einsum(
        "bqhk,bkhd->bqhd", all_scores.softmax(-1), all_values
    )
    assert torch.allclose(merged, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        merged_lse, torch.logsumexp(all_scores, dim=-1, keepdim=True)
    )


def test_paged_gather_crosses_pages_and_preserves_padding():
    cache = torch.arange(6 * 4).view(6, 4, 1, 1)
    block_table = torch.tensor([[2, 5, 1], [4, 0, 3]])
    indices = torch.tensor([[0, 3, 4, 9, -1], [1, 4, 7, 8, -1]])
    gathered = gather_paged_cache(
        cache, block_table, indices, validate=True
    ).squeeze((-1, -2))
    assert gathered.tolist() == [
        [8, 11, 20, 5, 0],
        [17, 0, 3, 12, 0],
    ]


def test_hybrid_index_helpers():
    seq_lens = torch.tensor([3, 9, 260])
    assert tail_logical_indices(seq_lens, 4).tolist() == [
        [0, 1, 2, -1],
        [5, 6, 7, 8],
        [256, 257, 258, 259],
    ]
    indices, lengths = compressed_logical_indices(seq_lens, 4)
    assert lengths.tolist() == [0, 2, 65]
    assert indices[0, :4].tolist() == [-1, -1, -1, -1]
    assert indices[1, :4].tolist() == [0, 1, -1, -1]
    packed, valid = front_pack_indices(torch.tensor([[4, -1, 7, -1, 2]]))
    assert packed.tolist() == [[4, 7, 2, -1, -1]]
    assert valid.tolist() == [[True, True, True, False, False]]
