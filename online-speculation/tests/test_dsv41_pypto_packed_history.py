import pytest

from blockspec.pypto_dsv41_packed_history import compressed_plan, window_plan
from blockspec.pypto_dsv41_packed_indexer import packed_index_controls


def test_swa_window_wraps_to_a_nonadjacent_page_once():
    plan = window_plan((2, 0, 1), 271, 4)
    assert plan.rows == (*range(15, 128), *range(128, 143))
    plan = window_plan((2, 0, 3), 271, 4)
    assert plan.rows == (*range(15, 128), *range(384, 399))
    assert plan.contiguous_starts == (15, 31, 47, 63, 79, 95, 111, -1)


def test_swa_unwritten_rows_are_masked():
    plan = window_plan((0, 3), 19, 4)
    assert plan.rows == (*range(19), *([-1] * 109))
    assert plan.contiguous_starts == (0, -1, -2, -2, -2, -2, -2, -2)


def test_c2a_pages_have_128_compressed_rows_and_page_zero_is_valid():
    selection = [127, 128, 129, 191, -1] + [0] * 11
    plan = compressed_plan((3, 0), 385, 2, 4, selection)
    assert plan.rows == (511, 0, 1, 63, -1, *([384] * 11))
    assert plan.contiguous_starts == (-1,)
    with pytest.raises(ValueError, match="beyond committed"):
        compressed_plan((3, 0), 385, 2, 4, [192] * 16)


def test_c1a_rejects_unmapped_committed_page():
    with pytest.raises(ValueError, match="no physical page"):
        compressed_plan((2, -1), 256, 1, 4, [128] * 16)


@pytest.mark.parametrize(("ratio", "anchor"), [(1, 129), (2, 259)])
def test_packed_index_prefix_excludes_partial_groups_and_keeps_page_zero(ratio, anchor):
    bucket, table, request_ids, lengths = packed_index_controls((3, 0, 2), anchor, ratio, 8)
    assert bucket == 512
    assert table.tolist() == [[3, 0, -1, -1]]
    assert request_ids.tolist() == [0] * 16
    assert lengths.tolist() == [129] * 16
    with pytest.raises(ValueError, match="allocated physical pages"):
        packed_index_controls((3, -1), anchor, ratio, 8)


def test_verifier_journal_covers_only_written_source_rows_and_rolling_state():
    from types import SimpleNamespace

    import torch

    from blockspec.dsv41_packed_verification import CacheWrite, verifier_cache_writes

    config = SimpleNamespace(num_hidden_layers=3, kv_source_layer_ids=(1, 2), compress_ratios=(0, 2, 1))
    caches = {
        "window_cache": torch.empty(3, 4, 128, 1, 512, dtype=torch.uint8),
        "window_scales": torch.empty(3, 4, 128, 1, 16, dtype=torch.bfloat16),
        "compressed_cache": torch.empty(2, 4, 128, 1, 256, dtype=torch.uint8),
        "compressed_scales": torch.empty(2, 4, 128, 1, 32, dtype=torch.bfloat16),
        "index_cache": torch.empty(2, 4, 128, 1, 64, dtype=torch.uint8),
        "index_scales": torch.empty(2, 4, 128, 1, 4, dtype=torch.uint8),
        "state_cache": torch.empty(1, 2, 4, 1024),
    }
    writes = list(verifier_cache_writes(config, caches, anchor=255, tokens=3,
                                        window_pages=(3, 2, 0), c1a_pages=(3, 2, 0),
                                        c2a_pages=(1, 0), state_block=1))
    assert writes[:4] == [
        CacheWrite("window_cache", (2 * 128 + 127) * 512, 512),
        CacheWrite("window_cache", 0, 2 * 512),
        CacheWrite("window_scales", (2 * 128 + 127) * 32, 32),
        CacheWrite("window_scales", 0, 2 * 32),
    ]
    assert [x for x in writes if x.name == "compressed_cache"] == [
        CacheWrite("compressed_cache", (128 + 127) * 256, 256),
        CacheWrite("compressed_cache", 0, 256),
        CacheWrite("compressed_cache", (4 * 128 + 2 * 128 + 127) * 256, 256),
        CacheWrite("compressed_cache", 4 * 128 * 256, 2 * 256),
    ]
    assert [x for x in writes if x.name == "state_cache"] == [
        CacheWrite("state_cache", 4 * 1024 * 4, 4 * 1024 * 4),
    ]
    assert sum(x.size for x in writes) < sum(t.numel() * t.element_size() for t in caches.values())


def test_packed_swa_controls_keep_the_normalization_mask_for_a_short_prefix():
    from blockspec.pypto_dsv41_packed_swa import packed_swa_controls

    rows, starts, spans, bias = packed_swa_controls((0, 3), 19, 4)
    assert rows.shape == (8, 16)
    assert starts.tolist() == [0, -1, -2, -2, -2, -2, -2, -2]
    assert spans.tolist() == [[0, 0, 128], [0, 0, 0]]
    assert bias.shape == (1, 128)
    assert not bias[0, :19].any()
    assert (bias[0, 19:] < -1e20).all()


@pytest.mark.parametrize(
    ("predictions", "committed", "emitted", "accepted", "calls"),
    [
        ((12, 13, 14), [11, 12, 13], (12, 13, 14), 2, 1),
        ((12, 44, 14), [11, 12], (12, 44), 1, 2),
        ((55, 13, 14), [11], (55,), 0, 2),
    ],
)
def test_verify_uses_packed_snapshot_to_commit_only_accepted_rows(
    predictions, committed, emitted, accepted, calls,
):
    from blockspec.dsv41_packed_verification import verify_packed_round

    cache = [10]
    forwards = []

    def forward(tokens):
        forwards.append(tokens)
        cache.extend(tokens)
        return predictions[:len(tokens)]

    def restore(snapshot):
        cache[:] = snapshot

    actual = verify_packed_round(11, (12, 13), (), forward=forward,
                                 save=lambda writes: cache.copy(), restore=restore)
    assert actual == (emitted, accepted)
    assert cache == [10, *committed]
    assert len(forwards) == calls


def test_verify_restores_after_an_invalid_number_of_logits():
    from blockspec.dsv41_packed_verification import verify_packed_round

    cache = [10]

    def forward(tokens):
        cache.extend(tokens)
        return ()

    def restore(snapshot):
        cache[:] = snapshot

    with pytest.raises(ValueError, match="one prediction for each input row"):
        verify_packed_round(11, (12, 13), (), forward=forward,
                            save=lambda writes: cache.copy(), restore=restore)
    assert cache == [10]


def test_verify_restores_again_when_replaying_an_accepted_prefix_fails():
    from blockspec.dsv41_packed_verification import verify_packed_round

    cache = [10]

    def forward(tokens):
        cache.extend(tokens)
        if len(tokens) == 2:
            raise RuntimeError("replay failed")
        return (12, 44, 14)

    def restore(snapshot):
        cache[:] = snapshot

    with pytest.raises(RuntimeError, match="replay failed"):
        verify_packed_round(11, (12, 13), (), forward=forward,
                            save=lambda writes: cache.copy(), restore=restore)
    assert cache == [10]


def test_packed_swa_views_borrow_only_the_rank_local_layer():
    from types import SimpleNamespace

    from blockspec.pypto_dsv41_packed_swa import packed_swa_views

    calls = []
    blobs = {
        "window_cache": SimpleNamespace(shape=(40, 129, 128, 1, 512), nbytes=40 * 129 * 128 * 512),
        "window_scales": SimpleNamespace(shape=(40, 129, 128, 1, 16), nbytes=40 * 129 * 128 * 16 * 2),
    }

    def view(bank, shape, *, byte_offset, worker_id):
        calls.append((bank, shape, byte_offset, worker_id))
        return byte_offset

    actual = packed_swa_views(SimpleNamespace(view_tensor=view), blobs, 39, worker_id=5)
    assert actual == (39 * 129 * 128 * 512, 39 * 129 * 128 * 16 * 2)
    assert [call[1] for call in calls] == [[129, 128, 1, 512], [129, 128, 1, 16]]
    assert [call[-1] for call in calls] == [5, 5]
