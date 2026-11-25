from ctypes import memmove
from types import SimpleNamespace
import sys

import pytest
import torch

from blockspec.dsv41_packed_serving import PackedDecodeVerifier


@pytest.mark.parametrize("state_capacity", [3, 32])
def test_native_verifier_reuses_host_rows_and_keeps_device_reads_small(monkeypatch, state_capacity):
    retain = state_capacity == 32
    def metadata(positions, requests, table):
        slots = positions + int(table[0, 0]) * 128
        return slots, slots[:, None].expand(-1, 128), positions + 1

    def compressed(positions, requests, table, block_size, divisor, complete):
        return (positions // divisor).masked_fill((positions + 1) % divisor != 0, -1)

    monkeypatch.setitem(sys.modules, "models.deepseek_v4_1_flash_a3.utils",
                        SimpleNamespace(window_metadata=metadata, paged_slots=compressed))

    def image():
        fields = {name: torch.full((16,), -1, dtype=torch.int32) for name in (
            "position_ids", "window_slots", "c1a_compressed_slots",
            "c2a_compressed_slots", "token_to_req_indices", "logit_row_indices",
        )}
        fields.update(x_hc=torch.zeros(16, 4, 4), engram_lookup=torch.zeros(2, 16, 4),
                      window_indices=torch.full((16, 128), -1, dtype=torch.int32),
                      c1a_index_block_table=torch.ones(1, 1, dtype=torch.int32),
                      c2a_index_block_table=torch.ones(1, 1, dtype=torch.int32),
                      query_start_loc=torch.zeros(2, dtype=torch.int32))
        return fields

    fields = [image() for _ in range(3)]
    uploads = []
    images = torch.zeros(3, 4096, dtype=torch.uint8)
    inputs = SimpleNamespace(fields=fields, images=images,
                             upload=lambda runtime, rank, owner: uploads.append((rank, owner)))
    sampled = [torch.zeros(16, 8, dtype=torch.int32) for _ in range(8)]
    states = [torch.full((3, 1, state_capacity, 4), float(rank)) for rank in range(8)]

    def handle(value):
        return SimpleNamespace(data_ptr=value.data_ptr(), shape=value.shape, dtype=value.dtype)

    counts = [torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)]
    launches = []
    rows = []
    history = {0: torch.arange(10), 4: torch.arange(10)}

    def lookup(ids, live, position, *, request_id):
        rows.append((request_id, ids.tolist()))
        history[request_id] = torch.cat((history[request_id][:position], ids))
        return torch.ones(2, live, 4)

    def run(*args, **kwargs):
        owner = int(counts[0].nonzero()[0])
        live = int(counts[0][owner])
        is_replay = bool((fields[2]["logit_row_indices"] < 0).all())
        assert args[-1] == (2 if is_replay else 1)
        launches.append((owner, live, is_replay))
        assert (fields[2]["position_ids"][live:] == -1).all()
        expected_embedding = torch.tensor((11, 12, 13)[:live])[:, None, None].expand(-1, 4, 4)
        torch.testing.assert_close(fields[2]["x_hc"][:live], expected_embedding.float())
        if is_replay:
            for rank in range(owner, owner + 4):
                assert (states[rank] == rank).all()
        else:
            sampled[owner][:live, 0] = torch.tensor((12, 13, 14) if owner == 4 else (12, 44, 14))
        for state in states[owner:owner + 4]:
            state.add_(1)

    runtime = SimpleNamespace(run=run,
                              copy_from=lambda a, b, n, **kwargs: memmove(a, b, n),
                              copy_to=lambda a, b, n, **kwargs: memmove(a, b, n))
    verifier = PackedDecodeVerifier(
        runtime, object(), None, (1,), counts, inputs,
        [{"state_cache": handle(t)} for t in states],
        [{"sampled_ids": handle(t)} for t in sampled],
        SimpleNamespace(lookup_rows=lambda ids: ids.float()[:, None].expand(-1, 4)),
        SimpleNamespace(history=history, lookup=lookup), SimpleNamespace(map=map), tp_size=4, has_replay_mode=True,
    )
    assert verifier.verify(11, (12, 13), position=10, owner=0) == ((12, 44), 1)
    assert verifier.verify(11, (12, 13), position=10, owner=4) == ((12, 13, 14), 2)
    if retain:
        assert launches == [(0, 3, False), (4, 3, False)]
    else:
        assert launches == [(0, 3, False), (0, 2, True), (4, 3, False)]
    assert rows == [(0, [11, 12, 13]), (4, [11, 12, 13])]
    assert history[0].tolist()[-2:] == [11, 12]
    assert history[4].tolist()[-3:] == [11, 12, 13]
    first_uploads = 4 if retain else 8
    assert uploads[:first_uploads] == [(rank, 0) for rank in range(4)] * (first_uploads // 4)
    assert set(uploads[first_uploads:]) == {(rank, 4) for rank in range(8)}
    assert verifier.stats.sampled_id_bytes == 2 * 3 * 8 * 4
    assert verifier.stats.restore_bytes == (0 if retain else 4 * states[0].nbytes)
    assert verifier.stats.snapshot_bytes == (0 if retain else 8 * states[0].nbytes)
    assert verifier.stats.replay_calls == (0 if retain else 1)
    assert verifier.stats.retained_prefix_calls == (2 if retain else 0)


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_resident_draft_uses_only_committed_history(lookup_fails):
    from blockspec.dsv41_packed_proposer import DraftRoundStats, PackedDraftProposer

    position, block, owner = 129, 32, 4
    original = torch.arange(position)
    history = {owner: original}

    def lookup(tokens, capacity, start, *, request_id):
        assert tokens.tolist() == [7] + [99] * 31
        assert (capacity, start, request_id) == (block, position, owner)
        history[owner] = torch.cat((original, tokens))
        if lookup_fails:
            raise RuntimeError("lookup failed")
        return torch.ones(2, block, 4)

    def image():
        return dict(
            x_hc=torch.zeros(block, 4, 4), pre_mix=torch.eye(4)[0].expand(block, -1).clone(),
            engram_rows=torch.zeros(2, block, 4), draft_cos=torch.zeros(2, block, 4),
            draft_sin=torch.zeros(2, block, 4), window_rows=torch.zeros(8, 16, dtype=torch.int32),
            window_starts=torch.zeros(8, dtype=torch.int32), window_spans=torch.zeros(2, 3, dtype=torch.int32),
            window_bias=torch.zeros(1, 128), c1a_table=torch.empty(1, 4, dtype=torch.int32),
            c2a_table=torch.empty(1, 4, dtype=torch.int32), c1a_visible=torch.zeros(block, dtype=torch.int32),
            c2a_visible=torch.zeros(block, dtype=torch.int32), logit_row_indices=torch.full((16,), -1),
        )

    fields = [image() for _ in range(3)]
    sampled = torch.zeros(16, 8, dtype=torch.int32)
    sampled[:3, 0] = torch.tensor([16, 17, 18])
    uploaded = []
    draft = PackedDraftProposer.__new__(PackedDraftProposer)
    draft.world, draft.tp, draft.context, draft.block = 8, 4, 512, block
    draft.pages, draft.buckets, draft.mask_token_id = 5, (512, 512), 99
    draft.sampled, draft.counts = torch.empty_like(sampled), torch.zeros(8, dtype=torch.int32)
    draft.ranks = [dict(sampled_ids=SimpleNamespace(data_ptr=sampled.data_ptr()))] * 8
    draft.runtime = SimpleNamespace(copy_from=lambda a, b, n, **kw: memmove(a, b, n))
    draft.graph, draft.config, draft.arguments = object(), None, ()
    draft.weights = SimpleNamespace(lookup_rows=lambda tokens: tokens.float()[:, None].expand(-1, 4))
    draft.engram = SimpleNamespace(history=history, lookup=lookup)
    draft.page_tables = torch.tensor([2, 1, 3, 4]), torch.tensor([0, 3])
    draft.rope_tables = torch.ones(2, 512, 4), torch.zeros(2, 512, 4)
    draft.inputs = SimpleNamespace(fields=fields, images=torch.zeros(3, 4096, dtype=torch.uint8),
                                   upload=lambda _, rank, active: uploaded.append((rank, active)))
    draft.previous_owner, draft.stats = 0, DraftRoundStats()
    uploads = SimpleNamespace(map=map)

    def run(*args, **kwargs):
        assert history[owner] is original
        assert draft.counts.tolist() == [0] * 4 + [block] * 4
        active = fields[2]
        assert active["logit_row_indices"].tolist() == [0, 1, 2] + [-1] * 13
        assert fields[1]["logit_row_indices"].tolist() == [-1] * 16
        assert active["c1a_table"].tolist() == [[2, 1, -1, -1]]
        assert active["c2a_table"].tolist() == [[0, -1, -1, -1]]
        assert active["c1a_visible"].unique().item() == position
        assert active["c2a_visible"].unique().item() == position // 2
        assert active["window_rows"].flatten()[0].item() == 257
        torch.testing.assert_close(active["x_hc"][0], torch.full((4, 4), 7.0))
        torch.testing.assert_close(active["x_hc"][1:], torch.full((31, 4, 4), 99.0))

    draft.runtime.run = run
    if lookup_fails:
        with pytest.raises(RuntimeError, match="lookup failed"):
            draft.propose(7, position=position, owner=owner, candidates=3, uploads=uploads)
        assert not uploaded
    else:
        assert draft.propose(7, position=position, owner=owner, candidates=3, uploads=uploads) == [16, 17, 18]
        assert set(uploaded) == {(rank, owner) for rank in range(8)}
        assert draft.stats.sampled_id_bytes == 3 * 8 * 4
    assert history[owner] is original
