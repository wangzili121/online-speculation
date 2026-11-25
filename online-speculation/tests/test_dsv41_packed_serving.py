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
