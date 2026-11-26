from ctypes import memmove
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from blockspec.dsv41_packed_serving import PackedDecodeVerifier


def test_packed_benchmark_matches_tokens_with_ragged_prefill_and_final_round():
    from blockspec.dsv41_packed_proposer import DraftRoundStats

    path = Path(__file__).parents[1] / "scripts" / "validate_dsv41_pypto_packed_generation.py"
    spec = spec_from_file_location("packed_generation", path)
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Session:
        def __init__(self):
            self.draft = SimpleNamespace(stats=DraftRoundStats(), block=16)
            self.context = 256
            self.prefill = {}
            self.limits = []
            self.feedback = None
            self.updates = 0

        def admit(self, request_id, dp, size):
            assert request_id not in self.prefill
            self.prefill[request_id] = size
            return _Handle(request_id, dp)

        def prefill_greedy_batch(self, queries):
            result = {}
            for handle, prompt in queries:
                self.prefill[handle.request_id] -= len(prompt)
                if self.prefill[handle.request_id] == 0:
                    result[handle] = prompt[-1] + 1
            return result

        def decode_greedy_batch(self, queries):
            return {handle: values[0] + 1 for handle, values in queries}

        def speculate_batch(self, queries, *, width, output_limits, student_capture=None):
            self.limits.append(sorted(output_limits.values()))
            emitted = {}
            for handle, anchor in queries:
                size = min(output_limits[handle], 2 if handle.dp else 3)
                emitted[handle] = tuple(range(anchor + 1, anchor + size + 1))
            proposals = {handle: rows[:-1] for handle, rows in emitted.items()}
            if student_capture is not None:
                assert self.feedback is None
                handle, limit = student_capture
                self.feedback = (handle, limit, 0)
            if self.feedback is not None:
                handle, limit, seen = self.feedback
                self.feedback = (handle, limit, seen + len(emitted[handle]))
            return SimpleNamespace(proposals=proposals, verified=SimpleNamespace(emitted=emitted))

        def update_draft(self, handle, *, rate, finished):
            assert self.feedback is not None and self.feedback[0] == handle
            _, limit, seen = self.feedback
            if seen < limit and not finished:
                return None
            self.feedback = None
            self.updates += 1
            from blockspec.dsv41_packed_adam import PackedUpdate

            return PackedUpdate(self.updates, True, min(limit, seen), .01, .5, 1., 0., 0., 0., 0.)

        def release(self, handle):
            assert self.feedback is None
            assert self.prefill.pop(handle.request_id) == 0

    from collections import namedtuple

    _Handle = namedtuple("Handle", "request_id dp")
    session = Session()
    result = runner.benchmark(session, [("a", [3]), ("b", [1, 2, 3, 4, 5])],
                              chunk=2, tokens=6, width=8, concurrency=2, warmup=1)
    assert result["passed"] and result["greedy_tokens_matched"] == 12
    assert session.limits == [[5, 5], [2, 3], [1]] * 2
    orthrus = result["arms"]["orthrus"]["waves"][0]
    assert orthrus["accepted_candidates"] == orthrus["proposed_candidates"] == 5
    assert orthrus["rounds"] == 3
    online = runner.OnlineSchedule(every=1, teacher_rows=4, learning_rate=1e-5, warmup_updates=2)
    updated = runner.benchmark(session, [("a", [3]), ("b", [5]), ("c", [5]), ("d", [2])],
                               chunk=2, tokens=6, width=8, concurrency=2, warmup=0, online=online)
    assert updated["passed"] and updated["greedy_tokens_matched"] == 24
    updates = [update for wave in updated["arms"]["orthrus"]["waves"] for update in wave["updates"]]
    assert [(row["dp"], row["teacher_rows"], row["version"]) for row in updates] == [(0, 4, 1), (1, 4, 2)]
    assert [row["learning_rate"] for row in updates] == [5e-6, 1e-5]
    assert updated["arms"]["orthrus"]["tokens_per_second_excluding_update"] > updated["arms"]["orthrus"]["tokens_per_second"]
    assert not session.prefill


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


@pytest.mark.parametrize("role", ["initialized-plumbing", "trained"])
def test_short_draft_uses_only_a_compatible_head(monkeypatch, role):
    from blockspec import dsv41_packed_proposer as proposer

    head = ({}, {"artifact_role": role, "block_size": 32}, {})
    monkeypatch.setattr(proposer, "read_draft_artifact", lambda *_: head)
    assert proposer.read_packed_head(SimpleNamespace(root=None), None, 32) == head
    if role == "trained":
        with pytest.raises(ValueError, match="trained draft head"):
            proposer.read_packed_head(SimpleNamespace(root=None), None, 16)
    else:
        assert proposer.read_packed_head(SimpleNamespace(root=None), None, 16) == head


@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("lookup_fails", [False, True])
def test_resident_draft_uses_only_committed_history(lookup_fails, block):
    from blockspec.dsv41_packed_proposer import DraftRoundStats, PackedDraftProposer

    position, owner = 129, 4
    original = torch.arange(position)
    history = {owner: original}

    def lookup(tokens, capacity, start, *, request_id):
        assert tokens.tolist() == [7] + [99] * (block - 1)
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
    draft.previous_owners, draft.stats = {0}, DraftRoundStats()
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
        torch.testing.assert_close(active["x_hc"][1:], torch.full((block - 1, 4, 4), 99.0))

    draft.runtime.run = run
    if lookup_fails:
        with pytest.raises(RuntimeError, match="lookup failed"):
            draft.propose(7, position=position, owner=owner, candidates=3, uploads=uploads)
        assert not uploaded
    else:
        assert draft.propose(7, position=position, owner=owner, candidates=3, uploads=uploads) == [16, 17, 18]
        assert uploaded == [(rank, owner) for rank in range(4, 8)] + [(rank, -4) for rank in range(4)]
        assert draft.stats.sampled_id_bytes == 3 * 8 * 4
        assert not draft.counts.any()
    assert history[owner] is original


@pytest.mark.parametrize(("retain_state", "last_layers"), [(False, 0), (True, 0), (True, 4)])
def test_draft_wave_dispatches_once_without_changing_target_counts(retain_state, last_layers):
    from blockspec.dsv41_packed_proposer import DraftRequest, DraftRoundStats, PackedDraftProposer

    draft = PackedDraftProposer.__new__(PackedDraftProposer)
    draft.world, draft.tp, draft.context, draft.block = 16, 4, 512, 32
    draft.stats, draft.previous_owners = DraftRoundStats(), {4}
    draft.sampled = torch.empty(16, 8, dtype=torch.int32)
    counts = torch.tensor([9] * 4 + [0] * 4 + [3] * 4 + [0] * 4, dtype=torch.int32)
    draft.counts = counts.clone()
    draft.ranks = [{"sampled_ids": SimpleNamespace(data_ptr=i), "logits": torch.empty(16, 10)} for i in range(16)]
    draft.arguments, draft.graph, draft.config = [draft.ranks[8]["logits"]], object(), None
    draft.student = {}
    draft.retain_final_state = retain_state
    draft.last_layers = last_layers
    draft._logits_argument = 0, "logits"
    draft._input_arguments = {"x_hc": (1, "x_hc"), "pre_mix": (2, "pre_mix")}
    live = torch.zeros(1), torch.zeros(1)
    draft.arguments.extend(live)
    tape_name = "ffn_tape_residual"
    draft._capture_owner_argument = 3
    draft._tape_arguments = {tape_name: (4, tape_name)}
    draft.tape_specs = {tape_name: ((1,), "float32")}
    draft.arguments.extend((-1, {}))
    draft._rank_argument = lambda name, parameter, *, replacement: (
        replacement if name == tape_name else next(iter(replacement.values()))
    )
    staged, uploaded, launches = [], [], []
    draft._stage = lambda request, *, student_rows: staged.append((request, student_rows))
    image = object()
    views = dict(x_hc=torch.zeros(1), pre_mix=torch.zeros(1))
    draft.inputs = SimpleNamespace(images=torch.zeros(3, 8), reserve=lambda rt, rank: (image, views),
                                   upload=lambda rt, rank, owner, allocation=None:
                                   uploaded.append((rank, owner, allocation)))

    def run(*args, **kwargs):
        launches.append(draft.counts.tolist())
        args[1].fill_(len(launches))
        args[2].fill_(len(launches) * 10)
        args[3].fill_(len(launches) * -10)
        if args[4] != -1:
            assert args[4] == 8 and tuple(args[5]) == (8, 9, 10, 11)
            for rank, tape in args[5].items():
                tape.fill_(rank)

    def readback(dst, src, size, *, worker_id):
        assert src == worker_id
        draft.sampled[:, 0] = torch.arange(16) + worker_id * 10

    freed = []
    draft.runtime = SimpleNamespace(run=run, copy_from=readback,
                                    alloc_tensor=lambda shape, dtype, **_: torch.empty(shape, dtype=dtype),
                                    free_tensor=lambda value, *, worker_id: freed.append((value, worker_id)))
    requests = [DraftRequest(7, 129, 0, 3, "r0", (2, 1)), DraftRequest(14, 95, 8, 2, "r2", (3,))]
    uploads = SimpleNamespace(map=map)
    assert draft.propose_batch(requests, uploads=uploads, capture=("r2", 7)) == {0: [0, 1, 2], 8: [80, 81]}
    assert staged == [(requests[0], 0), (requests[1], 7)]
    assert uploaded == ([(r, 0, None) for r in range(4)]
                        + [(r, 8, image if retain_state and r == 8 else None) for r in range(8, 12)]
                        + [(r, -4, None) for r in range(4, 8)])
    assert launches == [[32] * 4 + [0] * 4 + [32] * 4 + [0] * 4]
    assert torch.equal(draft.counts, counts)
    captured = draft.student["r2"]
    assert (captured.owner, captured.position, captured.rows) == (8, 95, 7)
    draft.propose_batch(requests, uploads=uploads)
    assert torch.all(captured.logits == 1)
    assert torch.all(draft.ranks[8]["logits"] == 2)
    if retain_state:
        assert captured.final_state.streams.item() == 10
        assert captured.final_state.premix.item() == -10
        assert live[0].item() == 20
    if last_layers:
        assert {rank: tape[tape_name].item() for rank, tape in captured.ffn_tape.items()} == {r: r for r in range(8, 12)}
    draft.release_student("r2")
    expected_freed = [(captured.logits, 8)] + ([(image, 8)] if retain_state else [])
    if last_layers:
        expected_freed.extend((tape[tape_name], rank) for rank, tape in captured.ffn_tape.items())
    assert freed == expected_freed
    assert not draft.student


def test_staged_draft_uses_local_worker_ids_and_exchanges_only_candidates():
    from blockspec.dsv41_packed_proposer import DraftRequest, DraftRoundStats, StagedPackedDraftProposer

    draft = StagedPackedDraftProposer.__new__(StagedPackedDraftProposer)
    draft.tp, draft.block, draft.dp_size = 4, 16, 4
    draft.last_layers = 0
    draft.previous_owners, draft.stats = set(), DraftRoundStats()
    draft.sampled = torch.empty(16, 8, dtype=torch.int32)
    draft.ranks = [{"sampled_ids": SimpleNamespace(data_ptr=rank)} for rank in range(8)]
    draft.attention, draft.head = object(), object()
    draft.attention_args = draft.head_args = ()
    draft.attention_rows = draft.ffn_rows = draft.x_hc = draft.pre_mix = object()
    draft.layer_weights = [(None, None, None)] * 2
    observed = []
    draft._stage = lambda request, *, student_rows: observed.append(("stage", request.owner))
    draft.inputs = SimpleNamespace(
        images=torch.zeros(3, 128, dtype=torch.uint8),
        upload=lambda _, rank, owner: observed.append(("upload", rank, owner)),
    )

    def exchange(stage, sent):
        observed.append(("exchange", stage, sent))
        return {"0": [32, 33]}

    def read(dst, src, size, *, worker_id):
        assert src == worker_id == 4
        draft.sampled[0, 0] = 51

    draft.target = SimpleNamespace(global_ranks=range(8, 16), rank_offset=8,
                                   link=SimpleNamespace(exchange=exchange))
    draft.runtime = SimpleNamespace(run=lambda *args: observed.append(("run", *args)), copy_from=read)
    draft.ffn = SimpleNamespace(
        moe=SimpleNamespace(bind_batch=lambda counts: observed.append(("counts", tuple(counts)))),
        run_layer=lambda name, **_: observed.append(("ffn", name)),
    )
    remote = DraftRequest(7, 129, 0, 2, "remote", (0, 1))
    local = DraftRequest(8, 95, 12, 1, "local", (3,))
    uploads = SimpleNamespace(map=map)
    assert draft.propose_batch([remote, local], uploads=uploads) == {0: [32, 33], 12: [51]}
    assert observed == [
        ("stage", 12), *(("upload", rank, 4) for rank in range(4, 8)), ("counts", (16, 0, 0, 16)),
        ("run", draft.attention, 0, -1, 1), ("ffn", "staged-draft/0/layer-0"),
        ("run", draft.attention, 1, -1, 1), ("ffn", "staged-draft/0/layer-1"),
        ("run", draft.head, 1), ("exchange", "staged-draft/0/sampled", {"12": [51]}),
    ]
    observed.clear()
    assert draft.propose_batch([remote], uploads=uploads) == {0: [32, 33]}
    assert observed[:5] == [*(("upload", rank, -4) for rank in range(4, 8)), ("counts", (16, 0, 0, 0))]
    assert observed[-1] == ("exchange", "staged-draft/1/sampled", {})
    assert draft.propose_batch([], uploads=uploads) == {}
    assert draft.stats.calls == 2
    assert draft.stats.sampled_id_bytes == 32


def test_staged_suffix_reverse_both_hosts_follow_same_layer_schedule():
    from concurrent.futures import ThreadPoolExecutor
    import socket

    from blockspec.dsv41_staged_reverse import StagedDraftReverse
    from blockspec.pypto_dsv41_cluster import ControlLink

    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    token = b"reverse-fixture-key"
    student = object()

    def both(node, endpoint):
        local = node == 1
        calls = []
        with ControlLink(endpoint, node, token) as link:
            reverse = StagedDraftReverse.__new__(StagedDraftReverse)
            reverse.layers = (39, 38, 37)
            reverse.draft = SimpleNamespace(
                block=16, layer_weights=[(None, f"weights-{i}") for i in range(40)],
                target=SimpleNamespace(global_ranks=range(node * 8, node * 8 + 8), link=link),
            )
            reverse.loss = lambda who, teacher: SimpleNamespace(value=.125, rows=len(teacher))
            reverse.lm_head = lambda loss: loss.rows
            reverse.final = lambda who, rows: rows
            reverse.attn_input = lambda who, projection: 16

            def post(who, boundary, *, layer):
                assert who is student and boundary == (3 if layer == 39 else 16)
                return SimpleNamespace(rows=boundary)

            def experts(stage, *, owner, layer, rows, weights, student: object, gradient: object):
                assert owner == 12 and weights == f"weights-{layer}"
                assert (student is not None) == (gradient is not None) == local
                peer = link.exchange(stage, [owner, layer, rows])
                assert peer == [owner, layer, rows]
                calls.append((layer, rows))
                return SimpleNamespace(gradients=layer) if local else None

            reverse.ffn_post = post
            reverse.experts = SimpleNamespace(run_layer=experts)
            reverse.ffn_input = lambda who, result, gradients: gradients
            reverse.ffn_mhc = lambda who, post, normed: normed
            reverse.attn_post = lambda who, mixed: mixed
            reverse.wo = lambda who, seed: seed
            reverse.attention = lambda who, projected: projected
            reverse.qkv = lambda who, attention: attention
            feedback = SimpleNamespace(student=student, logits=lambda **_: [1, 2, 3]) if local else None
            consumed = []
            result = reverse.run("suffix", owner=12, feedback=feedback, finished=False,
                                 consume=lambda layer, grad: consumed.append((layer, grad)))
            return result, calls, consumed

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(both, 0, left)
        second = pool.submit(both, 1, right)
        peer, peer_calls, peer_consumed = first.result()
        owner, owner_calls, owner_consumed = second.result()
    assert peer_calls == owner_calls == [(39, 3), (38, 16), (37, 16)]
    assert peer_consumed == [(39, None), (38, None), (37, None)]
    assert owner_consumed == [(39, 39), (38, 38), (37, 37)]
    for result in (peer, owner):
        assert result["owner"] == 12 and result["teacher_rows"] == 3 and result["loss"] == .125


def test_staged_adam_uses_the_same_version_when_one_host_rejects_an_update():
    from concurrent.futures import ThreadPoolExecutor
    import socket

    from blockspec.dsv41_packed_adam import PackedAdam
    from blockspec.dsv41_staged_adam import StagedPackedAdam
    from blockspec.pypto_dsv41_cluster import ControlLink

    def host(node, endpoint):
        actions = []
        with ControlLink(endpoint, node, b"adam-fixture-key") as link:
            optimizer = StagedPackedAdam.__new__(StagedPackedAdam)
            optimizer.link = link
            adam = optimizer.adam = PackedAdam.__new__(PackedAdam)
            adam.version = adam.epoch = 0
            adam.failed = False
            adam.config = None
            adam.layers = (38, 39)
            parameter = SimpleNamespace(name="w", program_key="shape")
            adam.parameters = (parameter,)
            adam.programs = {"shape": ("stage", "apply", "publish")}
            adam.params = {(layer, "w"): SimpleNamespace(weights=layer, state=None) for layer in adam.layers}
            adam.runtime = SimpleNamespace(run=lambda program, weights, *_args, **_kw:
                                           actions.append((program, weights)))
            adam._stage = lambda gradient, _: (4., gradient.valid)

            def exchange(stage, layer, gradient, item):
                assert (gradient == layer) if node == 1 else (gradient is None)
                assert item is parameter
                return SimpleNamespace(layer=layer, valid=(node == 1 or stage != "first"))

            def reverse(_stage, *, owner, feedback, finished, consume):
                assert owner == 12 and (feedback is not None) == (node == 1)
                for layer in reversed(adam.layers):
                    consume(layer, layer if node == 1 else None)
                return dict(owner=owner, teacher_rows=7, loss=.25)

            optimizer._exchange = exchange
            optimizer.reverse = SimpleNamespace(run=reverse)
            results = [optimizer.update(stage, owner=12, feedback=object() if node == 1 else None,
                                        finished=False, rate=1e-5, max_norm=1.)
                       for stage in ("first", "second")]
        return actions, [(item["version"], item["applied"]) for item in results]

    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(host, 0, left)
        second = pool.submit(host, 1, right)
        expected = ([("apply", 39), ("apply", 38), ("publish", 39), ("publish", 38)], [(0, False), (1, True)])
        assert first.result() == second.result() == expected


def test_staged_generation_updates_at_committed_rows_and_reuses_new_head():
    from collections import namedtuple

    path = Path(__file__).parents[1] / "scripts" / "run_dsv41_pypto_staged_target.py"
    spec = spec_from_file_location("staged_target", path)
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    handle_type = namedtuple("Handle", "request_id dp")

    class Session:
        context = 128
        draft = SimpleNamespace(tp=4, block=16)

        def __init__(self):
            self.feedback = {}
            self.captures = []

        def admit(self, request_id, dp, size):
            return handle_type(request_id, dp)

        def prefill_greedy_batch(self, queries):
            return {handle: tokens[-1] + 1 for handle, tokens in queries}

        def speculate_batch(self, queries, *, width, output_limits, student_capture=None):
            if student_capture:
                handle, rows = student_capture
                assert not self.feedback
                self.feedback[handle] = SimpleNamespace(limit=rows, seen=0)
                self.captures.append(handle.dp)
            proposals, emitted = {}, {}
            for handle, anchor in queries:
                # Force multiple waves of authoritative feedback before publishing.
                proposals[handle] = (999,) if output_limits[handle] > 1 else ()
                emitted[handle] = (anchor + 1,)
                if handle in self.feedback:
                    self.feedback[handle].seen += 1
            return SimpleNamespace(proposals=proposals, verified=SimpleNamespace(emitted=emitted))

        def release_student(self, handle):
            self.feedback.pop(handle)

        def release(self, handle):
            assert handle not in self.feedback

    session = Session()
    adam = SimpleNamespace(version=0)

    def update(stage, *, owner, feedback, finished, rate, max_norm):
        assert not finished and feedback.seen == feedback.limit == 3
        adam.version += 1
        return dict(teacher_rows=feedback.limit, version=adam.version, update_seconds=0.)

    cases = [dict(id=str(dp), dp=dp, input_ids=[1, 2], new_tokens=10) for dp in (0, 2)]
    result = runner.generate(session, cases, 16, 4, width=8, teacher_rows=3,
                             optimizer=SimpleNamespace(adam=adam, update=update),
                             online=dict(every=2, max_updates=2, rate=1e-5, max_norm=1.))
    assert session.captures == [0, 2]
    assert [row["round"] for row in result["online"]["updates"]] == [3, 7]
    assert result["online"]["rounds_after_publication"] == 5
    assert result["online"]["version"] == 2
    assert result["generated"] == {str(dp): list(range(3, 13)) for dp in (0, 2)}
