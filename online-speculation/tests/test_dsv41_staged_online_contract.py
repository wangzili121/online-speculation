from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest


def load_script(name):
    spec = spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("last_layers", (1, 2))
def test_staged_online_cross_host_feedback_publication_and_summary(last_layers, tmp_path):
    pytest.importorskip("pypto")
    native = pytest.importorskip("models.deepseek_v4_1_flash_a3.packed")
    from models.deepseek_v4_1_flash_a3 import config

    from blockspec.dsv41_packed_adam import PackedAdam
    from blockspec.dsv41_packed_proposer import PackedStudentRows
    from blockspec.dsv41_staged_adam import StagedPackedAdam
    from blockspec.dsv41_staged_reverse import StagedDraftReverse
    from blockspec.pypto_dsv41_cluster import ControlLink

    runner = load_script("run_dsv41_pypto_staged_target")
    summary = load_script("summarize_dsv41_gsm8k")
    cases = [dict(id=f"{dp}-{slot}", dp=dp, input_ids=[5, 6], new_tokens=10)
             for slot in range(2) for dp in (0, 2)]
    heldout = [dict(id="heldout", dp=2, input_ids=[10, 11], new_tokens=4)]
    layers = tuple(range(config.FLASH.num_hidden_layers - 1,
                         config.FLASH.num_hidden_layers - last_layers - 1, -1))

    def host(node, endpoint):
        with ControlLink(endpoint, node, b"staged-online-contract-key") as link:
            history = {}
            target = SimpleNamespace(link=link, global_ranks=range(node * 8, node * 8 + 8),
                                     owns=lambda dp: dp // 2 == node, clear=lambda *_: None)
            graphs = SimpleNamespace(staged_target=target, engram=SimpleNamespace(history=history))
            draft = SimpleNamespace(tp=4, block=16, student={}, target=target,
                                    layer_weights=[(None, layer) for layer in range(layers[0] + 1)])
            session = native.PackedSession(graphs, 2, 128, (16, 16), draft=draft)
            adam, optimizer = PackedAdam.__new__(PackedAdam), StagedPackedAdam.__new__(StagedPackedAdam)
            optimizer.adam, optimizer.link, optimizer.packet_capacity = adam, link, 128
            adam.version = adam.epoch = 0
            adam.failed, adam.layers, adam.config = False, layers, None
            adam.parameters = (SimpleNamespace(program_key="shape", name="weight"),)
            adam.programs = {"shape": ("stage", "apply", "publish")}
            adam.params = {(layer, "weight"): SimpleNamespace(weights=layer, state=None) for layer in layers}
            published = []
            adam.runtime = SimpleNamespace(run=lambda op, layer, *_args, **_kw: published.append((op, layer)))
            adam._stage = lambda *_: (1.0, True)
            versions_on_draft = []

            def propose(requests, *, uploads, capture=None):
                if requests:
                    versions_on_draft.append(adam.version)
                if capture:
                    handle, rows = capture
                    if target.owns(handle.dp):
                        request = next(item for item in requests if item.history_key is handle)
                        draft.student[handle] = PackedStudentRows(request.owner - node * 8, request.position, rows, None)
                return {request.owner: [999] * request.candidates for request in requests}

            def dispatch(groups, phase, capacity, *, sample_all, greedy):
                # CPU oracle rejects every draft candidate. Native Verify owns the
                # truncation, teacher selection, commits, and release.
                results = {}
                for group in groups:
                    for record, tokens in group:
                        handle = record.handle
                        if target.owns(handle.dp):
                            history[handle] = history.get(handle, [])[:record.committed] + list(tokens)
                        sample = [token + 1 for token in tokens]
                        results[handle] = sample if sample_all else sample[-1:]
                return results

            def read(phase, dp, name, output, *, row):
                assert phase == 1 and target.owns(dp) and name == "logits"
                output.fill_(row + 1)

            target.read = read
            draft.propose_batch = propose
            draft.release_student = lambda handle: draft.student.pop(handle, None)
            session._dispatch = dispatch
            reverse = optimizer.reverse = StagedDraftReverse.__new__(StagedDraftReverse)
            reverse.draft, reverse.layers = draft, layers
            reverse.loss = lambda student, teacher: SimpleNamespace(value=.25, rows=len(teacher))
            reverse.lm_head = reverse.final = lambda *items: items[-1]
            reverse.ffn_post = lambda _student, boundary, *, layer: (layer, boundary)
            reverse.ffn_input = reverse.ffn_mhc = reverse.attn_post = reverse.wo = lambda *items: items[-1]
            reverse.attention = reverse.qkv = reverse.attn_input = lambda *items: items[-1]

            def experts(stage, *, owner, layer, rows, weights, student, gradient):
                assert weights == layer and (student is not None) == target.owns(owner // draft.tp)
                assert rows == (3 if layer == layers[0] else draft.block)
                assert link.exchange(stage, [owner, layer, rows]) == [owner, layer, rows]
                return SimpleNamespace(gradients=layer)

            reverse.experts = SimpleNamespace(run_layer=experts)
            optimizer._exchange = lambda _stage, layer, source, _item: SimpleNamespace(layer=layer)
            ar = runner.generate(session, cases, 16, 4, slots_per_dp=2)
            eval_ar = runner.generate(session, heldout, 16, 4, slots_per_dp=2)
            eval_initial = runner.generate(session, heldout, 16, 4, width=8, expected=eval_ar["generated"])
            online = runner.generate(
                session, cases, 16, 4, width=8, teacher_rows=3, optimizer=optimizer, pack_decode=True,
                online=dict(every=2, max_updates=2, rate=1e-5, max_norm=1.), slots_per_dp=2, expected=ar["generated"],
            )
            eval_frozen = runner.generate(session, heldout, 16, 4, width=8, expected=eval_ar["generated"])
            assert not session.feedback and not draft.student and not session.history
            assert published == [(op, layer) for _ in range(2) for op in ("apply", "publish") for layer in layers]
            assert set(versions_on_draft) == {0, 1, 2} and versions_on_draft[-1] == 2
            return dict(ar=ar, orthrus=online, eval_ar=eval_ar, eval_initial=eval_initial, eval_frozen=eval_frozen)

    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=2) as pool:
        left.settimeout(15)
        right.settimeout(15)
        jobs = [pool.submit(host, node, endpoint) for node, endpoint in enumerate((left, right))]
        for node, job in enumerate(jobs):
            arms = job.result()
            ar, online = arms["ar"], arms["orthrus"]
            assert runner.compare_generation(ar, online)["passed"]
            updates = online["online"]["updates"]
            assert [(row["dp"], row["owner"], row["teacher_rows"], row["version"]) for row in updates] == [
                (0, 0, 3, 1), (2, 8, 3, 2),
            ]
            assert online["speculation"]["packed_decode_requests"] > 0
            result = tmp_path / f"node{node}.json"
            result.write_text(json.dumps(dict(node=node, passed=True, arms=arms, **ar)))
            summarized = summary.load_run(result)["arms"]
            assert "decode_speedup" in summarized["eval_initial"] and "decode_speedup" in summarized["eval_frozen"]
            measured = summarized["orthrus"]["online"]
            assert measured["publications"] == measured["head_version"] == 2
            assert measured["rounds_after_publication"] > 0
