"""Exercise the offline acceptance gate without loading a model."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pytest
import torch


COMMAND = Path(__file__).resolve().parents[1] / "scripts/verify_dsv41_pypto_acceptance.py"
SUMMARY = COMMAND.with_name("summarize_dsv41_pypto_cluster.py")


def test_gsm8k_benchmark_binds_learner_only_after_matched_fixed_arm(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(COMMAND.parent))
    module = runpy.run_path(str(COMMAND.with_name("run_dsv41_pypto_cluster.py")))
    benchmark = module["benchmark_workload"]
    calls = []
    session = SimpleNamespace(config=SimpleNamespace(topology={"dp": 4}), draft=object(),
                              native=object(), cluster=object(), online=None)

    def bind(draft, native, cluster):
        assert len(calls) == 4 and session.online is None
        assert (draft, native, cluster) == (session.draft, session.native, session.cluster)

    online = SimpleNamespace(bind=bind, summary=lambda: {"updates": 3})

    def generate(runner, prompts, tokens, assignments, *, speculative, request_start,
                 identities=None, metadata=None, decoder=None, reference=None):
        calls.append((speculative, request_start, session.online is not None, tokens, assignments))
        rows = [dict(index=i, prompt_ids=prompt, token_ids=[i] * tokens,
                     finish_reason="length", usage={"completion_tokens": tokens})
                for i, prompt in enumerate(prompts)]
        if reference is not None:
            assert rows == reference
        return rows, {}

    monkeypatch.setitem(benchmark.__globals__, "generate", generate)
    results, warmup = benchmark(
        object(), session, online, [[3], [4], [5]], [{"id": str(i)} for i in range(3)], 96,
        None, [0, 1], tmp_path,
    )
    assert calls == [
        (False, -1, False, 8, [0]), (False, 0, False, 96, [0, 1]),
        (True, -1, False, 8, [0]), (True, 3, False, 96, [0, 1]),
        (True, 0, True, 96, [0, 1]),
    ]
    assert set(warmup) == {"target", "fixed", "learner_bind"}
    assert module["request_parity"](results["fixed"], results["orthrus"]) == 192
    assert all((tmp_path / arm / "result.json").exists() for arm in ("target", "fixed", "orthrus"))


@pytest.mark.parametrize(("peer_affordable", "expected_updates"), [(True, 6), (False, 4)])
def test_profile_runs_matched_probes_around_bounded_updates(tmp_path, monkeypatch, peer_affordable, expected_updates):
    from blockspec.dsv41_online_profile import OnlineUpdateProfile, active_profile

    monkeypatch.syspath_prepend(str(COMMAND.parent))
    monkeypatch.setitem(sys.modules, "pypto_serving.config.types", SimpleNamespace(
        DecodeBatch=SimpleNamespace, PrefillBatch=SimpleNamespace, SamplingParams=SimpleNamespace,
    ))
    entry = runpy.run_path(str(COMMAND.with_name("run_dsv41_pypto_cluster.py")))
    profile = OnlineUpdateProfile(tmp_path / "online-profile.jsonl")
    receipts = tmp_path / "native/session-test/serving-dispatches.jsonl"
    receipts.parent.mkdir(parents=True)

    class Runner:
        cache_plan = SimpleNamespace(groups=[])
        config = SimpleNamespace(chunk_tokens=16)
        ledger = SimpleNamespace(requests={})
        version = 0

        def run(self, batch, phase):
            name, = batch.request_ids
            if phase == "prefill":
                self.ledger.requests[name] = SimpleNamespace(history=batch.token_ids.tolist())
                self.partition = batch.cache_partitions[0]
                self.steps = 0
                return SimpleNamespace(sampled_token_ids=[1])
            self.ledger.requests[name].history.extend(batch.token_ids.tolist())
            return SimpleNamespace(accepted_token_ids=None, sampled_token_ids=batch.token_ids + 1)

        def run_orthrus(self, batch):
            name, = batch.request_ids
            with receipts.open("a") as stream:
                stream.write(json.dumps(dict(phase="draft", draft_head_version=self.version,
                                            owners=[{"request_id": name}])) + "\n")
            tokens = batch.token_ids.tolist()
            accepted = self.version > 0 and self.steps % 2 == 0
            self.ledger.requests[name].history.extend([*tokens, tokens[0] + 1] if accepted else tokens)
            self.steps += 1 + int(accepted)
            if self.steps >= 7 and active_profile().can_capture():
                self.version += 1
                contract = dict(update_number=self.version, request_id=name, owner=[self.partition, 0, 1], teacher_rows=7)
                with profile.update(contract, owner_is_local=self.partition == 0) as row:
                    row.update(update_seconds=2.0, publication_seconds=0.1)
            return SimpleNamespace(accepted_token_ids=[[tokens[0] + i for i in range(1, 2 + accepted)]],
                                   num_draft_tokens=[31])

        def release_finished_requests(self, names):
            for name in names:
                self.ledger.requests.pop(name)

    def exchange(tag, local):
        assert tag == "acceptance/online-profile-extra" and local is True
        return peer_affordable

    runner = Runner()
    phases, comparisons = entry["profile_workload"](
        runner, SimpleNamespace(exchange=exchange), profile, [[100 + i] for i in range(5)],
        [{"id": str(i)} for i in range(5)], 8, None, tmp_path,
    )
    assert list(phases) == ["ar", "fixed_before", "update", *(
        ["update_extra"] if peer_affordable else []
    ), "fixed_after"]
    assert comparisons == {phase: {"matched_tokens": 32, "divergence": None}
                           for phase in ("fixed_before", "fixed_after")}
    assert [row["owner_dp"] for row in profile.records] == ([0, 2, 0, 0, 0, 0] if peer_affordable else [0, 2, 0, 0])
    assert len({row["request_id"] for row in profile.requests}) == len(profile.requests)
    versions = entry["profile_head_versions"](receipts, profile.requests)
    assert versions == {
        "fixed_before": [0], "fixed_after": [expected_updates],
    }
    gate = entry["profile_acceptance"](phases, comparisons, profile.records, versions,
                                      teacher_rows=7, publications=expected_updates)
    assert gate["passed"] and gate["expected_updates"] == expected_updates
    assert gate["compared_tokens"] == 64
    assert gate["speculation"]["draft_rounds"] == sum(
        arm["speculation"].get("draft_rounds", 0) for arm in phases.values()
    )
    assert phases["fixed_after"]["mean_accepted_candidates"] > phases["fixed_before"]["mean_accepted_candidates"]
    failure = tmp_path / "acceptance.json"
    failure.write_text(json.dumps(dict(passed=False, error="TypeError in final aggregation")))
    result = subprocess.run([sys.executable, str(SUMMARY), str(tmp_path), "--online-profile"],
                            check=True, capture_output=True, text=True)
    recovered = json.loads(result.stdout)
    assert recovered["greedy_comparison"] == comparisons
    assert recovered["probe_head_versions"] == {"fixed_before": [0], "fixed_after": [expected_updates]}
    assert len(recovered["updates"]) == expected_updates
    assert recovered["hot_update_median_seconds"] == 2.0
    assert recovered["probes"]["fixed_after"]["accepted"] > recovered["probes"]["fixed_before"]["accepted"]
    assert not json.loads(failure.read_text())["passed"]
    with pytest.raises(ValueError, match="fixed_after has 1 probes; AR has 2"):
        entry["profile_comparisons"]({
            **phases, "fixed_after": {"requests": phases["fixed_after"]["requests"][:1]},
        })


def test_online_kernel_report_correlates_host_and_device_spans(tmp_path):
    acceptance = tmp_path / "acceptance"
    session = acceptance / "native/session-test"
    session.mkdir(parents=True)
    request = {"request_id": "orthrus-request-8", "phase": "update"}
    (acceptance / "online-profile-update.json").write_text(json.dumps({"timing": [request]}))
    receipt = dict(phase="verify", owners=[request], seconds=0.31, timings={"chunk_seconds": 0.21})
    (session / "serving-dispatches.jsonl").write_text(json.dumps(receipt) + "\n")
    calls = [
        dict(start_ns=100, end_ns=200, update=2, operator="attention", rank=8, device=0, tid=11,
             first=True, counters={}),
        dict(start_ns=103, end_ns=315, update=2, operator="attention", rank=9, device=1, tid=12,
             first=False, counters={}),
    ]
    (acceptance / "online-profile-dispatches.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in calls)
    )
    (tmp_path / "cluster-accept.log").write_text(
        "[mono_ns=107] Callable image: device=0 upload_bytes=1024 children=3\n"
        "[STRACE] pid=1 tid=11 inv=6 hid=aa name=chip.run.runner_run.device_wall ts=0 dur=16 clk=dev dev_id=0\n"
        "[STRACE] pid=1 tid=11 inv=6 hid=aa name=chip.run ts=110 dur=40\n"
        "[STRACE] pid=1 tid=12 inv=7 hid=bb name=chip.run ts=120 dur=48\n"
    )
    target = acceptance / "online-profile-kernels.json"
    result = subprocess.run([sys.executable, str(SUMMARY), str(acceptance), "--online-kernels",
                             "--output", str(target)], check=True, capture_output=True, text=True)
    compact = json.loads(result.stdout)
    report = json.loads(target.read_text())
    first, second = report["operators"]
    assert (first["device_ns"], first["host_run_ns"], first["callable_upload_bytes"]) == (16, 40, 1024)
    assert second["calls_with_device_span"] == 0 and "device_ns" not in second
    assert second["host_run_ns"] == 48
    assert report["serving_phases"]["update"]["verify"]["wall_seconds"] == 0.31
    assert compact["calls"] == 2 and compact["calls_with_device_span"] == 1


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.generation = dict(status="executed", sampling={"temperature": 0}, devices=[0],
                               requests=[dict(index=0, prompt_ids=[1], token_ids=[2, 3],
                                              finish_reason="length", usage={"completion_tokens": 2})])
        owner = dict(dp=0, slot=0, generation=1)
        detail = dict(completed_layers=list(range(40)), TP_logits_equal=True, owners=[owner])
        self.round = [dict(phase="prefill"), dict(phase="draft", native_epoch=0, **detail),
                      dict(phase="verify", native_epoch=0, **detail,
                           runtime=dict(verify_backup_d2d_bytes=128, verify_restore_d2d_bytes=128)),
                      dict(phase="accept", native_epoch=1, **detail)]
        for mode in ("target", "orthrus"):
            result = self.root / mode
            result.mkdir()
            (result / "result.json").write_text(json.dumps(self.generation))
            session = self.root / f"{mode}-native" / "session-1"
            session.mkdir(parents=True)
            (session / "run-manifest.json").write_text(json.dumps(dict(
                orthrus_head="head" if mode == "orthrus" else None, checkpoint={"config": "same"},
            )))
            (session / "cleanup.json").write_text(json.dumps(dict(closed=True, runtime_closed=True, errors=[])))
        self.write_dispatches("target", [dict(phase="prefill"), dict(phase="decode")])
        self.write_dispatches("orthrus", self.round)

    def write_dispatches(self, mode, rows):
        path = self.root / f"{mode}-native" / "session-1" / "serving-dispatches.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_acceptance(self):
        result = subprocess.run([sys.executable, str(COMMAND), str(self.root)], check=False,
                                capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)

    def test_matching_tokens_and_complete_speculative_round_pass(self):
        code, report = self.run_acceptance()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["verifier_restored_bytes"], 128)

    def test_matching_tokens_with_only_fallback_decode_fail(self):
        self.write_dispatches("orthrus", [dict(phase="prefill"), dict(phase="decode")])
        code, report = self.run_acceptance()
        self.assertEqual(code, 1)
        self.assertIn("No Orthrus round executed", report["error"])

    def test_missing_cache_restore_fails(self):
        self.round[2]["runtime"]["verify_restore_d2d_bytes"] = 0
        self.write_dispatches("orthrus", self.round)
        code, report = self.run_acceptance()
        self.assertEqual(code, 1)
        self.assertIn("snapshot and restore disagree", report["error"])

    def test_token_divergence_reports_first_offset(self):
        self.generation["requests"][0]["token_ids"] = [2, 8]
        (self.root / "orthrus" / "result.json").write_text(json.dumps(self.generation))
        code, report = self.run_acceptance()
        self.assertEqual(code, 1)
        self.assertIn("generated token index 1", report["error"])

    def test_shared_runtime_audits_modes_without_allowing_an_early_draft(self):
        session = self.root / "native" / "session-1"
        session.mkdir(parents=True)
        (session / "run-manifest.json").write_text(json.dumps(dict(
            orthrus_head="head", deferred_draft=True, checkpoint={"config": "same"}, ranks=[{"dp_rank": 0}],
        )))
        (session / "cleanup.json").write_text(json.dumps(dict(closed=True, runtime_closed=True, errors=[])))
        base_owner = dict(dp=0, slot=0, generation=1, request_id="target-request-0")
        draft_owner = dict(dp=0, slot=0, generation=2, request_id="orthrus-request-0")
        rows = [dict(phase=phase, owners=[base_owner], draft_resident=False) for phase in ("prefill", "decode")]
        rows.append(dict(phase="release", owner=base_owner, draft_resident=False))
        rows.extend(dict(row, owners=[draft_owner], draft_resident=row["phase"] != "prefill") for row in self.round)
        rows.append(dict(phase="release", owner=draft_owner, draft_resident=True))
        receipt = session / "serving-dispatches.jsonl"
        receipt.write_text("".join(json.dumps(row) + "\n" for row in rows))
        code, report = self.run_acceptance()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["target_phases"], dict(prefill=1, decode=1, release=1))
        rows[0]["draft_resident"] = True
        receipt.write_text("".join(json.dumps(row) + "\n" for row in rows))
        code, report = self.run_acceptance()
        self.assertEqual(code, 1)
        self.assertIn("resident draft", report["error"])

    def test_single_request_peer_does_not_need_to_restore_unowned_kv(self):
        session = self.root / "native" / "session-1"
        session.mkdir(parents=True)
        (session / "run-manifest.json").write_text(json.dumps(dict(
            orthrus_head="head", deferred_draft=True, checkpoint={"config": "same"},
            ranks=[{"dp_rank": 2}],
        )))
        (session / "cleanup.json").write_text(json.dumps(dict(closed=True, runtime_closed=True, errors=[])))
        target = dict(dp=0, slot=0, generation=1, request_id="target-request-0")
        draft = dict(dp=0, slot=0, generation=2, request_id="orthrus-request-0")
        self.round[2]["runtime"] = dict(verify_backup_d2d_bytes=0, verify_restore_d2d_bytes=0)
        rows = [
            dict(phase="prefill", owners=[target], draft_resident=False),
            dict(phase="decode", owners=[target], draft_resident=False),
            dict(phase="release", owner=target, draft_resident=False),
            *(dict(row, owners=[draft], draft_resident=row["phase"] != "prefill") for row in self.round),
            dict(phase="release", owner=draft, draft_resident=True),
        ]
        (session / "serving-dispatches.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
        )
        code, report = self.run_acceptance()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["verifier_restored_bytes"], 0)

    def test_online_capture_checks_both_hosts_and_requires_publication(self):
        audit = runpy.run_path(str(COMMAND))["audit_online_update"]
        completed = dict(bound=True, updates=1, publications=1)
        audit(completed, completed, publish_every=1, require_update=True)
        with self.assertRaisesRegex(ValueError, "disagree on online update or publication"):
            audit(completed, completed | {"publications": 0}, publish_every=1, require_update=True)
        unpublished = completed | {"publications": 0}
        with self.assertRaisesRegex(ValueError, "configured cadence"):
            audit(unpublished, unpublished, publish_every=1, require_update=True)
        empty = unpublished | {"updates": 0}
        with self.assertRaisesRegex(ValueError, "without applying an update"):
            audit(empty, empty, publish_every=1, require_update=True)

    def test_online_capture_requires_a_draft_with_published_weights(self):
        audit = runpy.run_path(str(COMMAND))["audit_session"]
        directory = self.root / "orthrus-native"
        with self.assertRaisesRegex(ValueError, "No draft round used the updated head"):
            audit(directory, speculative=True, require_updated_draft=True)
        self.round[1]["draft_head_version"] = 1
        self.write_dispatches("orthrus", self.round)
        audit(directory, speculative=True, require_updated_draft=True)

    def test_timing_summary_separates_first_call_and_incomplete_receipt(self):
        session = self.root / "native" / "session-1"
        session.mkdir(parents=True)
        rows = [
            dict(phase="verify", owners=[dict(request_id="orthrus-request-0")], seconds=duration,
                 timings=dict(chunk_seconds=duration / 2),
                 runtime=dict(verify_backup_d2d_bytes=128 * (index + 1)))
            for index, duration in enumerate((0.8, 0.1, 0.3))
        ]
        (session / "serving-dispatches.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows) + '{"phase":'
        )
        (self.root / "acceptance.json").write_text(json.dumps(dict(passed=False, node=0, error="phase failed")))
        result = subprocess.run([sys.executable, str(SUMMARY), str(self.root)], check=True, capture_output=True)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"], "phase failed")
        self.assertEqual(report["dispatches"]["orthrus"]["verify"], dict(
            dispatch=dict(calls=3, first_ms=800.0, subsequent_median_ms=200.0),
            native_chunk=dict(calls=3, first_ms=400.0, subsequent_median_ms=100.0),
        ))
        self.assertEqual(report["node_local_counters"]["verify_backup_d2d_bytes"], 384)

    def test_verifier_summary_reports_first_substage_tensor_difference(self):
        session = self.root / "native" / "session-1"
        session.mkdir(parents=True)
        (session / "run-manifest.json").write_text(json.dumps(dict(
            ranks=[dict(rank=0, dp_rank=0)],
        )))
        owner = dict(request_id="target-request-0")
        (session / "serving-dispatches.jsonl").write_text(json.dumps(dict(
            phase="decode", owners=[owner], native_epoch=0, seconds=1,
            timings=dict(chunk_seconds=1),
        )) + "\n")
        (self.root / "target").mkdir(exist_ok=True)
        (self.root / "target/result.json").write_text(json.dumps(dict(requests=[dict(
            index=0, dp=0, prompt_ids=[10], token_ids=[2, 3],
        )])))
        divergence = dict(
            request=0, generated_index=1, phase="verify/accept", target_token=2, orthrus_token=3,
            native=dict(verify=dict(epoch=0, tokens=[3])),
        )
        (self.root / "acceptance.json").write_text(json.dumps(dict(passed=False, divergence=divergence)))

        stages = [
            dict(stage="embedding/hidden", rank=0, case="fresh", layer=None, role=None, row=0,
                 shape=[1, 2], dtype="torch.bfloat16", sha256="same", max_abs=1, mean=1),
            dict(stage="layer-0/query", rank=0, case="fresh", layer=0, role="swa", row=0,
                 shape=[1, 2], dtype="torch.bfloat16", sha256="target", max_abs=2, mean=1.5),
        ]
        for controls, forward, token, trace_hash, tensor in (
            ("controls-0", "decode/step-1", 2, "target", torch.tensor([[1, 2]], dtype=torch.bfloat16)),
            ("controls-verify-1", "verify-1", 3, "verify", torch.tensor([[1, 4]], dtype=torch.bfloat16)),
        ):
            control = session / controls
            output = session / forward
            logits_dir = output / "actual-layer39-output"
            control.mkdir(parents=True)
            logits_dir.mkdir(parents=True)
            (control / "input-contract.json").write_text(json.dumps(dict(DP_engines=[dict(requests=[dict(
                position_ids=[1], history_tokens=[10], token_ids=[2],
            )])])))
            logits = torch.zeros(1, 8, dtype=torch.bfloat16)
            logits[0, token] = 2
            torch.save(dict(logits=logits, positions=torch.tensor([1]), active=1),
                       logits_dir / "rank-0-fresh-network-logits.pt")
            actual_stages = [dict(row) for row in stages]
            actual_stages[-1]["sha256"] = trace_hash
            (output / "verifier-activations.json").write_text(json.dumps(dict(
                position=1, stages=actual_stages, tensors="verifier-tensors.pt",
            )))
            torch.save({"embedding/hidden/rank-0/case-fresh": torch.ones(1, 2, dtype=torch.bfloat16),
                        "layer-0/query/rank-0/case-fresh": tensor}, output / "verifier-tensors.pt")

        result = subprocess.run([sys.executable, str(SUMMARY), str(self.root), "--verifier-logits"],
                                check=True, capture_output=True)
        report = json.loads(result.stdout)
        first = report["activation_trace"]["first_difference"]
        self.assertEqual((first["stage"], first["layer"], first["role"]), ("layer-0/query", 0, "swa"))
        self.assertEqual(first["difference"]["mismatched"], 1)
        self.assertEqual(first["difference"]["max_abs"], 2.0)

    def test_failure_diagnosis_reports_prefill_substage_without_another_run(self):
        native = dict(
            target=[dict(ids=[2], logits=[4.0])],
            orthrus=[dict(ids=[3], logits=[4.5])],
            trace_position=9,
            first_activation_difference=dict(
                chunk=0,
                target=dict(stage="layer-0/query", rank=0, case="fresh"),
                orthrus=dict(stage="layer-0/query", rank=0, case="fresh"),
                tensor_difference=dict(mismatched=1, elements=2, max_abs=0.5),
            ),
        )
        divergence = dict(request=0, generated_index=0, phase="prefill",
                          target_token=2, orthrus_token=3, native=native)
        (self.root / "acceptance.json").write_text(json.dumps(dict(
            passed=False, node=0, divergence=divergence,
        )))
        result = subprocess.run([sys.executable, str(SUMMARY), str(self.root), "--failure-diagnosis"],
                                check=True, capture_output=True)
        report = json.loads(result.stdout)
        self.assertEqual((report["phase"], report["trace_position"]), ("prefill", 9))
        first = report["first_activation_difference"]
        self.assertEqual(first["target"]["stage"], "layer-0/query")
        self.assertEqual(first["tensor_difference"]["max_abs"], 0.5)

    def test_prefill_reduction_reconstructs_identical_tp_contributors(self):
        session = self.root / "native" / "session-reduction"
        session.mkdir(parents=True)
        (session / "run-manifest.json").write_text(json.dumps(dict(topology=dict(tp=2))))
        rows = []
        for mode, output in (("target", torch.tensor([[3, 7]], dtype=torch.bfloat16)),
                             ("orthrus", torch.tensor([[4, 6]], dtype=torch.bfloat16))):
            directory = session / mode
            directory.mkdir()
            tensors = {}
            for rank, value in enumerate((torch.tensor([[1, 2]], dtype=torch.bfloat16),
                                          torch.tensor([[3, 4]], dtype=torch.bfloat16))):
                tensors[f"layer-0/wo-b/rank-{rank}/case-fresh"] = value
            tensors["layer-0/attention-scattered/rank-1/case-fresh"] = output
            torch.save(tensors, directory / "prefill-tensors.pt")
            trace = directory / "prefill-activations.json"
            trace.write_text(json.dumps(dict(position=9, stages=[], tensors="prefill-tensors.pt")))
            rows.append(dict(phase="prefill", owners=[dict(request_id=f"{mode}-request-0")],
                             prefill_activation_trace=str(trace), seconds=1,
                             timings=dict(chunk_seconds=1)))
        (session / "serving-dispatches.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        stage = dict(stage="layer-0/attention-scattered", rank=1, case="fresh", layer=0)
        divergence = dict(request=0, generated_index=0, phase="prefill", target_token=2,
                          orthrus_token=3, native=dict(first_activation_difference=dict(target=stage)))
        (self.root / "acceptance.json").write_text(json.dumps(dict(
            passed=False, node=0, divergence=divergence,
        )))
        result = subprocess.run([sys.executable, str(SUMMARY), str(self.root), "--prefill-reduction"],
                                check=True, capture_output=True)
        report = json.loads(result.stdout)
        self.assertTrue(report["fixed_order_reference"]["byte_exact"])
        self.assertFalse(report["observed_output"]["byte_exact"])
        self.assertTrue(all(item["byte_exact"] for item in report["contributor_differences"].values()))


if __name__ == "__main__":
    unittest.main()


def test_gsm8k_revision_comparison_recovers_old_phase_receipts_and_requires_matching_tokens(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SUMMARY.parent))
    compare = runpy.run_path(str(SUMMARY))["compare_gsm8k_reports"]
    dirs = [tmp_path / name for name in ("baseline", "candidate")]
    arm_names = ("ar", "initial", "online", "frozen")
    for index, directory in enumerate(dirs):
        path = directory / "native" / "session-test"
        path.mkdir(parents=True)
        writes = {
            "acceptance.json": dict(
                node=0, build_id=str(index), passed=True, topology={"dp": 4, "tp": 4, "ep": 16},
                physical_to_runtime={"0": 0}, checkpoint={"config.json": "same"}, head_sha256="same",
                peer={"physical_to_runtime": {"0": 0}},
            ),
            "gsm8k-online-report.json": dict(
                complete=True, training_requests=1, heldout_requests=1,
                max_concurrent_requests=1, max_new_tokens=2, updates_policy={"last_layers": 1},
                arms={name: dict(
                    wall_tok_s=1 + index, decode_tok_s=1 + index, accepted_per_round=None, acceptance_rate=None,
                    updates=1, publications=1, update_seconds=0.5, publication_seconds=0.1,
                    first_update_and_publish_seconds=0.6, wall_tok_s_excluding_all_updates_estimate=1.5,
                ) for name in arm_names},
            ),
        }
        for name in arm_names:
            writes[f"gsm8k-online-{name}.json"] = {
                "requests": [{"index": 0, "workload_id": "same", "dp": 0, "prompt_ids": [1], "token_ids": [2, 3]}]
            }
        for filename, data in writes.items():
            (directory / filename).write_text(json.dumps(data))
        receipts = [
            dict(phase="prefill", owners=[dict(request_id="target-request-ar_warmup-0")],
                 runtime={"output_d2h_bytes": 10}),
            dict(phase="draft", owners=[dict(request_id="orthrus-request-initial-0")],
                 student_logit_rows=7 if index else None,
                 runtime={"output_d2h_bytes": 20 - index}, timings={"chunk_seconds": 1.0 - 0.5 * index}),
            dict(phase="verify", owners=[dict(request_id="orthrus-request-initial-0")],
                 vocabulary_layout="device_greedy" if index else None,
                 teacher_logit_rows=2 if index else None,
                 timings={"chunk_seconds": .4, "verify_snapshot_seconds": .2, "verify_restore_seconds": .1}),
            dict(phase="accept", owners=[dict(request_id="orthrus-request-initial-0")],
                 commit_tokens_per_dp=[4, 1, 0, 0]),
            dict(phase="accept", owners=[dict(request_id="orthrus-request-initial-0")],
                 commit_tokens_per_dp=[0, 17, 0, 0]),
        ]
        (path / "serving-dispatches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in receipts))

    report = compare(*dirs)
    assert report["compared_greedy"] and report["arms"]["initial"]["wall_throughput_ratio"] == 2
    phases = report["arms"]["initial"]["local_serving"]
    assert [p["draft"]["output_d2h_bytes"] for p in phases] == [10, 9]
    assert [p["draft"]["native_seconds"] for p in phases] == [1.0, 0.5]
    assert phases[1]["draft"]["student_logit_rows"] == 7
    assert phases[1]["verify"]["verify_snapshot_seconds"] == .2
    assert phases[1]["verify"]["verify_restore_seconds"] == .1
    assert phases[1]["verify"]["device_greedy_calls"] == 1
    assert phases[1]["verify"]["teacher_logit_rows"] == 2
    assert phases[1]["accept"]["max_commit_tokens_per_dp"] == 17
    assert phases[1]["accept"]["accept_calls_fitting_capacity16"] == 1
    candidate = dirs[1] / "gsm8k-online-online.json"
    data = json.loads(candidate.read_text())
    data["requests"][0]["token_ids"][-1] = 4
    candidate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="diverged at generated token index 1"):
        compare(*dirs)


@pytest.mark.parametrize("stale_frozen_draft", [False, True])
def test_online_report_batches_and_freezes_on_heldout_questions(tmp_path, monkeypatch, stale_frozen_draft):
    monkeypatch.syspath_prepend(str(COMMAND.parent))
    monkeypatch.setitem(sys.modules, "pypto_serving.config.types", SimpleNamespace(
        DecodeBatch=SimpleNamespace, PrefillBatch=SimpleNamespace, SamplingParams=SimpleNamespace,
    ))
    study = runpy.run_path(str(COMMAND.with_name("dsv41_gsm8k_report.py")))["study_workload"]
    bank = SimpleNamespace(active_version=0)
    receipts = tmp_path / "serving-dispatches.jsonl"
    receipts.touch()
    session = SimpleNamespace(
        online=None, draft=SimpleNamespace(online_bank=bank), native=object(), devices=list(range(8)),
        short_speculative_dispatch=True,
        receipt_path=receipts, cluster=SimpleNamespace(plan=SimpleNamespace(node=0),
                                                      link=SimpleNamespace(exchange=lambda tag, payload: payload)),
    )
    seen = []
    records = []

    class Learner:
        config = SimpleNamespace(contract=lambda: {"objective": "tv", "last_layers": 1})
        bridge = SimpleNamespace(controller=SimpleNamespace(records=records))

        def bind(self, draft, native, cluster):
            assert seen[-1] == ("initial", [0, 1, 2, 3])
            assert (draft, native, cluster) == (session.draft, session.native, session.cluster)
            assert session.online is None and bank.active_version == 0

        def summary(self):
            return dict(updates=bank.active_version, publications=bank.active_version,
                        update_seconds=0.0, publication_seconds=0.0)

    online = Learner()

    class Runner:
        cache_plan = SimpleNamespace(groups=[])
        config = SimpleNamespace(chunk_tokens=2)
        ledger = SimpleNamespace(requests={})
        verify_candidate_limit = 7

        output_bytes = 0

        def run(self, batch, phase):
            self.output_bytes += 32
            with receipts.open("a") as stream:
                stream.write(json.dumps(dict(phase=phase, seconds=1.0, timings={"chunk_seconds": 0.5},
                                             runtime={"output_d2h_bytes": self.output_bytes})) + "\n")
            if phase == "prefill":
                tokens = batch.token_ids.tolist()
                assert sum(batch.chunk_lens) == len(tokens)
                for name, offset, length, start in zip(
                    batch.request_ids, batch.chunk_offsets, batch.chunk_lens, batch.chunk_starts, strict=True,
                ):
                    state = self.ledger.requests.setdefault(name, SimpleNamespace(history=[]))
                    assert start == len(state.history)
                    state.history.extend(tokens[offset:offset + length])
                return SimpleNamespace(sampled_token_ids=torch.ones(len(batch.request_ids), dtype=torch.long))
            for name, token in zip(batch.request_ids, batch.token_ids.tolist(), strict=True):
                self.ledger.requests[name].history.append(token)
            return SimpleNamespace(accepted_token_ids=None, sampled_token_ids=batch.token_ids + 1)

        def run_orthrus(self, batch):
            phase = batch.request_ids[0].split("-")[2]
            with receipts.open("a") as stream:
                version = 0 if stale_frozen_draft and phase == "frozen" else bank.active_version
                stream.write(json.dumps(dict(phase="draft", draft_head_version=version)) + "\n")
                if phase == "online":
                    stream.write(json.dumps(dict(phase="verify", teacher_logit_rows=2)) + "\n")
            seen.append((phase, batch.cache_partitions))
            if phase in ("draft_warmup", "initial"):
                assert session.online is None and bank.active_version == 0
            elif phase == "online":
                assert session.online is online
                bank.active_version += 1
                records.append(dict(update_seconds=0.0, publication_seconds=0.0))
            elif phase == "frozen":
                assert session.online is None and bank.active_version > 0
            self.run(batch, "decode")
            return SimpleNamespace(accepted_token_ids=[[token + 1] for token in batch.token_ids.tolist()],
                                   num_draft_tokens=[31] * len(batch.request_ids))

        def release_finished_requests(self, names):
            for name in names:
                del self.ledger.requests[name]

    runner = Runner()
    prompts = [[100 + i] * (i % 4 + 1) for i in range(12)]
    entries = [dict(id=f"gsm8k-{i}", reference_answer="#### 2") for i in range(len(prompts))]
    options = dict(train_count=8, steps=4, batch_size=4, eos_ids={3},
                   decoder=lambda tokens: "#### 2", output=tmp_path, engram_seconds=5.0)
    if stale_frozen_draft:
        with pytest.raises(ValueError, match="frozen executed draft versions"):
            study(runner, session, online, prompts, entries, **options)
        assert not json.loads((tmp_path / "gsm8k-online-report.json").read_text())["complete"]
        return
    report = study(runner, session, online, prompts, entries, **options)
    assert report["passed"] and report["compared_tokens"] == {"initial": 12, "frozen": 12}
    assert report["verify_candidate_limit"] == 7 and report["short_speculative_dispatch"] is True
    assert report["startup_compared_tokens"] == 8
    assert report["arms"]["ar"]["draft_head_versions"] == []
    assert report["arms"]["initial"]["draft_head_versions"] == [0]
    assert report["arms"]["frozen"]["draft_head_versions"] == [report["arms"]["online"]["publications"]]
    assert report["arms"]["online"]["requests"] == 8
    assert report["arms"]["frozen"]["requests"] == 4
    assert report["arms"]["frozen"]["stopped"] == 4
    prefill = report["arms"]["ar"]["serving_phases_by_host"]["0"]["prefill"]
    assert prefill["calls"] > 0
    assert prefill["output_d2h_bytes"] == 32 * prefill["calls"]
    assert prefill["native_seconds"] == .5 * prefill["calls"]
    verify = report["arms"]["online"]["serving_phases_by_host"]["0"]["verify"]
    assert verify["teacher_logit_rows"] == 2 * verify["calls"]
    assert report["arms"]["frozen"]["head_version"] == report["arms"]["online"]["publications"]
    assert report["arms"]["online"]["wall_tok_s_excluding_all_updates_estimate"] > 0
    assert all(partitions == [0, 1, 2, 3] for _, partitions in seen)
    train = json.loads((tmp_path / "gsm8k-online-online.json").read_text())
    eval_ = json.loads((tmp_path / "gsm8k-online-frozen.json").read_text())
    assert {row["workload_id"] for row in train["requests"]}.isdisjoint(row["workload_id"] for row in eval_["requests"])
    assert not runner.ledger.requests
