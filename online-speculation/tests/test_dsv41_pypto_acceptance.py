"""Exercise the offline acceptance gate without loading a model."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


COMMAND = Path(__file__).resolve().parents[1] / "scripts/verify_dsv41_pypto_acceptance.py"
SUMMARY = COMMAND.with_name("summarize_dsv41_pypto_cluster.py")


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


if __name__ == "__main__":
    unittest.main()
