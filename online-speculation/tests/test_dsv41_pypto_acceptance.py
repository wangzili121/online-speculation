"""Exercise the offline acceptance gate without loading a model."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


COMMAND = Path(__file__).resolve().parents[1] / "scripts/verify_dsv41_pypto_acceptance.py"


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


if __name__ == "__main__":
    unittest.main()
