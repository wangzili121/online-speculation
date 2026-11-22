"""用临时目录检查使用记录审计的保守边界；不连接服务器或模型。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import zlib


def main():
    q = Path(__file__).resolve().parent
    source = q / "audit_usage.py"
    spec = importlib.util.spec_from_file_location("usage349", source)
    usage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(usage)
    compile(usage.REMOTE, "remote scanner", "exec")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = base / "online-test"
        assets = root / "assets"
        assets.mkdir(parents=True)

        def save(path, value):
            path.write_text(json.dumps(value) + "\n")
            return hashlib.sha256(path.read_bytes()).hexdigest()

        used = save(assets / "used.json", ["Question: Used question?  Explain."])
        unlaunched_sha = save(assets / "unlaunched.json", ["Never seen question?"])
        save(root / "done.json", dict(stage="completed", prompts_sha256=used))
        plan = root / "queue-later.json"
        save(plan, dict(cases=[dict(tag="not-launched", args=["--request-file", "unlaunched.json"])]))
        candidates = [dict(index=i, normalized=text,
                           sha256=hashlib.sha256(text.encode()).hexdigest(),
                           normalized_sha256=hashlib.sha256(text.encode()).hexdigest())
                      for i, text in [(1, "Used question?"), (2, "Never seen question?")]]

        def run(active=()):
            payload = dict(base=str(base), required_roots=["online-test"], candidates=candidates,
                           test_mode=True, active_plans=list(active))
            call = subprocess.run(["python3", "-c", "P=" + repr(payload) + "\n" + usage.REMOTE],
                                  capture_output=True, check=True)
            return json.loads(zlib.decompress(call.stdout))

        result = run()
        assert result["completed"] and {v["index"] for v in result["candidate_substring_hits"]} == {1}
        save(root / "assignment.json", dict(schema="orthrus-reserved-requests-v1",
                                             assigned_input_sha256={"unlaunched.json": unlaunched_sha}))
        result = run()
        assert result["completed"] and {v["index"] for v in result["candidate_substring_hits"]} == {1, 2}
        (root / "assignment.json").unlink()
        result = run([str(plan)])
        assert result["completed"] and {v["index"] for v in result["candidate_substring_hits"]} == {1, 2}
        save(root / "unknown.json", dict(stage="starting", prompts_sha256="f" * 64))
        result = run()
        assert not result["completed"] and result["unknown_assets"]
        (root / "unknown.json").unlink()
        save(assets / "nontext.json", dict(tokens=[1]))
        save(root / "bad.source.json", dict(files={}))
        save(root / "queue-bad.json", dict(cases=[dict(tag="bad", args=["--request-file", "nontext.json"])]))
        result = run()
        assert not result["completed"] and result["unknown_assets"]
        (root / "queue-bad.json").unlink()
        (root / "corrupt.json").write_text("{")
        result = run()
        assert not result["completed"] and result["read_errors"]
    proof = dict(passed=True, wrapped_question_detected=True,
                 unlaunched_unassigned_assets_not_marked_used=True,
                 live_queued_assignment_excluded=True, unknown_input_invalidates_completeness=True,
                 frozen_assignment_without_running_controller_excluded=True,
                 nontext_referenced_asset_invalidates_completeness=True,
                 unreadable_json_invalidates_completeness=True, NPU_or_model_calls=False,
                 scanner_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (q / "selfcheck.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
