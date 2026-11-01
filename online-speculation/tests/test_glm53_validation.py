import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def controller(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("validate_glm53", scripts / "validate_glm53_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("direction", ["causal", "bidir"])
def test_sampling_checks_the_matching_head_export(controller, tmp_path, direction):
    original = "orthrus" if direction == "causal" else "orthrus-bidir"
    digest = [dict(rank=0, sha256=direction)]
    (tmp_path / f"{original}.json").write_text(json.dumps(dict(head_exports=digest)))
    loaded = tmp_path / f"sampling-{direction}.json"
    loaded.write_text(json.dumps(dict(head_digests=digest)))
    controller.check_reloaded(tmp_path, f"sampling-{direction}")
    loaded.write_text(json.dumps(dict(head_digests=[dict(rank=0, sha256="wrong")])))
    with pytest.raises(RuntimeError, match="differ"):
        controller.check_reloaded(tmp_path, f"sampling-{direction}")


def test_sampling_stages_load_each_directions_own_head(controller):
    stages = {label: extra for _, label, extra in controller.stages()}
    for direction in ("causal", "bidir"):
        extra = stages[f"sampling-{direction}"]
        assert extra[extra.index("--head-dir") + 1] == f"/results/head-{direction}"
        assert extra[extra.index("--temperature") + 1] == "0.7"


def test_controller_passes_block_width_to_native_job(controller, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["validate_glm53_native", "--source", str(tmp_path),
        "--model", str(tmp_path), "--output", str(tmp_path), "--image", "runtime@sha256:123",
        "--devices", "2", "--block-size", "4", "--stages", "baseline"])
    monkeypatch.setattr(controller.subprocess, "check_output", lambda *a, **kw: "snapshot")
    monkeypatch.setattr(controller, "idle_devices", lambda snapshot: [2])
    monkeypatch.setattr(controller, "runtime_ready", lambda *args: True)

    def run(args, mode, *, label, extra):
        calls.append(extra)
        (tmp_path / f"{label}.json").write_text('{"status":"passed"}')

    monkeypatch.setattr(controller, "run_job", run)
    controller.main()
    assert calls[0][calls[0].index("--block-size") + 1] == "4"
    assert json.loads((tmp_path / "native-status.json").read_text())["state"] == "passed"


def test_replica_verifier_checks_manifest_and_file_bytes(controller, tmp_path):
    import verify_glm53_snapshot as verify

    content = b"checkpoint"
    (tmp_path / "weights").write_bytes(content)
    item = dict(path="weights", size=len(content), sha256=hashlib.sha256(content).hexdigest())
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(dict(model_id="Eco-Tech/GLM-5.3-Flash-w8a8", files=[item])))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify.load_manifest(path, digest)["files"] == [item]
    verify.verify_file(tmp_path, item)
    with pytest.raises(ValueError, match="manifest SHA256"):
        verify.load_manifest(path, "wrong")
    (tmp_path / "weights").write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify.verify_file(tmp_path, item)
    item["path"] = "../weights"
    path.write_text(json.dumps(dict(model_id="Eco-Tech/GLM-5.3-Flash-w8a8", files=[item])))
    with pytest.raises(ValueError, match="manifest entry"):
        verify.load_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_native_preflight_revokes_ready_before_importing_extensions(controller, tmp_path, monkeypatch):
    import builtins
    import preflight_glm53_runtime as preflight

    path = tmp_path / "runtime-ready.json"
    path.write_text('{"status":"passed"}')
    monkeypatch.setattr(sys, "argv", ["preflight_glm53_runtime", "--output", str(path),
                                     "--image", "runtime@sha256:123"])
    original = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "vllm_ascend.ops":
            assert json.loads(path.read_text())["status"] == "failed"
            raise SystemExit("native extension aborted")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(SystemExit, match="extension aborted"):
        preflight.main()
    assert json.loads(path.read_text())["status"] == "failed"


def test_eight_card_preflight_never_uses_a_partial_window(controller):
    import prepare_glm53_runtime as prepare

    assert prepare.candidate_groups([2, 3, 6, 7], all_devices=True) == []
    assert prepare.candidate_groups(list(range(8)), all_devices=True) == [list(range(8))]
    assert prepare.candidate_groups([2, 3, 4], allowed=[3]) == [[3]]
