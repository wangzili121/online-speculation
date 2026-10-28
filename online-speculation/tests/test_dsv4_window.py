import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("window", Path(__file__).parents[1] / "scripts/watch_dsv4_tp8.py")
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


def free_snapshot():
    return "\n".join(f"| {i} 910B3 | OK |\n| 0 | 0 | 3420 / 65536 |\n"
                     f"| No running processes found in NPU {i} |" for i in range(8))


def test_only_complete_idle_snapshot_triggers():
    snapshot = free_snapshot()
    assert window.idle_devices(snapshot)
    for invalid in ("", "permission denied", snapshot.replace("NPU 7", "NPU 6"),
                    snapshot.replace("3420 / 65536", "47000 / 65536", 1),
                    snapshot + "\n| 0 0 | 12345 | VLLMWorker |",
                    snapshot.replace("| 7 910B3 | OK |", "")):
        assert not window.idle_devices(invalid)


def test_container_with_other_work_is_not_restarted():
    assert window.container_is_idle("PID COMMAND\n123 sleep infinity\n")
    assert not window.container_is_idle("PID COMMAND\n")
    assert not window.container_is_idle("PID COMMAND\n123 sleep infinity\n456 python train.py\n")


def make_window(tmp_path):
    return window.Window(SimpleNamespace(container="wangzili-orthrus-test", root="/workspace",
                                         watch_hours=12, output=tmp_path))


def test_cleanup_does_not_wait_for_other_users_devices(tmp_path, monkeypatch):
    runner = make_window(tmp_path)
    calls = []
    monkeypatch.setattr(window.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(runner, "wait_free", lambda *a: pytest.fail("cleanup must not wait for free cards"))
    process = SimpleNamespace(wait=lambda **k: calls.append("waited"))
    runner.finish_service(process, failed=True)
    assert calls == [["docker", "restart", "-t", "20", "wangzili-orthrus-test"], "waited"]


@pytest.mark.parametrize("failed", [True, False])
def test_cleanup_failure_preserves_service_failure(tmp_path, monkeypatch, failed):
    runner = make_window(tmp_path)

    def restart_error():
        raise RuntimeError("cleanup failed")
    monkeypatch.setattr(runner, "restart", restart_error)
    if failed:
        runner.finish_service(None, failed=True)
    else:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            runner.finish_service(None, failed=False)
    assert "cleanup failed" in runner.report["cleanup_error"]
