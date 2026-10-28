import importlib.util
from pathlib import Path

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
