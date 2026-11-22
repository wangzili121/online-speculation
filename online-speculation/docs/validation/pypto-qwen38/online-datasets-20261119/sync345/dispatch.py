"""同步三卡功能审核通过后，派发固定的旧384题探索；不分配确认题。"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def dispatch(host):
    q = Path(__file__).resolve().parent
    root = q.parent
    gate_launch = json.loads((root / "sync344" / f"launch-{host}.json").read_text())
    gate = json.loads((root / "sync344" / (gate_launch["tag"] + ".audit.json")).read_text())
    assert gate["completed"] and gate["functional_only"] and gate["resources_checked"]
    assert gate["accounting_passed"] and gate["frozen_source_head_assets_base_checked"]
    assert len(gate["actual_published_shards"]) == 2 and gate["first_feedback_replays"]
    protocol = json.loads((q / "protocol.json").read_text())
    plan_raw = (q / f"queue-{host}.json").read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == protocol["plans"][host]
    assert not protocol["NN_code_changed"] and not protocol["fresh_reserved_questions_used"]
    assets = {n: (q / n).read_text() for n in protocol["asset_names"]}
    assert all(hashlib.sha256(v.encode()).hexdigest() == protocol["files_sha256"][n]
               for n, v in assets.items())
    payload = dict(suite=q.name, host=host, root=gate_launch["root"], assets=assets,
                   plan=json.loads(plan_raw), registered_UTC=protocol["registered_UTC"],
                   controller_sha256=gate_launch["controller_sha256"])
    delivery = module(root / "sync344/dispatch.py", "delivery_sync345")
    address = "159.138.5.111" if host == "A" else "189.1.227.167"
    call = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
        input=("P=" + repr(payload) + "\n" + delivery.REMOTE).encode(),
        capture_output=True, timeout=60,
    )
    if call.returncode:
        raise RuntimeError(call.stderr.decode()[-3000:])
    out = json.loads(call.stdout)
    assert out["plan_sha256"] == protocol["plans"][host]
    assert out["tag"] == payload["plan"]["cases"][0]["tag"]
    out["functional_gate_raw_sha256"] = gate["raw_sha256"]
    out["recorded_locally_UTC"] = datetime.now(timezone.utc).isoformat()
    (q / f"launch-{host}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("A", "B"), required=True)
    print(json.dumps(dispatch(parser.parse_args().host)))
