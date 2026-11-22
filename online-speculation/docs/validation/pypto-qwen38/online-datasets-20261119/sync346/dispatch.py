"""补采111的短功能测试资源证据，不重启任何在途模型。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    q = Path(__file__).resolve().parent
    parent = q.parent
    previous = json.loads((parent / "sync344/launch-A.json").read_text())
    proof = json.loads((parent / "sync344" / (previous["tag"] + ".audit.json")).read_text())
    assert proof["completed"] and proof["functional_only"] and not proof["resources_checked"]
    protocol = json.loads((q / "protocol.json").read_text())
    raw = (q / "queue-A.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol["files_sha256"]["queue-A.json"]
    plan = json.loads(raw)
    old = json.loads((parent / "sync344/queue-A.json").read_text())["cases"][0]
    def numeric_args(args):
        value = list(args)
        value[value.index("--devices") + 1] = "<physical device candidates>"
        return value
    assert numeric_args(plan["cases"][0]["args"]) == numeric_args(old["args"])
    assets = {n: (q / n).read_text() for n in protocol["asset_names"]}
    assert all(hashlib.sha256(text.encode()).hexdigest() == protocol["files_sha256"][n]
               for n, text in assets.items())
    spec = importlib.util.spec_from_file_location("delivery346", parent / "sync344/dispatch.py")
    delivery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(delivery)
    payload = dict(suite=q.name, host="A", root=previous["root"], assets=assets, plan=plan,
                   registered_UTC=protocol["registered_UTC"], controller_sha256=previous["controller_sha256"])
    result = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "wangzili@159.138.5.111", "python3 -"],
        input=("P=" + repr(payload) + "\n" + delivery.REMOTE).encode(),
        capture_output=True, timeout=60, check=True,
    )
    out = json.loads(result.stdout)
    assert out["plan_sha256"] == protocol["files_sha256"]["queue-A.json"]
    assert out["tag"] == plan["cases"][0]["tag"]
    (q / "launch-A.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
