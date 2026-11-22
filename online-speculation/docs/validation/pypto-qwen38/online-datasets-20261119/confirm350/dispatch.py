"""只派发已经完整冻结的新题三次确认；重复调用不重复启动模型。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from common import module, qualified, read, sha


def dispatch():
    q = Path(__file__).resolve().parent
    root = q.parent
    frozen = read(q / "frozen-recipe.json")
    assert sha((q / "registration.json").read_bytes()) == frozen["registration_sha256"]
    assert sha((q / frozen["freshness_report"]).read_bytes()) == frozen["freshness_sha256"]
    assert all(sha((q / name).read_bytes()) == value for name, value in frozen["files_sha256"].items())
    qualification = qualified(root)
    assert qualification == frozen["qualification"]
    registration = read(q / "registration.json")
    plan = read(q / "queue-B.json")
    assets = {path.name: path.read_text() for path in (q / "assets").glob("*.json")}
    protocol = read(q / "protocol.json")
    assert all(sha(text.encode()) == protocol["files_sha256"][name] for name, text in assets.items())
    old = read(root / "sync345/launch-B.json")
    assignment = read(q / "assignment.json") | dict(
        schema="orthrus-reserved-requests-v1", assigned_input_sha256=protocol["files_sha256"],
        frozen_recipe_sha256=sha((q / "frozen-recipe.json").read_bytes()))
    payload = dict(suite=q.name, host="B", root=old["root"], assets=assets, plan=plan,
                   registered_UTC=registration["registered_UTC"], controller_sha256=old["controller_sha256"],
                   fresh_reserved_questions_used=True, assignment=assignment)
    delivery = module(root / "sync344/dispatch.py", "delivery350")
    # 显式留下分配记录；即使控制器以后退出，使用记录审计也会排除这些输入。
    prefix = r'''
from pathlib import Path
import json
r=Path(P['root']);path=r/'confirm350-assignment.json'
if path.exists():assert json.loads(path.read_text())==P['assignment']
else:
 with path.open('x')as handle:json.dump(P['assignment'],handle,indent=2);handle.write('\n')
'''
    call = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "wangzili@189.1.227.167", "python3 -"],
        input=("P=" + repr(payload) + "\n" + prefix + "\n" + delivery.REMOTE).encode(),
        capture_output=True, timeout=60,
    )
    if call.returncode:
        raise RuntimeError(call.stderr.decode()[-3000:])
    out = json.loads(call.stdout)
    assert out["plan_sha256"] == sha((q / "queue-B.json").read_bytes())
    assert out["fresh_reserved_questions_used"] is True
    out.update(tags=[c["tag"] for c in plan["cases"]],
               frozen_recipe_sha256=sha((q / "frozen-recipe.json").read_bytes()),
               recorded_locally_UTC=datetime.now(timezone.utc).isoformat())
    (q / "launch-B.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(dispatch()))
