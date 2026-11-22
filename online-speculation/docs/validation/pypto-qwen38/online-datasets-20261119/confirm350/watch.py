"""资格完整通过才冻结并派发新题确认；观察超时不重启模型。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import traceback

from common import module, read
from dispatch import dispatch
from prepare import prepare


PROBE = r'''
from pathlib import Path
import json
r=Path(P['root']);p=Path('/proc')/str(P['pid'])/'cmdline';cmd=p.read_bytes().replace(b'\0',b' ').decode()if p.exists()else''
f=r/(P['tag']+'.json');d=json.loads(f.read_text())if f.exists()else{}
print(json.dumps(dict(controller_pid=P['pid'],controller_live=P['plan']in cmd,cmdline=cmd,
                     completed=d.get('completed'),stage=d.get('stage'),error=d.get('error'))))
'''


def main():
    q = Path(__file__).resolve().parent
    root = q.parent
    collector = module(root / "sync344/collect.py", "collect350")
    registration = read(q / "registration.json")
    qualifier_tag = registration["qualification_tag"]
    qualifier_proof = root / "sync345" / (qualifier_tag + ".audit.json")

    def save(stage, **fields):
        collector.write_json(q / "watch-state.json", dict(observed_UTC=datetime.now(timezone.utc).isoformat(),
                             stage=stage, goal_complete=False, NN_automatically_retried=False, **fields))

    while not qualifier_proof.exists():
        try:
            old = read(root / "sync345/launch-B.json")
            call = subprocess.run(
                ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=10", "wangzili@189.1.227.167", "python3 -"],
                input=("P=" + repr(old) + "\n" + PROBE).encode(), capture_output=True, timeout=40,
            )
            if call.returncode:
                raise RuntimeError(call.stderr.decode()[-2000:])
            observation = json.loads(call.stdout)
            save("waiting_complete_qualification", observation=observation, fresh_cohort_assigned=False)
            if observation["completed"] or observation["error"] or not observation["controller_live"]:
                # 收集已有终态证据，不派发或重启旧模型。
                collector.collect(root, "B", suite="sync345")
        except (subprocess.SubprocessError, ValueError, RuntimeError) as error:
            save("qualification_observation_pending", observation_error=repr(error), terminal=False,
                 fresh_cohort_assigned=False)
        if not qualifier_proof.exists():
            time.sleep(30)
    proof = read(qualifier_proof)
    if not proof.get("completed") or not proof.get("performance_qualification"):
        save("qualification_did_not_pass", terminal=True, fresh_cohort_assigned=False, proof=proof)
        return
    try:
        frozen = prepare()
        save("frozen_ready", terminal=False, fresh_cohort_assigned=True,
             already_dispatched=(q / "launch-B.json").exists(), frozen_UTC=frozen["frozen_UTC"])
        launch = read(q / "launch-B.json") if (q / "launch-B.json").exists() else dispatch()
    except Exception as error:
        save("preparation_or_dispatch_needs_inspection", terminal=True, error=repr(error),
             traceback=traceback.format_exc(), fresh_cohort_assigned=(q / "frozen-recipe.json").exists())
        return
    cases = read(q / "queue-B.json")["cases"]
    assert len(cases) == 3
    complete = {}
    while len(complete) < 3:
        observations = []
        for case in cases:
            tag = case["tag"]
            if tag in complete:
                continue
            try:
                result = collector.collect(root, "B", suite=q.name, case_tag=tag)
                if result.get("terminal"):
                    complete[tag] = result
                    print(json.dumps(result), flush=True)
                observations.append(result)
                if not result.get("terminal"):
                    break  # 单一控制器顺序执行，不轮询尚未放行的后续模型。
            except Exception as error:
                observations.append(dict(tag=tag, observation_error=repr(error), terminal=False,
                                         NN_restarted=False))
        save("observing_three_frozen_confirmations", terminal=False, controller=launch,
             completed_cases=list(complete.values()), observations=observations,
             fresh_cohort_assigned=True)
        if len(complete) < 3:
            time.sleep(30)
    from summarize import summarize
    summary = summarize(root)
    collector.write_json(q / "three-confirmation-audit.json", summary)
    save("three_confirmations_terminal", terminal=True, results=summary, fresh_cohort_assigned=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
