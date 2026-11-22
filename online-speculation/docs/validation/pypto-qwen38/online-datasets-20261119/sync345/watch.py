"""监督功能门槛与固定旧题探索；不重试失败模型，不自动启动确认集。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time
import traceback


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    q = Path(__file__).resolve().parent
    collect = module(q.parent / "sync344/collect.py", "collect_sync345")
    delivery = module(q / "dispatch.py", "dispatch_sync345")
    finished = {}
    while len(finished) < 2:
        def one(host):
            if host in finished:
                return finished[host]
            try:
                protocol = json.loads((q / "protocol.json").read_text())
                gate_suite = protocol.get("gate_suites", {}).get(host, "sync344")
                gate = collect.collect(q.parent, host, suite=gate_suite)
                if not gate.get("terminal"):
                    return dict(host=host, stage="waiting_functional_gate", gate=gate)
                if not gate.get("completed") or not gate.get("resources_checked"):
                    out = dict(host=host, terminal=True, completed=False,
                               stage="functional_gate_needs_inspection", gate=gate)
                    finished[host] = out
                    return out
                launch_path = q / f"launch-{host}.json"
                if not launch_path.exists():
                    # 派发失败时停止自动推进，避免不确定状态下重复启动。
                    try:
                        launch = delivery.dispatch(host)
                    except Exception as error:
                        out = dict(host=host, terminal=True, completed=False,
                                   stage="dispatch_needs_inspection", error=repr(error),
                                   traceback=traceback.format_exc(), NN_automatically_retried=False)
                        finished[host] = out
                        return out
                    print(json.dumps(dict(stage="exploration_dispatched", launch=launch)), flush=True)
                out = collect.collect(q.parent, host, suite=q.name)
                if out.get("terminal"):
                    finished[host] = out
                    print(json.dumps(out), flush=True)
                return out
            except Exception as error:
                return dict(host=host, observation_error=repr(error), terminal=False,
                            NN_automatically_retried=False)

        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(one, ("A", "B")))
        state = dict(observed_UTC=datetime.now(timezone.utc).isoformat(),
                     completed=len(finished) == 2, results=rows,
                     all_qualified=len(finished) == 2 and all(v.get("performance_qualification") for v in rows),
                     fresh_confirmation_dispatched=False, NN_automatically_retried=False)
        tmp = q / "watch-state.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(q / "watch-state.json")
        if len(finished) < 2:
            time.sleep(30)


if __name__ == "__main__":
    main()
