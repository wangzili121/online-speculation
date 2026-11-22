"""短测试加密只读资源采集；不派发模型，也不进行性能计时。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    q = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("resource_capture346", q.parent / "resource338/capture_resources.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    spec = importlib.util.spec_from_file_location("atomic346", q.parent / "sync344/collect.py")
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    launch = json.loads((q / "launch-A.json").read_text())
    case = json.loads((q / "queue-A.json").read_text())["cases"][0]
    output = q / (case["tag"] + ".resources.json")
    payload = dict(host="159.138.5.111", root=launch["root"], tag=case["tag"], launcher=case["launcher"])
    while not output.exists():
        try:
            call = subprocess.run(
                ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=10", "wangzili@159.138.5.111", "python3 -"],
                input=("P=" + repr(payload) + "\n" + capture.REMOTE).encode(),
                capture_output=True, timeout=40, check=True,
            )
            observation = json.loads(call.stdout)
        except (subprocess.SubprocessError, ValueError) as error:
            observation = dict(observation_error=repr(error), terminal=False)
        with (q / "resource-observations.jsonl").open("a") as history:
            history.write(json.dumps(observation) + "\n")
        if observation.get("captured"):
            storage.write_json(output, observation)
            print(json.dumps(observation), flush=True)
            return
        if observation.get("completed") or observation.get("error"):
            print(json.dumps(dict(resource_capture_missing=True, model_not_restarted=True,
                                  last_observation=observation)), flush=True)
            return
        time.sleep(4 if observation.get("stage") else 30)


if __name__ == "__main__":
    main()
