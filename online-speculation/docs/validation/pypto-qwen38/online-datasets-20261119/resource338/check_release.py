"""只读核对已完成用例的原进程和容器退出；不停止或释放任何任务。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess


REMOTE = r'''
from pathlib import Path
import json,subprocess
instances=[]
for old in P['process_instances']:
 path=Path('/proc')/str(old['pid'])/'stat'
 try:
  ticks=int(path.read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:
  ticks=None
 instances.append(dict(pid=old['pid'],recorded_start_ticks=old['start_ticks'],current_start_ticks=ticks,original_instance_live=ticks==old['start_ticks']))
container=P.get('container')
container_exists=None
if container:
 ids=subprocess.check_output(['docker','ps','-aq','--no-trunc'],text=True).splitlines()
 container_exists=container['id']in ids
print(json.dumps(dict(original_processes_gone=all(not x['original_instance_live']for x in instances),original_container_gone=not container_exists if container else None,instances=instances)))
'''


def main(root):
    q = root / "resource338"
    bindings = []
    for path in q.glob("*.resources.json"):
        tag = path.name.removesuffix(".resources.json")
        directory = root / "aux333" if "spoc333" in tag else q
        proof = directory / (tag + ".audit.json")
        if proof.exists() and json.loads(proof.read_text()).get("completed"):
            bindings.append((tag, json.loads(path.read_text())))

    def check(item):
        tag, binding = item
        try:
            call = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no",
                                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                   "wangzili@" + binding["host"], "python3 -"],
                                  input=("P=" + repr(binding) + "\n" + REMOTE).encode(),
                                  capture_output=True, timeout=40, check=True)
            proof = json.loads(call.stdout)
            proof.update(tag=tag, host=binding["host"],
                         released=proof["original_processes_gone"] and proof["original_container_gone"] is not False,
                         globally_idle_cards_not_asserted=True, actions_taken=[])
            return proof
        except (subprocess.SubprocessError, ValueError) as error:
            return dict(tag=tag, observation_error=repr(error), released=None, actions_taken=[])

    with ThreadPoolExecutor(max_workers=3) as pool:
        cases = list(pool.map(check, bindings))
    result = dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases,
                  read_only=True, all_observed_original_instances_released=bool(cases) and all(x.get("released") is True for x in cases),
                  scope="Only original recorded process instances and containers. Cards may be reused by later own or other tasks.")
    (q / "original-instance-release.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": [{"tag": x["tag"], "released": x.get("released"),
                                 "error": x.get("observation_error")} for x in cases]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    main(parser.parse_args().validation_root)
