"""仅排队旧输入功能测试；等当前冻结用例完成后，使用现有同步交错路径。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


REMOTE = r'''
from pathlib import Path
import hashlib,json,subprocess,os,datetime
r=Path(P['root']);suite=P.get('suite','sync344');marker=r/(suite+'-'+P['host']+'-launch.json')
if marker.exists():
 print(marker.read_text());raise SystemExit(0)
sha=lambda b:hashlib.sha256(b).hexdigest()
controller=r/'study_queue_resource338.py'
assert sha(controller.read_bytes())==P['controller_sha256']
c=P['plan']['cases'][0];files=json.loads((r/c['expected_source_manifest']).read_text());files=files.get('files',files)
for name,expected in files.items():
 path=(r if len(Path(name).parts)==1 else r.parent)/name
 assert sha(path.read_bytes())==expected,name
head=r.parent/c['expected_initial_head']
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb')as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert [file_sha(head/('rank'+str(i)+'.safetensors'))for i in range(2)]==c['expected_initial_head_shards']
for name,text in P['assets'].items():
 path=r/'assets'/name;data=text.encode()
 if path.exists():assert path.read_bytes()==data
 else:path.write_bytes(data)
name='queue-'+suite+'-'+P['host']+'.json';path=r/name;data=(json.dumps(P['plan'],indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==data
else:path.write_bytes(data)
with(r/(name+'.controller.log')).open('xb')as log:
 child=subprocess.Popen(['python3',str(controller),'--plan',name],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
out=dict(host=P['host'],root=str(r),pid=child.pid,controller=controller.name,plan=name,tag=c['tag'],registered_UTC=P['registered_UTC'],dispatched_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_checked=len(files),controller_sha256=P['controller_sha256'],NN_code_changed=False,original_frozen_cases_changed=False,fresh_reserved_questions_used=P.get('fresh_reserved_questions_used',False),requires_original_complete=c['requires'],plan_sha256=sha(data))
marker.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


def main():
    q = Path(__file__).resolve().parent
    protocol = json.loads((q / "protocol.json").read_text())
    assets = {name: (q / name).read_text() for name in ("sync344-smoke6.json", "sync344-smoke-holdout2.json")}
    base = "/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/"

    def dispatch(host):
        plan = json.loads((q / f"queue-{host}.json").read_text())
        payload = dict(host=host, plan=plan, assets=assets, registered_UTC=protocol["registered_UTC"],
                       controller_sha256="303d9ae5fb1e53c4e7b67d271fbf4e0f42491c28d500cfe03b6cd9e4cfad0394",
                       root=base + ("online-studies" if host == "A" else "online-background"))
        address = "159.138.5.111" if host == "A" else "189.1.227.167"
        result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no",
                                 "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
                                input=("P=" + repr(payload) + "\n" + REMOTE).encode(), capture_output=True,
                                timeout=60)
        if result.returncode:
            error = dict(host=host, returncode=result.returncode, stderr=result.stderr.decode(),
                         stdout=result.stdout.decode())
            (q / f"dispatch-error-{host}.json").write_text(json.dumps(error, indent=2) + "\n")
            raise RuntimeError(json.dumps(error))
        value = json.loads(result.stdout)
        (q / f"launch-{host}.json").write_text(json.dumps(value, indent=2) + "\n")
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(dispatch, ("A", "B")):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
