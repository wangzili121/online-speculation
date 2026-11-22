"""每条完整流实机记录一次worker与物理NPU绑定；不占卡、不停止进程。"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import textwrap
import time


LIVE_REMOTE = r'''
from pathlib import Path
import json,os,subprocess,re,datetime
r=Path(P['root']);p=r/(P['tag']+'.json')
try:d=json.loads(p.read_text())if p.exists()else{}
except json.JSONDecodeError:d={}
out=dict(tag=P['tag'],observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),captured=False,
         stage=d.get('stage'),completed=d.get('completed',False),error=d.get('error'))
if d.get('stage')=='measuring'and not d.get('completed'):
 rows=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
 launchers=[int(x.split(None,1)[0])for x in rows if P['launcher']+' --tag '+P['tag']+' 'in x]
 out['matching_launcher_count']=len(launchers)
 if len(launchers)==1:
  launcher=launchers[0];processes=[];container=None
  if P['launcher'].endswith('_docker.py'):
   name='orthrus-online-111-'+P['tag'];x=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
   if x.returncode==0:
    value=json.loads(x.stdout)[0];container=dict(id=value['Id'],name=name,image=value['Image'])
    top=subprocess.check_output(['docker','top',name,'-eo','pid,ppid,args'],text=True)
    processes=[dict(pid=int(x.split(None,2)[0]),ppid=int(x.split(None,2)[1]))for x in top.splitlines()[1:]if len(x.split(None,2))==3]
  else:
   pairs=[tuple(map(int,line.split()))for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
   found={launcher}
   while True:
    extra={pid for pid,parent in pairs if parent in found}-found
    if not extra:break
    found.update(extra)
   processes=[dict(pid=pid,ppid=parent)for pid,parent in pairs if pid in found]
  for proc in processes:
   try:proc['start_ticks']=int((Path('/proc')/str(proc['pid'])/'stat').read_text().split(') ',1)[1].split()[19])
   except FileNotFoundError:proc['start_ticks']=None
  own={x['pid']for x in processes};matched=[]
  smi=subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)
  for line in smi.splitlines():
   cells=[s.strip()for s in line.strip().strip('|').split('|')]
   if len(cells)>=4 and re.fullmatch(r'\d+\s+\d+',cells[0])and cells[1].isdigit()and int(cells[1])in own:
    device,chip=map(int,cells[0].split());matched.append(dict(physical_npu=device,chip=chip,pid=int(cells[1]),process_memory_MB=int(cells[3])))
  visible=[int(v)for v in d.get('visible_devices','').split(',')if v]
  out.update(declared_physical_cards=visible,matched_own_NPU_workers=matched,
             process_count=len(processes),container_observed=container is not None)
  if len(visible)==3 and len(matched)==3 and {x['physical_npu']for x in matched}==set(visible):
   out.update(captured=True,host=P['host'],root=str(r),launcher_pid=launcher,container=container,
              physical_cards=visible,target_physical_cards=visible[:2],learner_physical_card=visible[2],
              workers=matched,process_instances=processes,no_other_users_rows_retained=True)
print(json.dumps(out))
'''

# 服务器本地可保存同一段只读采集程序的结果。联网观察恢复后，先取这份
# 原始快照；即使 worker 已正常退出，也不会丢掉运行时的真实设备绑定。
REMOTE = '''
from pathlib import Path
import json
saved=Path(P['root'])/'resource-observations-confirm350'/(P['tag']+'.resources.json')
if saved.exists():
 binding=json.loads(saved.read_text())
 assert binding['capture_program_sha256']==''' + repr(hashlib.sha256(LIVE_REMOTE.encode()).hexdigest()) + '''
 assert binding['captured'] and binding['tag']==P['tag'] and binding['root']==P['root'] and binding['host']==P['host']
 assert binding['stage']=='measuring' and not binding['completed']
 assert len(set(binding['physical_cards']))==3 and len(binding['workers'])==3
 assert {v['physical_npu']for v in binding['workers']}==set(binding['physical_cards'])
 assert binding['process_instances'] and all(v['start_ticks']is not None for v in binding['process_instances'])
 print(json.dumps(binding))
else:
''' + textwrap.indent(LIVE_REMOTE, ' ')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--artifacts", type=Path, required=True)
    args = p.parse_args()
    D, Q = args.validation_root, args.validation_root / "resource338"
    jobs = []
    for host, address in (("A", "159.138.5.111"), ("B", "189.1.227.167")):
        stage = json.loads((Q / f"stage-{host}.json").read_text())
        for launch in stage["launched"]:
            key = "A" if host == "A" else "B_gsm" if launch["root"].endswith("online-background") else "B_spoc"
            for case in json.loads((Q / f"queue-{key}.json").read_text())["cases"]:
                if case.get("resource_repeat_group"):
                    jobs.append(dict(host=address, root=launch["root"], tag=case["tag"], launcher=case["launcher"]))
    jobs.append(dict(host="159.138.5.111", root=jobs[0]["root"],
                     tag="a111-v292-spoc333-aux3card-confirmed256-order743", launcher="run_dense_v292_docker.py"))
    captured = {job["tag"] for job in jobs if (Q / (job["tag"] + ".resources.json")).exists()}

    def one(job):
        try:
            call = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                                   "-o", "ConnectTimeout=10", "wangzili@" + job["host"], "python3 -"],
                                  input=("P=" + repr(job) + "\n" + REMOTE).encode(), capture_output=True, timeout=40)
            if call.returncode:
                return dict(tag=job["tag"], observation_error=call.stderr.decode()[-1500:], terminal=False)
            return json.loads(call.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            return dict(tag=job["tag"], observation_error=repr(error), terminal=False)

    while len(captured) < len(jobs):
        with ThreadPoolExecutor(max_workers=3) as pool:
            observations = list(pool.map(one, [job for job in jobs if job["tag"] not in captured]))
        for result in observations:
            if result.get("captured"):
                (Q / (result["tag"] + ".resources.json")).write_text(json.dumps(result, indent=2) + "\n")
                captured.add(result["tag"])
                print(json.dumps(result), flush=True)
        (args.artifacts / "resource338-bindings-state.json").write_text(json.dumps(
            dict(completed=len(captured) == len(jobs), captured=sorted(captured), observations=observations), indent=2) + "\n")
        if len(captured) < len(jobs):
            time.sleep(30)


if __name__ == "__main__":
    main()
