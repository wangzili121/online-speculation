"""仅修正未启动模型的111等待任务可选卡范围；不停止任何NPU工作。"""
import hashlib
import json
from pathlib import Path
import subprocess


REMOTE = r'''
from pathlib import Path
import os,signal,json,hashlib,subprocess,time,datetime
r=Path(P['launch']['root']);marker=r/'sync346-A-launch.json';receipt=r/'sync348-unstarted-device-amendment.json'
if receipt.exists():
 out=json.loads(marker.read_text());assert out['plan_sha256']==P['new_plan_sha256'];print(json.dumps(dict(launch=out,receipt=json.loads(receipt.read_text()))));raise SystemExit(0)
old=json.loads(marker.read_text());assert old['pid']==P['launch']['pid'];plan=r/old['plan']
assert hashlib.sha256(plan.read_bytes()).hexdigest()==P['old_plan_sha256']
before=json.loads(plan.read_text());after=P['plan'];x=list(before['cases'][0]['args']);y=list(after['cases'][0]['args']);pos=x.index('--devices')+1
assert y[pos]=='0,1,2,3,4,5,6,7';x[pos]=y[pos];assert x==y
check=json.loads(json.dumps(before));check['cases'][0]['args'][pos]=y[pos];assert check==after
tag=old['tag'];result=r/(tag+'.json');assert not result.exists(),'Model report already exists; do not alter task'
rows=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
launchers=[int(v.split(None,1)[0])for v in rows if before['cases'][0]['launcher']+' --tag '+tag+' 'in v]
assert len(launchers)==1,launchers
controller=old['pid'];launcher=launchers[0];instances=[];stopped=[]
def state(pid):return(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
try:
 for pid,expected in [(controller,old['plan']),(launcher,' --tag '+tag+' ')]:
  p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert p.stat().st_uid==os.getuid()and expected in cmd
  assert state(pid)not in ('T','t'),'Preexisting stopped task requires inspection'
  instances.append(dict(pid=pid,cmdline=cmd,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19])))
  os.kill(pid,signal.SIGSTOP);stopped.append(pid)
 for _ in range(40):
  if all(state(pid)in('T','t')for pid in stopped):break
  time.sleep(.05)
 assert all(state(pid)in('T','t')for pid in stopped)
 assert not result.exists(),'Model started before control stop; resume unchanged'
 pairs=[v.split(None,2)for v in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
 own={controller,launcher}
 while True:
  extra={int(v[0])for v in pairs if int(v[1])in own}-own
  if not extra:break
  own.update(extra)
 descendants=[dict(pid=int(v[0]),cmdline=v[2])for v in pairs if int(v[0])in own-{controller,launcher}]
 assert all(v['cmdline'].split()[0].endswith('npu-smi')for v in descendants),descendants
 container=subprocess.run(['docker','inspect','orthrus-online-111-'+tag],capture_output=True,text=True,timeout=6)
 assert container.returncode!=0,'A model container exists; resume unchanged'
 smi=subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)
 active=[]
 for line in smi.splitlines():
  cells=[v.strip()for v in line.strip().strip('|').split('|')]
  if len(cells)>=4 and cells[1].isdigit()and int(cells[1])in own:active.append(line)
 assert not active,'Owned waiting task already has NPU memory'
 for pid in stopped:os.kill(pid,signal.SIGTERM)
 for pid in stopped:os.kill(pid,signal.SIGCONT)
 stopped.clear()
 for _ in range(60):
  if all(not(Path('/proc')/str(v['pid'])).exists()for v in instances):break
  time.sleep(.05)
 assert all(not(Path('/proc')/str(v['pid'])).exists()for v in instances),'Waiting control did not exit; inspect before redispatch'
except BaseException:
 for pid in stopped:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 raise
record=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),retired_waiting_instances=instances,
            no_model_report_before_or_after_control_stop=not result.exists(),no_NN_children=True,no_owned_NPU_process=True,
            original_controller_and_launcher_exited=True,NN_models_stopped_or_restarted=False,
            old_candidates=before['cases'][0]['args'][pos],new_candidates=y[pos],minimum_available_host_gib=after['cases'][0]['minimum_available_host_gib'],
            numerical_args_unchanged=True,B_running_task_untouched=True)
(r/(old['plan']+'.before-device-amendment348')).write_bytes(plan.read_bytes())
(r/'sync346-A-launch.before-device-amendment348.json').write_bytes(marker.read_bytes())
raw=(json.dumps(after,indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==P['new_plan_sha256'];plan.write_bytes(raw)
with(r/(old['plan']+'.controller.log')).open('ab')as log:
 log.write((json.dumps(dict(unstarted_device_amendment=record))+'\n').encode());log.flush()
 child=subprocess.Popen(['python3',str(r/old['controller']),'--plan',old['plan']],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out=old|dict(pid=child.pid,plan_sha256=P['new_plan_sha256'],dispatched_UTC=record['observed_UTC'],previous_unstarted_controller_pid=controller,device_candidates_amended=True)
receipt.write_text(json.dumps(record,indent=2)+'\n');marker.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(launch=out,receipt=record)))
'''


def main():
    q = Path(__file__).resolve().parent
    source = q.parent / "sync346"
    old = (source / "queue-A.json.before-device-amendment348").read_bytes()
    raw = (source / "queue-A.json").read_bytes()
    payload = dict(launch=json.loads((source / "launch-A.json").read_text()),
                   old_plan_sha256=hashlib.sha256(old).hexdigest(),
                   new_plan_sha256=hashlib.sha256(raw).hexdigest(), plan=json.loads(raw))
    call = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "wangzili@159.138.5.111", "python3 -"],
        input=("P=" + repr(payload) + "\n" + REMOTE).encode(), capture_output=True, timeout=45,
    )
    if call.returncode:
        (q / "amendment-needs-inspection.json").write_text(json.dumps(
            dict(returncode=call.returncode, stdout=call.stdout.decode(), stderr=call.stderr.decode()), indent=2) + "\n")
        raise RuntimeError(call.stderr.decode()[-3000:])
    out = json.loads(call.stdout)
    backup = source / "launch-A.json.before-device-amendment348"
    if not backup.exists():
        backup.write_bytes((source / "launch-A.json").read_bytes())
    else:
        assert json.loads(backup.read_text())["pid"] == out["launch"]["previous_unstarted_controller_pid"]
    (source / "launch-A.json").write_text(json.dumps(out["launch"], indent=2) + "\n")
    (q / "unstarted-device-amendment.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
