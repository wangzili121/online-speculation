from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess

HERE = Path('/Users/li/Documents/Codex/2026-09-13/kan-xi')
A = HERE / 'artifacts/qwen38-online-datasets-20261119'
BASE = '/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/'
CODE = r'''from pathlib import Path
import json,os,signal,subprocess,time
root=Path(ROOT);uid=os.getuid();rows=[]
for line in subprocess.check_output(['ps','-u',str(uid),'-o','pid=,ppid=,args='],text=True).splitlines():
 p,pp,cmd=line.strip().split(None,2);proc=Path('/proc')/p
 args=cmd.split()
 if len(args)<2 or args[0]!='python3' or not args[1].startswith(('study_queue_','run_dense_v')):continue
 try:
  if proc.stat().st_uid==uid and(proc/'cwd').resolve()==root:rows.append(dict(pid=int(p),ppid=int(pp),cmd=cmd))
 except (FileNotFoundError,PermissionError):pass
stopped=[]
for row in rows:
 args=row['cmd'].split()
 if len(args)>2 and args[0]=='python3' and args[1].startswith('study_queue_') and '--plan'in args:
  os.kill(row['pid'],signal.SIGTERM);stopped.append(row)
# Stop only an owned wrapper that has not entered model execution. Freeze it briefly to avoid a spawn race.
waiting=[];running=[]
for row in rows:
 args=row['cmd'].split()
 if len(args)<5 or args[0]!='python3' or not args[1].startswith('run_dense_v')or args[2]!='--tag':continue
 tag=args[3];proc=Path('/proc')/str(row['pid'])
 if not proc.exists():continue
 os.kill(row['pid'],signal.SIGSTOP)
 try:
  children=[]
  for line in subprocess.check_output(['ps','-u',str(uid),'-o','pid=,ppid=,args='],text=True).splitlines():
   p,pp,cmd=line.strip().split(None,2)
   if int(pp)==row['pid']:children.append(dict(pid=int(p),cmd=cmd))
  result=root/(tag+'.json');d=json.loads(result.read_text())if result.exists()else{}
  computing=any('npu-smi'not in c['cmd']and not c['cmd'].startswith('ps ')for c in children)
  if not computing and d.get('stage')in(None,'waiting')and not d.get('completed'):
   os.kill(row['pid'],signal.SIGTERM);waiting.append(dict(**row,tag=tag,reason='User requested drain; waiting wrapper had no model child'))
  else:running.append(dict(**row,tag=tag,stage=d.get('stage'),children=children))
 finally:
  if proc.exists():os.kill(row['pid'],signal.SIGCONT)
report=dict(user_requested_pause_after_current=True,recorded_at=time.time(),root=str(root),stopped_dispatchers=stopped,canceled_waiting_wrappers=waiting,draining_model_jobs=running,policy='No new experiments. Let actual current computation finish; preserve outputs and release own devices. Resume only on explicit user instruction.')
old=root/'USER_PAUSE_AFTER_CURRENT134.json'
if old.exists():report['previous']=json.loads(old.read_text())
(root/'USER_PAUSE_AFTER_CURRENT134.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
'''

def drain(job):
    host, sub = job
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                             'wangzili@'+host,'python3 -c '+shlex.quote(CODE.replace('ROOT',repr(BASE+sub)))],
                            capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr
    return host,json.loads(result.stdout)

out={}
with ThreadPoolExecutor(max_workers=3)as pool:
    for host,result in pool.map(drain,[('159.138.5.111','online-studies'),
                                     ('189.1.227.167','online-background'),
                                     ('159.138.54.61','online-tp2-test')]):
        out[host]=result
        print(host,'stopped dispatchers',[r['pid']for r in result['stopped_dispatchers']],
              'canceled waiting',[r['pid']for r in result['canceled_waiting_wrappers']],
              'draining',[(r['pid'],r['tag'])for r in result['draining_model_jobs']])
(A/'user-pause-after-current134.json').write_text(json.dumps(out,indent=2)+'\n')
state=json.loads((A/'ACTIVE_STUDY_STATE.json').read_text());state['user_pause_after_current134']=dict(requested=True,dispatchers_stopped=True,new_experiments_allowed=False,drain_record='user-pause-after-current134.json',resume_requires_explicit_user_instruction=True);(A/'ACTIVE_STUDY_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
