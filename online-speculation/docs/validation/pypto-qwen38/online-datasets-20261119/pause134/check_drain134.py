from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess

A=Path('/Users/li/Documents/Codex/2026-09-13/kan-xi/artifacts/qwen38-online-datasets-20261119')
manifest=json.loads((A/'user-pause-after-current134.json').read_text())
CODE=r'''from pathlib import Path
import json,os,subprocess,time
r=Path(ROOT);jobs=json.loads(JOBS);out=[]
for job in jobs:
 p=r/(job['tag']+'.json');d=json.loads(p.read_text())if p.exists()else{}
 proc=Path('/proc')/str(job['pid']);cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()if proc.exists()else''
 progress=d.get('progress',{});latest={k:v[-1].get('requests')for k,v in progress.items()if v}
 out.append(dict(tag=job['tag'],pid=job['pid'],running=job['tag']in cmd,completed=d.get('completed'),clean_shutdown=d.get('clean_shutdown'),stage=d.get('stage'),error=d.get('error'),progress=latest))
controllers=[line for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()if line.startswith('python3 study_queue_')]
print(json.dumps(dict(jobs=out,remaining_dispatchers=controllers,time=time.time())))
'''

def poll(kv):
 h,m=kv;c=CODE.replace('ROOT',repr(m['root'])).replace('JOBS',repr(json.dumps(m['draining_model_jobs'])))
 x=subprocess.run(['ssh','-o','ControlPath=none','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@'+h,'python3 -c '+shlex.quote(c)],capture_output=True,timeout=40)
 assert x.returncode==0,x.stderr
 return h,json.loads(x.stdout)
out={}
with ThreadPoolExecutor(max_workers=3)as p:
 for h,v in p.map(poll,manifest.items()):
  out[h]=v
  print(h,'dispatchers',len(v['remaining_dispatchers']),[(z['tag'],z['running'],z['completed'],z['clean_shutdown'],z['progress'],z['error'])for z in v['jobs']])
(A/'pause-drain-latest134.json').write_text(json.dumps(out,indent=2)+'\n')
