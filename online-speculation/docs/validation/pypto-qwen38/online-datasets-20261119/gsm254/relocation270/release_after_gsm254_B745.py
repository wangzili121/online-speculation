"""只解除后续探索的资源依赖，不代表 GSM 性能验收通过。"""
from pathlib import Path
import time,json,subprocess,os
r=Path(__file__).parent;tag='b167-v245-gsm254-confirm745';p=r/(tag+'.json');status=r/'queue-gsm254-B745.status.json'
while True:
 d={}
 if p.exists():
  try:d=json.loads(p.read_text())
  except ValueError:time.sleep(1);continue
 terminal=d.get('completed')or d.get('error')
 if not terminal and status.exists():
  s=json.loads(status.read_text());terminal=s.get('execution_complete')and any(c.get('error')for c in s['cases'])
 live=any('run_dense_v245.py --tag '+tag+' 'in line for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines())
 if terminal and not live:break
 time.sleep(5)
proof=dict(completed=True,prior_trial_terminal=True,prior_success=d.get('completed')is True and d.get('clean_shutdown')is True,prior_error=d.get('error'),scope='Resource release only after actual launcherexit. FullNNexploration mayrun evenif primaryconfirmation failed; no performance acceptance implied.')
(r/'gsm254-B745-resource-release.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
