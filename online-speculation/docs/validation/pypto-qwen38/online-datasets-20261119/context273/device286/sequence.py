from pathlib import Path
import sys,runpy,json
from _guard_bootstrap import Guard
q=Path(__file__).resolve().parent;r=q.parent/'oneserving-v109-b99e118/online-studies';status=dict(completed=False,cases=[],scope='IndependentNPU allnodeTorch comparison plusunchangedrealweighttraining retry aftercontrol_mallocOOM. Onepersistent16GiBguard acrosscases; no newrequest stream/head/recipe.')
cases=[('independent-TV','validate_independent.py',['--suffix','--combined','--context','1024','--objective','tv']),('independent-KL','validate_independent.py',['--suffix','--combined','--context','1024','--objective','kl']),('training1024','validate_training.py',['--context','1024','--model','/data/disk/models/Qwen3.8-27B','--head',str(r.parent/'kl-prewarm-suffix-tp2')])]
g=Guard()
try:
 for name,script,args in cases:
  print('BEGIN',name,flush=True);sys.argv=[str(q/script),*args,'--device','0','--output',str(q/(name+'.json'))]
  try:runpy.run_path(str(q/script),run_name='__main__')
  except BaseException as e:
   status['cases'].append(dict(name=name,passed=False,error=repr(e)));raise
  status['cases'].append(dict(name=name,passed=True));(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 status['completed']=True
finally:
 try:g.close()
 finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
