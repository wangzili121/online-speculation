from pathlib import Path
import subprocess,sys,json,time,hashlib
r=Path(__file__).resolve().parent;b=r.parent.parent;q=r/'loss-greedy147';q.mkdir();sys.path[:0]=[str(b),str(r)]
from host_runtime import PYTHON,environment
from watch_glm53_validation import acquire_devices,idle_devices,release_devices
locks=None
try:
 deadline=time.monotonic()+1200
 while locks is None:
  for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
   if locks is not None:break
  if locks is None:
   if time.monotonic()>deadline:raise TimeoutError('No genuinely idle card')
   time.sleep(5)
 e=environment();e['PYTHONPATH']=str(r/'src-dense-v147')+':'+e['PYTHONPATH'];e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PYPTO_CACHE_DIR']=str(r/'jit');e['PYPTO_CACHE']='1'
 source=b/'scope-v147-review/scripts/validate_qwen38_pypto_position_loss.py'
 (q/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((r/'src-dense-v147').rglob('*.py'))+[source]},indent=2)+'\n')
 with(q/'run.log').open('x')as f:x=subprocess.run([str(PYTHON),str(source),'--output',str(q/'report.json'),'--teacher-targets','greedy','--weighting','first_rejection'],env=e,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 d=dict(returncode=x.returncode,physical_card=card,device_count=1,scope='Independent actual full-vocabulary hard-target loss/gradient vs autograd labelCE; not fullstream performance')
 if(q/'report.json').exists():d.update(json.loads((q/'report.json').read_text()))
 d['completed']=x.returncode==0 and d.get('passed')and d.get('clean_shutdown')
 if not d['completed']:d['error']='Greedy teacher loss validation failed; inspect retained raw log'
 (r/'loss147-validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
finally:
 if locks is not None:release_devices(locks)
