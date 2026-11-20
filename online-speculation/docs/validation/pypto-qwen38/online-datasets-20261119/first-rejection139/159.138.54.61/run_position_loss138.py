from pathlib import Path
import sys,subprocess,json,hashlib,time
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-tp2-test');b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');q=r/'loss-v138';q.mkdir();sys.path[:0]=[str(b),str(r)];from host_runtime import PYTHON,environment
from watch_glm53_validation import acquire_devices,idle_devices,release_devices
proof=json.loads((b/'scope-v138-review/report.json').read_text());assert proof['completed'];locks=None;deadline=time.monotonic()+3600
try:
 while True:
  for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
   if locks is not None:break
  if locks is not None:break
  if time.monotonic()>deadline:raise TimeoutError('No idle card for independent loss validation')
  time.sleep(5)
 e=environment();e['PYTHONPATH']=str(r/'src-dense-v138')+':'+e['PYTHONPATH'];e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PYPTO_CACHE_DIR']=str(r/'jit');e['PYPTO_CACHE']='1'
 (q/'source.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((r/'src-dense-v138').rglob('*.py'))+[r/'position_loss138.py']},indent=2)+'\n')
 with(q/'run.log').open('x')as f:x=subprocess.run([str(PYTHON),str(r/'position_loss138.py'),'--weighting','first_rejection','--teacher-temperature','0.0625','--output',str(q/'report.json')],env=e,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 d=dict(completed=False,returncode=x.returncode,physical_card=card,scope='Independent full-vocabulary weighted loss+gradient; not service throughput')
 if(q/'report.json').exists():d.update(json.loads((q/'report.json').read_text()));d['completed']=x.returncode==0 and d.get('passed')and d.get('clean_shutdown')
 if not d['completed']:d['error']='Loss138 device validation failed'
 (r/'loss138-validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
except BaseException as error:
 (r/'loss138-validation.json').write_text(json.dumps(dict(completed=False,error=repr(error)),indent=2)+'\n')
 raise
finally:
 if locks is not None:release_devices(locks)
