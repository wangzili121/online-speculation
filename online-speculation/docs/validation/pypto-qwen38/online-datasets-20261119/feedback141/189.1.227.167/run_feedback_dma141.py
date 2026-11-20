from pathlib import Path
import sys,subprocess,json,hashlib,time
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-background');b=r.parent.parent;q=r/'feedback-dma141';q.mkdir();sys.path[:0]=[str(b),str(r)];from host_runtime import PYTHON,environment
from watch_glm53_validation import acquire_devices,idle_devices,release_devices
assert json.loads((r/'v141-cpu-report.json').read_text())['completed'];locks=None;deadline=time.monotonic()+3600
try:
 while True:
  for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
   if locks is not None:break
  if locks is not None:break
  if time.monotonic()>deadline:raise TimeoutError('No idle card for bounded independent DMA check')
  time.sleep(5)
 e=environment();e['PYTHONPATH']=str(r/'src-dense-v141')+':'+str(r.parent/'pypto-serving-qwen3.8-27B')+':'+e['PYTHONPATH'];e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PYPTO_CACHE_DIR']=str(r/'jit');e['PYPTO_CACHE']='1'
 (q/'source.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((r/'src-dense-v141').rglob('*.py'))+[r/'feedback_dma141.py',r/'src-dense-v138/blockspec/pypto_qwen38/oneserving_tp2_online.py']},indent=2)+'\n')
 with(q/'run.log').open('x')as f:p=subprocess.run([str(PYTHON),str(r/'feedback_dma141.py'),'--reference',str(r/'src-dense-v138/blockspec/pypto_qwen38/oneserving_tp2_online.py'),'--output',str(q/'report.json')],env=e,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 d=dict(completed=False,returncode=p.returncode,physical_card=card)
 if(q/'report.json').exists():d.update(json.loads((q/'report.json').read_text()));d['completed']=p.returncode==0 and d.get('passed')and d.get('clean_shutdown')
 if not d['completed']:d['error']='Feedback141 standalone validation failed'
 (r/'feedback141-validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
except BaseException as err:
 (r/'feedback141-validation.json').write_text(json.dumps(dict(completed=False,error=repr(err)),indent=2)+'\n');raise
finally:
 if locks is not None:release_devices(locks)
