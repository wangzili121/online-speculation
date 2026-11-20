from pathlib import Path
import argparse,json,os,sys,time,subprocess
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();r=Path(__file__).resolve().parent;b=r.parent.parent;q=b/'pub-suffix-arena-v74-r2';sys.path[:0]=[str(b),str(r)]
from host_runtime import PYTHON,environment,prepare
from watch_glm53_validation import idle_devices,acquire_devices,release_devices
prepare();e=environment();e['PYTHONPATH']=str(q/'src')+':'+str(r.parent/'pypto-serving-qwen3.8-27B')+':'+str(r)+':'+e['PYTHONPATH'];locks=None;deadline=time.monotonic()+7200;d=dict(completed=False,stage='waiting',physical_device_count=1,scope='Standalone actual head packing/readback, not a third learner for an online arm')
def save():
 p=r/(a.tag+'.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
save()
try:
 while time.monotonic()<deadline:
  for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
   if card not in(6,7):continue
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',[card])
   if locks is not None:break
  if locks is not None:break
  time.sleep(1)
 if locks is None:raise TimeoutError('No idle card for publication validation')
 e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PUB_PHYSICAL_CARD']=str(card);d.update(stage='running',visible_devices=str(card));save()
 with(r/(a.tag+'.log')).open('x')as f:x=subprocess.run([str(PYTHON),str(q/'micro.py')],cwd=q,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=2400)
 d.update(json.loads((q/'report.json').read_text())if(q/'report.json').exists()else{},micro_exit_code=x.returncode,physical_device_count=1)
 assert x.returncode==0 and d.get('completed') and d.get('two_updates_exact'),d
 assert len(d['steps'])==2 and d['avoided_publication_allocation_bytes']==582211072
 d['clean_shutdown']=True
except BaseException as err:d.update(completed=False,error=repr(err));raise
finally:
 save()
 if locks is not None:release_devices(locks)
