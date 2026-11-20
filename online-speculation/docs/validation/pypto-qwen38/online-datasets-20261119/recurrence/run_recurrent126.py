from pathlib import Path
import subprocess,sys,json,time,os,hashlib
r=Path(__file__).resolve().parent;b=r.parent.parent;sys.path[:0]=[str(b),str(r)]
from watch_glm53_validation import acquire_devices,idle_devices,release_devices
q=r/'recurrent-v126';locks=None;name='orthrus-recurrent126'
try:
 deadline=time.monotonic()+600
 while True:
  idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
  for card in idle:
   if card not in tuple(range(8)):continue
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
   if locks is not None:break
  else:
   if time.monotonic()>deadline:raise RuntimeError('no idle card in permitted set')
   time.sleep(2);continue
  break
 (q/'source.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((q/'src').rglob('*.py'))+[q/'validate.py',q/'summarize_qwen38_pypto_profile.py',r/'docker_entry_recurrent126.py']},indent=2)+'\n')
 cmd=['docker','run','--rm','--name',name,'--privileged','--network','host','--shm-size','32g','-e','ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1','-v',str(b)+':'+str(b),'-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3','quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc',str(r/'docker_entry_recurrent126.py'),str(card),str(q/'validate.py'),'--output',str(q/'report.json'),'--variants','sequential64,decay64']
 with(q/'run.log').open('x')as f:x=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 d=dict(completed=False,returncode=x.returncode,physical_card=card,scope='independent TP2 recurrence; not model throughput acceptance')
 if(q/'report.json').exists():d.update(json.loads((q/'report.json').read_text()))
 d['completed']=x.returncode==0 and d.get('passed')and d.get('clean_shutdown');(r/'a111-recurrent126-audit-r1.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
finally:
 subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if locks is not None:release_devices(locks)
