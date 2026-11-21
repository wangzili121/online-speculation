from pathlib import Path
import os,sys,subprocess,time,json
r=Path(__file__).parent;b=r.parent.parent;q=b/'scope-v214-review';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
sys.path.insert(0,str(r));from watch_glm53_validation import acquire_devices,idle_devices,release_devices
assert json.loads((q/'report.json').read_text())['completed']
lockroot=b.parent/'orthrus-glm53-flash-20261101';deadline=time.monotonic()+36000;locks=None
while time.monotonic()<deadline:
 info=dict(l.split(':',1)for l in Path('/proc/meminfo').read_text().splitlines());available=int(info['MemAvailable'].split()[0])*1024
 starting=[]
 for row in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
  a=row.split()
  if len(a)>4 and a[0]=='python3'and a[1].startswith('run_dense_')and a[2]=='--tag':
   p=r/(a[3]+'.json')
   if p.exists()and json.loads(p.read_text()).get('stage')=='starting':starting.append(a[3])
 if available<12<<30 or starting:time.sleep(5);continue
 idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
 for card in idle:
  locks=acquire_devices(lockroot,(card,))
  if locks is not None:break
 if locks is not None:break
 time.sleep(5)
else:raise TimeoutError('No genuinelyidle singlecard')
e=environment();e['PYTHONPATH']=':'.join([str(q/'scripts'),str(r/'src-dense-v214'),str(r.parent/'pypto-serving-qwen3.8-27B'),str(r),e['PYTHONPATH']]);e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PYPTO_CACHE_DIR']=str(r/'jit');e['PYPTO_CACHE']='1'
p=r/'statistics214-device-launch.json';p.write_text(json.dumps(dict(card=card,visible=e['ASCEND_RT_VISIBLE_DEVICES'],source='v214',startup_without_other_own_compilers=True,host_available=available),indent=2)+'\n')
try:
 with(r/'statistics214-device.log').open('x')as f:
  rc=subprocess.call([str(PYTHON),'-c','from validate_qwen38_pypto_statistics import main; from _guard_bootstrap import Guard; g=Guard();\ntry: main(guard=g)\nfinally: g.close()','--output',str(r/'statistics214-device.json')],cwd=r,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 print(json.dumps(dict(returncode=rc,card=card)))
finally:release_devices(locks)
