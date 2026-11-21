from pathlib import Path
import sys,subprocess,json,time,os,signal
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-background';q=b/'training-linear-v261';sys.path[:0]=[str(b),str(r)];from host_runtime import PYTHON,environment
from watch_glm53_validation import idle_devices,acquire_devices,release_devices
locks=None;deadline=time.monotonic()+28800
while locks is None:
 idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
 for card in idle:
  locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
  if locks is not None:break
 if locks is None:
  if time.monotonic()>deadline:raise TimeoutError('no idle single NPU')
  time.sleep(2)
try:
 e=environment();e['ASCEND_RT_VISIBLE_DEVICES']=str(card);e['PYPTO_CACHE_DIR']=str(r/'jit');e['PYPTO_CACHE']='1';e['PYTHONPATH']=':'.join([str(r/'src-dense-v259'),str(r.parent/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH'],str(r)])
 (q/'device.json').write_text(json.dumps(dict(physical_card=card,physical_card_count=1,scope='Independent component validation, never an additional learner for primary2card experiments.')))
 entry="from _guard_bootstrap import Guard; import runpy; g=Guard();\ntry:runpy.run_path('validate.py',run_name='__main__')\nfinally:g.close()"
 command=[str(PYTHON),'-c',entry,]
 with(q/'device.log').open('x')as file:
  p=subprocess.Popen(command,cwd=q,env=e,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 (q/'exit.json').write_text(json.dumps(dict(returncode=rc,completed=rc==0,scope='No auto retry; preserve compile/device error.')))
finally:release_devices(locks)
