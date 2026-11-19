from pathlib import Path
import sys,subprocess,time,json,os
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'pub-suffix-npu-v3';sys.path.insert(0,str(b/'oneserving-v109-b99e118/online-studies'))
from watch_glm53_validation import idle_devices,acquire_devices,release_devices
deadline=time.monotonic()+7200;locks=None
(r/'waiting.json').write_text(json.dumps(dict(pid=os.getpid(),stage='waiting_for_idle_card',allowed_devices=[2,3,4,5,6,7]))+'\n')
while time.monotonic()<deadline:
 idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
 for card in idle:
  if card not in {2,3,4,5,6,7}:continue
  locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',[card])
  if locks is not None:break
 else:time.sleep(1);continue
 break
if locks is None:raise TimeoutError('no idle single card for loss-training microbenchmark')
(r/'waiting.json').write_text(json.dumps(dict(pid=os.getpid(),stage='running',physical_card=card))+'\n')
name='orthrus-pub-suffix-npu-v3';cmd=['docker','run','--pull=never','--rm','--name',name,'--privileged','--network','host','--shm-size','16g','-e','ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1','-v',str(b)+':'+str(b),'-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3','quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc',str(r/'entry.py'),str(card)]
try:
 with (r/'run.log').open('x') as f:x=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 (r/'exit.json').write_text(json.dumps(dict(returncode=x.returncode))+'\n')
finally:
 subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);release_devices(locks)
