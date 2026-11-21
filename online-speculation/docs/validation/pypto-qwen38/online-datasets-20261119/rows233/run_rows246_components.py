from pathlib import Path
import sys,subprocess,json,time,os
r=Path(__file__).parent;b=r.parent.parent;sys.path.insert(0,str(r));from watch_glm53_validation import acquire_devices,idle_devices,release_devices
locks=None
for _ in range(36000):
 idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
 for card in idle:
  locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
  if locks is not None:break
 if locks is not None:break
 time.sleep(1)
else:raise TimeoutError('no idle A246 component card')
name='orthrus-rows246-components';output=r/'rows246-components-device.json';uid,gid=os.getuid(),os.getgid()
script='from validate_qwen38_pypto_row_layout import main;from _guard_bootstrap import Guard;import os;from pathlib import Path;g=Guard();\ntry: main(guard=g)\nfinally:\n g.close()\n p=Path('+repr(str(output))+')\n if p.exists():os.chown(p,'+str(uid)+','+str(gid)+')'
cmd=['docker','run','--rm','--name',name,'--privileged','--network','host','--shm-size','128g','-e','ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1','-v',str(b)+':'+str(b),'-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3','quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc',str(r/'docker_entry_v246.py'),str(card),'-c',script,'--output',str(output)]
(r/'rows246-device-launch.json').write_text(json.dumps(dict(card=card,source233_exact=True,probe_sha256=__import__('hashlib').sha256((r/'rows246-probe/validate_qwen38_pypto_row_layout.py').read_bytes()).hexdigest(),scope='Independent actualonecard TP2localdims/paging128_256_512 andGDN checks, not modelperformance.'),indent=2)+chr(10))
try:
 with(r/'rows246-device.log').open('x')as f:rc=subprocess.call(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 print(json.dumps(dict(returncode=rc,card=card)),flush=True)
finally:
 subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);release_devices(locks)
