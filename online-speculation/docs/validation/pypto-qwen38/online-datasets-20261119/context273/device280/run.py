from pathlib import Path
import subprocess,sys,time,json,os,signal
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');root=b/'oneserving-v109-b99e118/online-studies';q=b/'context-v280';sys.path[:0]=[str(b),str(root)];from watch_glm53_validation import acquire_devices,idle_devices,release_devices
locks=None;deadline=time.monotonic()+28800
while locks is None:
 for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
  locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
  if locks is not None:break
 if locks is None:
  if time.monotonic()>deadline:raise TimeoutError('no idle single NPU')
  time.sleep(2)
name='orthrus-context280-single'
try:
 image='quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc';image_id=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
 (q/'device.json').write_text(json.dumps(dict(physical_card=card,physical_cards=1,image=image_id,scope='Separate standalone component validation; no extra learner or resources counted for primary2card experiments.')))
 command=['docker','run','--pull=never','--rm','--name',name,'--privileged','--network','host','--shm-size','16g','-e','ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1','-v',str(b)+':'+str(b),'-v','/data/disk/models/Qwen3.8-27B:/data/disk/models/Qwen3.8-27B:ro','-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3',image_id,str(q/'entry.py'),str(card),str(os.getuid()),str(os.getgid())]
 with(q/'device.log').open('x')as log:
  child=subprocess.Popen(command,cwd=q,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=child.wait(timeout=3600)
  except subprocess.TimeoutExpired:
   subprocess.run(['docker','rm','-f',name],capture_output=True);child.wait(timeout=30);raise
 (q/'exit.json').write_text(json.dumps(dict(completed=rc==0,returncode=rc,no_automatic_retry=True)))
finally:
 subprocess.run(['docker','rm','-f',name],capture_output=True);release_devices(locks)
