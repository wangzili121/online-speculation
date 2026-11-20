from pathlib import Path
import sys,json,time,subprocess,hashlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');b=r.parent.parent;q=r/'teacher-device142';q.mkdir();sys.path.insert(0,str(r));from watch_glm53_validation import acquire_devices,idle_devices,release_devices
assert json.loads((r/'v142-cpu-report.json').read_text())['completed'];locks=None;deadline=time.monotonic()+3600;cid=None
try:
 while True:
  for card in idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)):
   locks=acquire_devices(b.parent/'orthrus-glm53-flash-20261101',(card,))
   if locks is not None:break
  if locks is not None:break
  if time.monotonic()>deadline:raise TimeoutError('No idle card for teacher transfer validation')
  time.sleep(5)
 (q/'source.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((r/'src-dense-v142').rglob('*.py'))+[r/'feedback_device142.py',r/'src-dense-v138/blockspec/pypto_qwen38/oneserving_tp2_online.py',r/'docker_entry_v142.py']},indent=2)+'\n')
 command=['docker','run','--rm','--pull=never','--cidfile',str(q/'container.cid'),'--name','orthrus-teacher-device142','--label','orthrus.owner=wangzili','--label','orthrus.task=teacher-device142','--privileged','--network','host','--shm-size','4g','-e','ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1','-v',str(b)+':'+str(b),'-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3','quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc',str(r/'docker_entry_v142.py'),str(card),str(r/'feedback_device142.py'),'--reference',str(r/'src-dense-v138/blockspec/pypto_qwen38/oneserving_tp2_online.py'),'--output',str(q/'report.json')]
 (q/'command.json').write_text(json.dumps(command,indent=2)+'\n')
 with(q/'run.log').open('x')as f:p=subprocess.run(command,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 d=dict(completed=False,returncode=p.returncode,physical_card=card)
 if(q/'report.json').exists():d.update(json.loads((q/'report.json').read_text()));d['completed']=p.returncode==0 and d.get('passed')and d.get('clean_shutdown')
 if not d['completed']:d['error']='Teacher142 standalone device transform failed'
 (r/'teacher142-validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
except BaseException as e:
 (r/'teacher142-validation.json').write_text(json.dumps(dict(completed=False,error=repr(e)),indent=2)+'\n');raise
finally:
 if(q/'container.cid').exists():
  cid=(q/'container.cid').read_text().strip();v=subprocess.run(['docker','inspect','--format','{{json .Config.Labels}}',cid],capture_output=True,text=True)
  if v.returncode==0:
   labels=json.loads(v.stdout);assert labels.get('orthrus.owner')=='wangzili'and labels.get('orthrus.task')=='teacher-device142';subprocess.run(['docker','rm','-f',cid],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if locks is not None:release_devices(locks)
