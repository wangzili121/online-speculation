from pathlib import Path
import json,copy,subprocess,gzip
import argparse
p=argparse.ArgumentParser();p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();D=a.validation_root;Q=D/'relocate318';plan=copy.deepcopy(json.loads((Q/'queue-spoc322B.json').read_text()));oldtag=plan['cases'][0]['tag'];newtag=oldtag+'-metadata323';plan['cases'][0]['tag']=newtag
for spec in plan['cases'][1]['requires']:
 if spec['file']==oldtag+'.json':spec['file']=newtag+'.json'
remote=r'''from pathlib import Path
import os,signal,subprocess,time,json,hashlib
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-studies';pid=3511299;proc=Path('/proc')/str(pid);cmd=None
assert not(r/'a111-v292-spoc312-confirm745.json').exists()and not(r/'a111-v292-spoc312-confirm745.log').exists()
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert proc.stat().st_uid==os.getuid()and'study_queue_spoc312.py --plan queue-spoc322B.json'in cmd
 os.kill(pid,signal.SIGSTOP)
 try:
  assert not [x for x in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]if int(x.split(None,2)[1])==pid]
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 except BaseException:
  os.kill(pid,signal.SIGCONT);raise
else:
 status=json.loads((r/'queue-spoc322B.status.json').read_text());assert status['execution_complete']and all(not x.get('completed')and x.get('error')for x in status['cases']),status
 cmd='Exited naturally after failed diagnostic and blocked formal prerequisite'
failed={n:(r/n).read_text()for n in [P['oldtag']+ext for ext in ['.json','.source.json','.log','.launch.log']]+['queue-spoc322B.status.json','queue-spoc322B.json.controller.log']}
public=Path('/data/disk/models/Qwen3.8-27B');cache=b/'inference-tuning-20261116/model-view/.cache';image='sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2'
code="""from pathlib import Path
import hashlib,json,os
original=Path('/original-cache/huggingface/download/config.json.metadata').read_bytes();assert hashlib.sha256(original).hexdigest()=='f94e5f0724c3362f96856b32980d436e24e27b784090897097bed20c148c2dc5'
assert original.decode().splitlines()[0]=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
for n,h in {'config.json':'191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab','model.safetensors.index.json':'77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df','tokenizer.json':'0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'}.items():assert hashlib.sha256((Path('/model')/n).read_bytes()).hexdigest()==h,n
p=Path('/model/.cache/huggingface/download/config.json.metadata');p.parent.mkdir(parents=True,exist_ok=True)
with p.open('xb')as f:f.write(original)
os.chmod(p,0o644)
print(json.dumps(dict(restored_only_original_revision_marker=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_weights_unchanged=True,no_NPU_devices_mounted=True)))
"""
x=subprocess.run(['docker','run','--rm','--network','none','-v',str(public)+':/model','-v',str(cache)+':/original-cache:ro','--entrypoint','/usr/bin/python3',image,'-c',code],capture_output=True,text=True);assert x.returncode==0,x.stderr;marker=json.loads(x.stdout)
with(r/'queue-spoc323B.json').open('x')as f:json.dump(P['plan'],f,indent=2)
with(r/'queue-spoc323B.json.controller.log').open('x')as log:p=subprocess.Popen(['python3','study_queue_spoc312.py','--plan','queue-spoc323B.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
old=json.loads((r/'spoc322-frozen-source.json').read_text());print(json.dumps(dict(pid=p.pid,source=old['files'],base=json.loads((r/'spoc318B-base-files.json').read_text()),failed=failed,marker_restoration=marker,stopped_waiting_controller=dict(pid=pid,cmd=cmd,no_children=True),NN_code_changed=False,A745_unchanged=True)))
'''
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -'],input=('P='+repr(dict(plan=plan,oldtag=oldtag))+'\n'+remote).encode(),capture_output=True,timeout=45);assert x.returncode==0,x.stderr.decode();out=json.loads(x.stdout);failed=out.pop('failed');q=Q/'failed-preweight322';q.mkdir(exist_ok=True)
for n,t in failed.items():
 z=t.encode();suffix='.gz'if n.endswith('.log')else'';(q/(n+suffix)).write_bytes(gzip.compress(z,mtime=0)if suffix else z)
(Q/'recovery323-stage.json').write_text(json.dumps(out,indent=2)+'\n');(Q/'queue-spoc323B.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(pid=out['pid'],marker=out['marker_restoration'],no_reserved_execution=True)))
