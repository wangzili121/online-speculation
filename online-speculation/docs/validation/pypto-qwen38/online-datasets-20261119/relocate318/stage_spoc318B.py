"""Prepare the still-unstarted third confirmation on B; activation needs an audited A handoff."""
from pathlib import Path
import json,subprocess,zlib,base64,gzip,hashlib,copy
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();D=a.validation_root;A=a.artifacts;Q=D/'relocate318';Q.mkdir(exist_ok=True)
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
read=r'''from pathlib import Path
import json,zlib,sys,hashlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');freeze=json.loads((r/'spoc312-frozen-source.json').read_text())['files'];files={}
for n,h in freeze.items():
 p=(r if '/'not in n else r.parent)/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
 if n.startswith('online-studies/src-dense-v292/'):files[n.removeprefix('online-studies/src-dense-v292/')]=p.read_text()
names=['run_dense_v292_docker.py','docker_entry_v292.py','orthrus_bench_dense_v292.py','study_queue_spoc312.py','_study_metrics_windows209.py','watch_glm53_validation.py'];extras={n:(r/n).read_text()for n in names};extras.update({str(p.relative_to(r)):p.read_text()for p in(r/'rows246-probe').rglob('*.py')});assets={n:(r/'assets'/n).read_text()for n in ['budget303-boundary2.json','context284-synthetic-holdout2.json','spoc312-confirm-order745.json','spoc312-independent-holdout32.json']};sys.stdout.buffer.write(zlib.compress(json.dumps(dict(files=files,extras=extras,assets=assets,source=freeze)).encode()))
'''
x=subprocess.run([*ssh,'wangzili@159.138.5.111','python3 -'],input=read.encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode();bundle=json.loads(zlib.decompress(x.stdout));raw=json.dumps(bundle).encode();(Q/'source292_A_bundle.json.gz').write_bytes(gzip.compress(raw,mtime=0))
base=json.loads((D/'spoc312/queue-spoc312-confirm745.json').read_text())['cases'][0]
gate=copy.deepcopy(base);gate.update(tag='a111-v292-spoc318B-boundary',requires=[],purpose='B old synthetic context1024 + live KL/replay/publication diagnostic before any new reserved NN. Not performance qualification. OriginalA745 untouched until B passes and A is proved unstarted.');args=gate['args']
def setarg(a,k,v):
 if k in a:a[a.index(k)+1]=str(v)
 else:a.extend([k,str(v)])
for k,v in [('--tokens',24),('--strides',4),('--warm-update-stride',2),('--warm-update-steps',4),('--publish-every',1),('--warmup-updates',4),('--request-file','budget303-boundary2.json'),('--holdout-file','context284-synthetic-holdout2.json'),('--ar-check-requests',2),('--wait',28800)]:setarg(args,k,v)
args.remove('--stop-on-eos');args.append('--audit-replay');gate.pop('prompts_sha256',None)
formal=copy.deepcopy(base);formal['requires']=[dict(file='spoc318B-activated.json'),dict(file=gate['tag']+'.json',audit=True,greedy=True)];formal['purpose']+=' Operational relocation only if B gate passes AND originalA745 waiting process has no model report/log/children; same original trialID/data/head/Adam/NN/generation/update configuration. No retry of completed measurements.'
image='sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2'
for c in [gate,formal]:
 c['expected_source_manifest']='spoc318B-frozen-source.json';c['expected_base_manifest']='spoc318B-base-files.json';c['expected_image_tag']=image;c['expected_image_id']=image;c['max_attempts']=1;setarg(c['args'],'--devices','0,1,4,5');setarg(c['args'],'--wait',28800)
 c['requires']=[dict(file='spoc318B-stage.json'),dict(file='spoc318B-model-runtime-equivalence.json'),dict(file='../online-background/b167-v308-aux3card-compact16-step128-gsm384-TV.json',greedy=True)]+c['requires']
code=r'''from pathlib import Path
import json,hashlib,subprocess,os
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');s=b/'oneserving-v109-b99e118';r=s/'online-studies';assert not r.exists(),'private destination alreadyexists; inspect before proceeding';image=P['image'];assert json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]['Id']==image
native={n:h for n,h in P['bundle']['source'].items()if n.startswith('pypto-serving-')}
for n,h in native.items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
model=Path('/data/disk/models/Qwen3.8-27B');metadata={n:hashlib.sha256((model/n).read_bytes()).hexdigest()for n in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','generation_config.json']};assert metadata==P['base_metadata']
head=s/'kl-prewarm-suffix-tp2';assert json.loads((head/'manifest.json').read_text())['step']==128
for rank,h in enumerate(P['cases'][1]['expected_initial_head_shards']):assert hashlib.sha256((head/f'rank{rank}.safetensors').read_bytes()).hexdigest()==h
r.mkdir();(r/'assets').mkdir();src=r/'src-dense-v292';src.mkdir();extra=P['bundle']['extras'];original_entry=extra['docker_entry_v292.py'];assert 'os.chown(v,1015,1016)'in original_entry;extra['docker_entry_v292.py']=original_entry.replace('os.chown(v,1015,1016)',f'os.chown(v,{os.getuid()},{os.getgid()})')
old='quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc';assert old in extra['run_dense_v292_docker.py'];extra['run_dense_v292_docker.py']=extra['run_dense_v292_docker.py'].replace(old,image)
for n,t in P['bundle']['files'].items():p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t)
for n,t in extra.items():p=r/n;p.parent.mkdir(parents=True,exist_ok=True);compile(t,n,'exec');p.write_text(t)
for n,t in P['bundle']['assets'].items():(r/'assets'/n).write_text(t)
source=dict(P['bundle']['source']);source['docker_entry_v292.py']=hashlib.sha256(extra['docker_entry_v292.py'].encode()).hexdigest();source['run_dense_v292_docker.py']=hashlib.sha256(extra['run_dense_v292_docker.py'].encode()).hexdigest();weight_stats={}
for n in sorted(set(json.loads((model/'model.safetensors.index.json').read_text())['weight_map'].values())):
 st=(model/n).stat();weight_stats[n]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
base=dict(model=str(model),metadata_sha256=metadata,weight_files_stat=weight_stats);proof=dict(completed=True,NN_execution=False,source_from='A frozen292',compute_package_and_benchmark_byte_exact=True,native_source_byte_exact=True,initial_head_rehashed=True,image_id=image,ownership_uid=os.getuid(),ownership_gid=os.getgid(),only_source_differences=['Docker tag pinned to same actual digest','new result ownership uses B login UID/GID'],extra_files={n:hashlib.sha256(t.encode()).hexdigest()for n,t in extra.items()if n not in source},gate_before_reserved_NN=True,A745_not_stopped=True)
for n,v in [('spoc318B-stage.json',proof),('spoc318B-frozen-source.json',dict(files=source)),('spoc318B-base-files.json',base),('queue-spoc318B.json',dict(cases=P['cases'],automatic_reserved=False,frozen_confirmation=True))]:
 with(r/n).open('x')as f:json.dump(v,f,indent=2)
x=subprocess.run(['python3',str(r/'run_dense_v292_docker.py'),'--help'],capture_output=True,text=True);assert x.returncode==0,x.stderr
with(r/'queue-spoc318B.json.controller.log').open('x')as log:p=subprocess.Popen(['python3','study_queue_spoc312.py','--plan','queue-spoc318B.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,stage=proof,source=source,base=base)))
'''
payload=dict(bundle=bundle,cases=[gate,formal],image=image,base_metadata=json.loads((D/'spoc312/base-files.json').read_text())['metadata_sha256']);wire='P='+repr(payload)+'\n'+code;wrapped='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(wire.encode())).decode())+')))'
x=subprocess.run([*ssh,'wangzili@189.1.227.167','python3 -'],input=wrapped.encode(),capture_output=True,timeout=120);(A/'spoc318B-stage-output.txt').write_bytes(x.stdout+x.stderr);assert x.returncode==0,x.stderr.decode()[-2500:];out=json.loads(x.stdout);(Q/'stage.json').write_text(json.dumps(out,indent=2)+'\n');(Q/'queue-spoc318B.json').write_text(json.dumps(dict(cases=[gate,formal],automatic_reserved=False,frozen_confirmation=True),indent=2)+'\n');print(json.dumps(dict(pid=out['pid'],waiting_for='full model/runtime equivalence proof, then existing B auxiliary study finish; A745 unchanged, B reserved NN blocked pending handoff proof')))
