"""Stage higher-cadence exploration on old questions only; no reserved-data changes."""
from pathlib import Path
import json,gzip,hashlib,copy,subprocess,base64,zlib
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True);args=p.parse_args();A,D=args.artifacts,args.validation_root;Q=D/'explore314';bundle=json.loads(gzip.decompress((Q/'source303_bundle.json.gz').read_bytes()));old=json.loads(gzip.decompress((D/'aux308/source233_bundle.json.gz').read_bytes()));launcher=old['launcher'].replace('v233','v314')
needle='a = p.parse_args()';assert needle in launcher;launcher=launcher.replace(needle,"p.add_argument('--max-model-len',type=int,choices=(512,1024),default=512)\np.add_argument('--training-rows',type=int,choices=(16,32))\np.add_argument('--clip-output-to-context',action='store_true')\n"+needle)
needle='    if a.gdn_valid_rows:';assert needle in launcher;launcher=launcher.replace(needle,"    command.extend(['--max-model-len',str(a.max_model_len)])\n    if a.training_rows is not None:command.extend(['--training-rows',str(a.training_rows)])\n    if a.clip_output_to_context:command.append('--clip-output-to-context')\n"+needle)
compile(launcher,'run_dense_v314.py','exec');controller=(D/'spoc312/study_queue_spoc312.py').read_text().replace('_study_metrics_windows209','_study_metrics_explore314');metrics=(D/'windows209/_study_metrics_windows209.py').read_text()
basegate=json.loads((D/'context273/budget303/queue-budget303-boundary.json').read_text())['cases'][0];baseLong=json.loads((D/'context273/budget303/queue-length307-longproc.json').read_text())['cases'][0];baseSpoc=next(x for x in json.loads((D/'context273/recovery301/queue-explore301-spoc.json').read_text())['cases']if'kl-after301'in x['tag'])
cases=[];asset_names=set()
def setarg(args,k,v):
 if k in args:args[args.index(k)+1]=str(v)
 else:args.extend([k,str(v)])
def remove(args,k):
 if k in args:i=args.index(k);del args[i:i+2]
for spec,tag,gate in [(basegate,'c61-v314-context1024-boundary-replay',True),(baseLong,'c61-v314-explore-longproc30-kl-constant16-output512',False),(baseSpoc,'c61-v314-explore-spoc64-kl-constant16-output192',False)]:
 c=copy.deepcopy(spec);c.update(tag=tag,launcher='run_dense_v314.py',max_attempts=1,requires=[dict(file='explore314-preflight.json')],expected_source_manifest='explore314-frozen-source.json',expected_base_manifest='explore314-base-files.json',expected_initial_head='kl-prewarm-suffix-tp2',expected_initial_head_shards=['f0308fed571f54be1d75aec6ae33d9a58230ccc1bbcea1d3f4df0d1fab6de610','131649c093812f5d895e304ca0f7655b934bc456c9325a3917b1f9ccdcf0f7fd'],purpose='Old exploration questions only. C host source303 compute bytes unchanged, host-only launcher supports context/output budget. Two physical A2 TP2/rank0, all full-flow controls/costs. Only continuing update cadence changes from staged64 to constant16; no new reserved questions, no auto confirmation.')
 for k in ['confirmation_group','confirm','unless_achieved']:c.pop(k,None)
 args=c['args'];setarg(args,'--devices','0,1,2,3,4,5,6,7');setarg(args,'--wait',28800);setarg(args,'--training-rows',16);setarg(args,'--max-model-len',1024)
 for flag in ['--request-file','--holdout-file']:
  n=args[args.index(flag)+1];asset_names.add(n);setarg(args,flag,'explore314-'+n)
 if not gate:
  setarg(args,'--strides',16);remove(args,'--warm-update-stride');remove(args,'--warm-update-steps');c['requires'].append(dict(file='c61-v314-context1024-boundary-replay.json',audit=True,greedy=True))
 c['minimum_available_host_gib']=96;c['allow_parallel_startup']=False;cases.append(c)
fetch="""from pathlib import Path
import json,zlib,sys
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies/assets');sys.stdout.buffer.write(zlib.compress(json.dumps({n:(r/n).read_text()for n in N}).encode()))
"""
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10'];x=subprocess.run([*ssh,'wangzili@159.138.5.111','python3 -'],input=('N='+repr(sorted(asset_names))+'\n'+fetch).encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode();assets={'explore314-'+n:t for n,t in json.loads(zlib.decompress(x.stdout)).items()}
code=r'''from pathlib import Path
import json,hashlib,subprocess,sys
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');s=b/'oneserving-v109-b99e118';r=s/'online-tp2-test';assert not(r/'explore314-preflight.json').exists()
for n,h in P['bundle']['native'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
model=b/'inference-tuning-20261116/model-view';meta={n:hashlib.sha256((model/n).read_bytes()).hexdigest()for n in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','generation_config.json']};assert meta['config.json']=='191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'and meta['model.safetensors.index.json']=='77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df'
weights={}
for n in sorted(set(json.loads((model/'model.safetensors.index.json').read_text())['weight_map'].values())):
 t=(model/n).stat();weights[n]=dict(size=t.st_size,mtime_ns=t.st_mtime_ns,inode=t.st_ino)
head=s/'kl-prewarm-suffix-tp2';assert json.loads((head/'manifest.json').read_text())['step']==128
for rank,h in enumerate(P['cases'][0]['expected_initial_head_shards']):assert hashlib.sha256((head/f'rank{rank}.safetensors').read_bytes()).hexdigest()==h
source=r/'src-dense-v314';source.mkdir()
for n,t in P['bundle']['files'].items():p=source/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(t)
for n,t in [('run_dense_v314.py',P['launcher']),('orthrus_bench_dense_v314.py',P['bundle']['benchmark']),('study_queue_explore314.py',P['controller']),('_study_metrics_explore314.py',P['metrics'])]:
 compile(t,n,'exec')
 with(r/n).open('x')as f:f.write(t)
for n,t in P['assets'].items():
 with(r/'assets'/n).open('x')as f:f.write(t)
files={'run_dense_v314.py':hashlib.sha256(P['launcher'].encode()).hexdigest(),'orthrus_bench_dense_v314.py':hashlib.sha256(P['bundle']['benchmark'].encode()).hexdigest(),**P['bundle']['native']}
files.update({'online-tp2-test/src-dense-v314/'+n:hashlib.sha256(t.encode()).hexdigest()for n,t in P['bundle']['files'].items()})
base=dict(model=str(model),metadata_sha256=meta,weight_files_stat=weights)
pre=dict(completed=True,NN_execution=False,compute_package_matches_frozen_A303=True,benchmark_matches_frozen_A303=True,native_source_matches=True,initial_head_rehashed=True,host='159.138.54.61',native_driver=subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True).splitlines()[1],scope='C runtime needs own real TP2 boundary/replay/greedy gate before full exploration. Not pooled with A292/A303. No reserved requests or new kernel math.')
for n,v in [('explore314-frozen-source.json',dict(files=files)),('explore314-base-files.json',base),('explore314-preflight.json',pre),('queue-explore314.json',dict(cases=P['cases'],automatic_reserved=False))]:
 with(r/n).open('x')as f:json.dump(v,f,indent=2)
check=subprocess.run(['python3',str(r/'run_dense_v314.py'),'--help'],capture_output=True,text=True);assert check.returncode==0,check.stderr
with(r/'queue-explore314.json.controller.log').open('x')as log:x=subprocess.Popen(['python3','study_queue_explore314.py','--plan','queue-explore314.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=x.pid,preflight=pre,base=base,source=files)))
'''
wire='P='+repr(dict(bundle=bundle,launcher=launcher,controller=controller,metrics=metrics,assets=assets,cases=cases))+'\n'+code
wrapped='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(wire.encode())).decode())+')))'
x=subprocess.run([*ssh,'wangzili@159.138.54.61','python3 -'],input=wrapped.encode(),capture_output=True,timeout=120);(A/'explore314-launch-output.txt').write_bytes(x.stdout+x.stderr);assert x.returncode==0,x.stderr.decode()[-2500:];result=json.loads(x.stdout)
for n,v in [('launch.json',result),('queue-explore314.json',dict(cases=cases,automatic_reserved=False))]: (Q/n).write_text(json.dumps(v,indent=2)+'\n')
for n,t in [('run_dense_v314.py',launcher),('study_queue_explore314.py',controller),('_study_metrics_explore314.py',metrics)]: (Q/n).write_text(t)
print(json.dumps(dict(pid=result['pid'],preflight=result['preflight'])))
