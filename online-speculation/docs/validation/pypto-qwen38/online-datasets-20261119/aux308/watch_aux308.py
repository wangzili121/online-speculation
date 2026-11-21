"""只读核对独立学习卡实验；始终标为三卡辅助结果。"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, subprocess, time, zlib, gzip, traceback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True)
a=p.parse_args();A,D=a.artifacts,a.validation_root;Q=D/'aux308'
plan=json.loads((Q/'queue-aux308-threecard.json').read_text());pre=json.loads((Q/'aux308-preflight.json').read_text());bundle=json.loads((Q/'bundle-manifest.json').read_text())
sp=importlib.util.spec_from_file_location('accounting308',D/'rows255/audit_accounting.py');acc=importlib.util.module_from_spec(sp);sp.loader.exec_module(acc)
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -']
remote=r'''from pathlib import Path
import json,zlib,sys,hashlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-background');state=json.loads((r/'queue-aux308-threecard.status.json').read_text());out={'status':state,'cases':[]}
for case in P:
 tag=case['tag'];path=r/(tag+'.json');d=json.loads(path.read_text())if path.exists()else{};s=next((x for x in state['cases']if x['tag']==tag),{});terminal=bool(s.get('completed')or s.get('error'));item=dict(tag=tag,terminal=terminal,completed=bool(s.get('completed')and d.get('completed')and d.get('clean_shutdown')),error=s.get('error'),stage=d.get('stage'),progress=d.get('progress',{}))
 if terminal:
  item['files']={tag+ext:(r/(tag+ext)).read_text()for ext in ['.json','.source.json','.log']if(r/(tag+ext)).exists()};item['actual_shards']={}
  for key,v in d.get('results',{}).items():
   if v.get('exported_head'):
    x=v['exported_head'];item['actual_shards'][key]={f['name']:hashlib.sha256((Path(x['path'])/f['name']).read_bytes()).hexdigest()for f in x['manifest']['rank_files']}
 out['cases'].append(item)
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
def audit(case,item):
 assert item['completed']and not item.get('error'),item.get('error');tag=case['tag'];d=json.loads(item['files'][tag+'.json']);args=case['args'];opt=lambda flag,default=None:args[args.index(flag)+1]if flag in args else default;stride=int(opt('--strides'));smoke='--audit-replay'in args
 v=acc.audit(d,stride,layout='separate_streams',allow_replay=smoke)
 assert d['physical_device_count']==3 and len(set(d['visible_devices'].split(',')))==3 and d['device_ids']==[0,1]and d['learner_device']==2 and d['background']is True
 step=135 if 'domain2'in tag else 128
 assert d['head_step']==step and d['head_shard_sha256']==[x['sha256']for x in pre['heads'][str(step)]]and d['optimizer_init']=='fresh'
 for k,value in dict(last_layers=1,parameter_scope='full',parameterization='dense',draft_width=8,candidates=7,teacher_tokens=7,paired_physical_rows=16,publication_backend='device',temperature=0,stop_on_eos=True,feedback_mode='delayed',teacher_targets='greedy',draft_convolution='parallel128').items():assert d[k]==value,(k,d[k])
 for flag,key,cast in [('--learning-rate','learning_rate',float),('--objective','objective',str),('--tokens','output_tokens_per_request',int),('--publish-every','publish_every',int),('--warm-update-stride','warm_update_stride',int),('--warm-update-steps','warm_update_steps',int),('--warmup-updates','warmup_updates',int)]:assert d[key]==cast(opt(flag))
 for flag,key in [('--request-file','prompts_sha256'),('--holdout-file','holdout_sha256')]:assert d[key]==pre['assets'][opt(flag).removeprefix('aux308-')]
 source=json.loads(item['files'][tag+'.source.json'])['files'];prefix='online-background/src-dense-v308/'
 assert {k.removeprefix(prefix):v for k,v in source.items()if k.startswith(prefix)}==pre['source_files']
 assert {k:v for k,v in source.items()if k.startswith('pypto-serving-')}==bundle['native_source']
 assert source['orthrus_bench_dense_v308.py']==bundle['benchmark_sha256'];assert source['run_dense_v308.py']==hashlib.sha256((Q/'run_dense_v308.py').read_bytes()).hexdigest()
 on=d['results'][str(stride)];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==step+on['published_version'];assert on['published_version']==d['publish_every']*on['stream']['stats']['publications'];assert item['actual_shards'][str(stride)]=={x['name']:x['sha256']for x in m['rank_files']}
 audits=[x for u in on['updates']for x in u.get('replay_audits',[])]
 if smoke:assert len(audits)==len(on['updates'])and all(x['served_replay_tv']<=.01 and x['relative_l2']<=.01 for x in audits)
 v.update(completed=True,tag=tag,raw_sha256=hashlib.sha256(item['files'][tag+'.json'].encode()).hexdigest(),source_initial_head_and_plan_checked=True,actual_published_shards=item['actual_shards'][str(stride)],replay_audits=len(audits),physical_cards=3,tp=2,learner_device=2,primary_qualification=False,time_drift_caveat='Separate streams, not interleaved with controls; auxiliary exploration only.')
 return v
cache={}
while len(cache)<len(plan['cases']):
 try:
  pending=[x for x in plan['cases']if x['tag']not in cache];r=subprocess.run(ssh,input=('P='+repr(pending)+'\n'+remote).encode(),capture_output=True,timeout=60);assert r.returncode==0,r.stderr.decode()[-1500:];state=json.loads(zlib.decompress(r.stdout));obs=[]
  for item in state['cases']:
   tag=item['tag'];obs.append({k:v for k,v in item.items()if k not in ['files','actual_shards']})
   if not item['terminal']:continue
   for n,s in item['files'].items():
    b=s.encode();suffix='.gz'if len(b)>2<<20 else'';(Q/(n+suffix)).write_bytes(gzip.compress(b,mtime=0)if suffix else b)
   (Q/(tag+'.actual-shards.json')).write_text(json.dumps(item['actual_shards'],indent=2)+'\n')
   try:proof=audit(next(x for x in pending if x['tag']==tag),item)
   except Exception as e:proof=dict(completed=False,tag=tag,error=repr(e),traceback=traceback.format_exc())
   cache[tag]=proof;(Q/(tag+'.audit.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
  state=dict(completed=len(cache)==len(plan['cases']),results=list(cache.values()),observations=obs,controller=state['status'],primary_qualification=False)
  (A/'aux308-watcher-state.json').write_text(json.dumps(state,indent=2)+'\n')
 except (subprocess.TimeoutExpired,AssertionError,ValueError,zlib.error)as e:print(json.dumps(dict(read_error=repr(e))),flush=True)
 if len(cache)<len(plan['cases']):time.sleep(30)
summary=dict(completed=all(v['completed']for v in cache.values()),primary_qualification=False,physical_cards=3,cases=list(cache.values()))
(Q/'auxiliary-audit.json').write_text(json.dumps(summary,indent=2)+'\n');(A/'aux308-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
