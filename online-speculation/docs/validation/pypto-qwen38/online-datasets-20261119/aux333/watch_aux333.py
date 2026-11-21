"""只读收集同配方三卡辅助对照；不调参、不派发新题。"""
from pathlib import Path
import argparse,json,zlib,subprocess,hashlib,gzip,importlib.util,time,traceback
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();A,D=a.artifacts,a.validation_root;Q=D/'aux333';plan=json.loads((Q/'queue-spoc333aux.json').read_text());launch=json.loads((Q/'launch.json').read_text());asset_hash=launch['assets'];cache={}
sp=importlib.util.spec_from_file_location('accounting314',D/'rows255/audit_accounting.py');acc=importlib.util.module_from_spec(sp);sp.loader.exec_module(acc)
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@159.138.5.111','python3 -']
read=r'''from pathlib import Path
import json,zlib,hashlib,sys
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');state=json.loads((r/'queue-spoc333aux.status.json').read_text())if(r/'queue-spoc333aux.status.json').exists()else{};proc=Path('/proc')/str(P['pid'])/'cmdline';live=proc.exists()and'study_queue_spoc312.py'in proc.read_bytes().replace(bytes([0]),b' ').decode();out=dict(controller_live=live,status=state,cases=[])
for case in P['cases']:
 tag=case['tag'];p=r/(tag+'.json');d=json.loads(p.read_text())if p.exists()else{};entry=next((x for x in state.get('cases',[])if x['tag']==tag),{});error=entry.get('error');terminal=bool(entry.get('completed')or error)
 if not live and not terminal:error='Controller missing before complete status';terminal=True
 item=dict(tag=tag,terminal=terminal,completed=bool(entry.get('completed')and d.get('completed')and d.get('clean_shutdown')),error=error,stage=d.get('stage'),progress=d.get('progress',{}))
 if terminal:
  item['files']={tag+ext:(r/(tag+ext)).read_text()for ext in ['.json','.source.json','.log']if(r/(tag+ext)).exists()};item['actual_shards']={}
  if item['completed']:
   for k,v in d['results'].items():
    if v.get('exported_head'):
     x=v['exported_head'];item['actual_shards'][k]={f['name']:hashlib.sha256((Path(x['path'])/f['name']).read_bytes()).hexdigest()for f in x['manifest']['rank_files']}
  else:item['controller_log_tail']=(r/'queue-spoc333aux.json.controller.log').read_text()[-3000:]
 out['cases'].append(item)
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
def audit(case,item):
 assert item['completed']and not item.get('error'),item.get('error');tag=case['tag'];d=json.loads(item['files'][tag+'.json']);args=case['args'];opt=lambda k,default=None:args[args.index(k)+1]if k in args else default;stride=int(opt('--strides'));gate='--audit-replay'in args
 proof=acc.audit(d,stride,layout='separate_streams',allow_replay=gate)
 assert d['physical_device_count']==3 and len(set(d['visible_devices'].split(',')))==3 and d['device_ids']==[0,1]and d['learner_device']==2 and d['background']is True
 assert d['head_step']==128 and d['head_shard_sha256']==case['expected_initial_head_shards']and d['optimizer_init']=='fresh'
 for k,v in dict(training_rows=16,max_model_len=1024,last_layers=1,parameter_scope='full',parameterization='dense',draft_width=8,candidates=7,teacher_tokens=7,learning_rate=3e-5,objective='kl',teacher_targets='greedy',temperature=0,paired_physical_rows=16,publication_backend='device',draft_convolution='parallel128',feedback_mode='delayed',feedback_timing='early',teacher_scaling='host',gdn_valid_rows=True).items():assert d[k]==v,(k,d[k],v)
 assert d['requests']==d['ar_checked_requests']==int(opt('--ar-check-requests'))
 for flag,key,cast in [('--tokens','output_tokens_per_request',int),('--publish-every','publish_every',int),('--warm-update-stride','warm_update_stride',int),('--warm-update-steps','warm_update_steps',int),('--warmup-updates','warmup_updates',int),('--seed','seed',int)]:assert d[key]==cast(opt(flag))
 assert d['stop_on_eos']==('--stop-on-eos'in args)
 for flag,key in [('--request-file','prompts_sha256'),('--holdout-file','holdout_sha256')]:assert d[key]==asset_hash[opt(flag)]
 for arm in ['ar','frozen',str(stride)]:assert all(0<n<=d['output_tokens_per_request']for row in d['results'][arm]['request_timeline']for n in row['output_tokens'])
 assert json.loads(item['files'][tag+'.source.json'])['files']==launch['source']
 on=d['results'][str(stride)];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==128+on['published_version'];assert on['published_version']==d['publish_every']*on['stream']['stats']['publications'];assert item['actual_shards'][str(stride)]=={x['name']:x['sha256']for x in m['rank_files']}
 replays=[x for u in on['updates']for x in u.get('replay_audits',[])]
 if gate:assert replays and all(x['served_replay_tv']<=.01 and x['relative_l2']<=.01 for x in replays)
 proof.update(completed=True,tag=tag,physical_cards=3,tp=2,learner_device=2,primary_qualification=False,raw_sha256=hashlib.sha256(item['files'][tag+'.json'].encode()).hexdigest(),source_initial_head_config_assets_checked=True,actual_published_shards=item['actual_shards'][str(stride)],replay_audits=replays,scope='Three-card auxiliary, separate-stream timing; all256 already assigned primary requests when used. Not fresh confirmation or a replacement for two-card proof. Physical target pair recorded per run; no causal two-versus-three-card attribution without matching hardware/timing.')
 return proof

while len(cache)<len(plan['cases']):
 try:
  pending=[c for c in plan['cases']if c['tag']not in cache];x=subprocess.run(ssh,input=('P='+repr(dict(pid=launch['pid'],cases=pending))+'\n'+read).encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode()[-1500:];reply=json.loads(zlib.decompress(x.stdout));obs=[]
  for item in reply['cases']:
   tag=item['tag'];obs.append({k:v for k,v in item.items()if k not in ['files','actual_shards']})
   if not item['terminal']:continue
   for n,s in item.get('files',{}).items():
    b=s.encode();suffix='.gz'if len(b)>2<<20 else'';(Q/(n+suffix)).write_bytes(gzip.compress(b,mtime=0)if suffix else b)
   try:proof=audit(next(c for c in pending if c['tag']==tag),item)
   except Exception as e:proof=dict(completed=False,tag=tag,error=repr(e),traceback=traceback.format_exc(),controller_log_tail=item.get('controller_log_tail'))
   cache[tag]=proof;(Q/(tag+'.audit.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
  (A/'aux333-watcher-state.json').write_text(json.dumps(dict(completed=len(cache)==len(plan['cases']),controller_live=reply['controller_live'],results=list(cache.values()),observations=obs,automatic_reserved=False),indent=2)+'\n')
 except (subprocess.TimeoutExpired,AssertionError,ValueError,zlib.error)as e:print(json.dumps(dict(read_error=repr(e))),flush=True)
 if len(cache)<len(plan['cases']):time.sleep(30)
proof=dict(completed=all(x['completed']for x in cache.values()),cases=list(cache.values()),automatic_reserved=False)
(Q/'exploration-audit.json').write_text(json.dumps(proof,indent=2)+'\n');(A/'aux333-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
