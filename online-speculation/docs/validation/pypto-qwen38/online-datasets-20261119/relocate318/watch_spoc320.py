"""冻结配方的三次新题确认；按已证明的主机放置只读收集，不调参或派发更多题。"""
from pathlib import Path
import argparse,json,subprocess,zlib,time,hashlib,importlib.util,traceback,random,gzip
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();A,D=a.artifacts,a.validation_root;Q=D/'spoc312';Q.mkdir(exist_ok=True);freeze=json.loads((Q/'frozen-recipe.json').read_text());launch=json.loads((Q/'launch.json').read_text());base=json.loads((D/'context273/spoc283/spoc283-confirm256.json').read_text());selection=json.loads((D/'context273/spoc283/spoc283-selection-manifest.json').read_text());cache={};targets=[dict(tag=x['tag'],pid=x['pid'],seed=seed)for x,seed in zip(launch['launched'],[743,744,745])];ssh=['ssh','-o','ControlPath=none','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=10']

placement_path=D/'relocate318/placement.json'
assert placement_path.exists(), 'No completed operational handoff; use original A observer'
placement=json.loads(placement_path.read_text());stage=json.loads((D/'relocate318'/placement['stage_manifest']).read_text())
assert placement['completed'] and placement['original_A_waiting_only_stopped'] and placement['no_reserved_NN_started_on_A']
assert json.loads((D/'relocate318/model-runtime-equivalence.json').read_text())['passed']
assert json.loads((D/'relocate318/extended-runtime-equivalence.json').read_text())['passed']
assert json.loads((D/'relocate318/boundary-audit.json').read_text())['completed']
changes={k for k in set(freeze['source_files'])|set(stage['source']) if freeze['source_files'].get(k)!=stage['source'].get(k)}
assert changes=={'run_dense_v292_docker.py','docker_entry_v292.py'}, changes
for target in targets:
 target.update(host='159.138.5.111', status='queue-spoc312-confirm'+str(target['seed'])+'.status.json', base='spoc312-base-files.json', image='quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc', controller_log='queue-spoc312-confirm'+str(target['seed'])+'.json.controller.log')
 if target['seed']==745:
  target.update(host='189.1.227.167',pid=stage['pid'],status=placement['controller_status'],base='spoc318B-base-files.json',image=freeze['expected_image_id'],controller_log=placement['controller_log'])

read=r'''from pathlib import Path
import json,hashlib,zlib,sys,subprocess
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');p=r/(P['tag']+'.json');d=json.loads(p.read_text())if p.exists()else{};status=r/P['status'];state=json.loads(status.read_text())if status.exists()else{};c=next((x for x in state.get('cases',[])if x['tag']==P['tag']),{});proc=Path('/proc')/str(P['pid'])/'cmdline';alive=proc.exists()and 'study_queue_spoc312.py'in proc.read_bytes().replace(bytes([0]),b' ').decode();error=c.get('error');terminal=bool(c.get('completed')or error)
if not alive and not terminal:error='Controller missing before complete status';terminal=True
out=dict(tag=P['tag'],stage=d.get('stage'),completed=bool(c.get('completed')and d.get('completed')and d.get('clean_shutdown')),error=error,terminal=terminal,controller_live=alive,progress={k:(v[-1]if isinstance(v,list)and v else v)for k,v in d.get('progress',{}).items()})
if terminal:
 out['files']={P['tag']+ext:(r/(P['tag']+ext)).read_text()for ext in ['.json','.source.json','.log']if(r/(P['tag']+ext)).exists()};out['controller_status']=state
 if not out['completed']:out['controller_log_tail']=(r/P['controller_log']).read_text()[-2500:]
 if out['completed']:
  out['actual_published_shards']={};on=d['results']['64']
  for part in on['exported_head']['manifest']['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
   out['actual_published_shards'][part['name']]=h.hexdigest()
  out['asset_text']=(r/'assets'/('spoc312-confirm-order'+str(P['seed'])+'.json')).read_text();binding=json.loads((r/P['base']).read_text());model=Path(binding['model']);current=dict(metadata_sha256={n:hashlib.sha256((model/n).read_bytes()).hexdigest()for n in binding['metadata_sha256']},weight_files_stat={})
  for n in binding['weight_files_stat']:
   st=(model/n).stat();current['weight_files_stat'][n]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
  out['current_base_binding']=current;out['current_revision_marker_sha256']=hashlib.sha256((model/'.cache/huggingface/download/config.json.metadata').read_bytes()).hexdigest();out['current_image_id']=json.loads(subprocess.check_output(['docker','image','inspect',P['image']],text=True))[0]['Id']
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
for n in ['audit_accounting','_study_metrics_windows209']:
 path=D/('rows255/audit_accounting.py'if n=='audit_accounting'else'windows209/_study_metrics_windows209.py');sp=importlib.util.spec_from_file_location(n,path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 if n=='audit_accounting':accounting=mod
 else:metrics=mod

def check(t,item):
 assert item['completed']and not item.get('error'),item.get('error');d=json.loads(item['files'][t['tag']+'.json']);proof=accounting.audit(d,64);assert d['requests']==d['ar_checked_requests']==256 and d['physical_device_count']==2 and len(set(d['visible_devices'].split(',')))==2 and len(d['device_ids'])==2 and d['learner_device']==0 and not d['background'];assert d['head_step']==128 and d['head_shard_sha256']==freeze['initial_head_shards']and d['optimizer_init']=='fresh'
 opts=dict(training_rows=16,max_model_len=1024,output_tokens_per_request=192,last_layers=1,parameter_scope='full',parameterization='dense',draft_width=8,candidates=7,teacher_tokens=7,objective='kl',learning_rate=3e-5,warmup_updates=8,warm_update_stride=16,warm_update_steps=64,lr_schedule='constant',replay_blocks=1,publish_every=4,inner_steps=1,teacher_targets='greedy',teacher_temperature=1,teacher_scaling='host',feedback_mode='delayed',feedback_timing='early',publication_backend='device',temperature=0,stop_on_eos=True,paired_physical_rows=16,draft_convolution='parallel128',gdn_valid_rows=True)
 for k,v in opts.items():assert d[k]==v,(k,d[k],v)
 assert d['seed']==t['seed'];order=list(range(256));random.Random(t['seed']).shuffle(order);text=item['asset_text'];assert json.loads(text)==[base[i]for i in order]and hashlib.sha256(text.encode()).hexdigest()==d['prompts_sha256'];assert d['holdout_sha256']==freeze['holdout_sha256'];
 for arm in ['ar','frozen','64']:
  rows=d['results'][arm]['request_timeline'];assert all(0<n<=192 for row in rows for n in row['output_tokens'])
 assert json.loads(item['files'][t['tag']+'.source.json'])['files']==(stage['source'] if t['seed']==745 else freeze['source_files']);assert item['controller_status']['controller_sha256']==hashlib.sha256((Q/'study_queue_spoc312.py').read_bytes()).hexdigest();binding=stage['base'] if t['seed']==745 else json.loads((Q/'base-files.json').read_text());assert item['current_base_binding']=={k:binding[k]for k in ['metadata_sha256','weight_files_stat']}
 assert item['current_image_id']==freeze['expected_image_id']
 if t['seed']==745:assert item['current_revision_marker_sha256']==stage['marker_restoration']['sha256']
 on=d['results']['64'];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==128+on['published_version']and on['optimizer_version']==len(on['updates']);assert on['published_version']==4*on['stream']['stats']['publications'];assert item['actual_published_shards']=={p['name']:p['sha256']for p in m['rank_files']};proof.update(completed=True,tag=t['tag'],seed=t['seed'],execution_host=t['host'],operational_relocation=t['seed']==745,raw_sha256=hashlib.sha256(item['files'][t['tag']+'.json'].encode()).hexdigest(),frozen_source_head_config_and_budget_checked=True,base_files_and_initial_dispatch_guard_checked=True,actual_published_shards=item['actual_published_shards']);return proof,d
while len(cache)<3:
 observations=[]
 for target in targets:
  if target['tag']in cache:continue
  try:
   x=subprocess.run([*ssh,'wangzili@'+target['host'],'python3 -'],input=('P='+repr(target)+'\n'+read).encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode()[-1200:];item=json.loads(zlib.decompress(x.stdout));observations.append({k:v for k,v in item.items()if k not in ['files','asset_text']})
   if item['terminal']:
    for n,s in item.get('files',{}).items():
     b=s.encode();suffix='.gz'if len(b)>2<<20 else'';(Q/(n+suffix)).write_bytes(gzip.compress(b,mtime=0)if suffix else b)
    try:proof,d=check(target,item);cache[target['tag']]=dict(proof=proof,raw=d)
    except BaseException as e:proof=dict(completed=False,tag=target['tag'],error=repr(e),traceback=traceback.format_exc(),controller_status=item.get('controller_status'),controller_log_tail=item.get('controller_log_tail'));cache[target['tag']]=dict(proof=proof)
    (Q/(target['tag']+'.audit.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
  except (subprocess.TimeoutExpired,AssertionError,ValueError,zlib.error)as e:observations.append(dict(tag=target['tag'],read_error=repr(e),terminal=False))
 state=dict(completed=len(cache)==3,terminal_cases=list(cache),observations=observations,results=[v['proof']for v in cache.values()],no_retuning=True)
 (A/'spoc312-mixed320-watcher-state.json').write_text(json.dumps(state,indent=2)+'\n')
 if len(cache)<3:time.sleep(30)
proof=dict(completed=False,cases=[cache[t['tag']]['proof']for t in targets],scope='Supplementary SPoC-derived pseudocode-to-C++ task, NOT original LongProc benchmark. Frozen source292/head128/2A2TP2/rank0/samegeneration budget192C1024, all256samefreshpool shuffled743/744/745 with independenthead+Adam resets; seed745 operationally relocated unstarted A->B with byte-checked model, PyPTO, PTOAS and selected runtime files plus B diagnostic; driver and three CANN-library differences separately recorded, no numerical recipe change; exactsame controls/allcosts/fullcontinuous. No retuning from confirmation.')
try:
 assert all(v['proof']['completed']for v in cache.values());values=[]
 for t in targets:
  d=cache[t['tag']]['raw'];v={arm+'_'+key:d['results'][mode]['stream'][key]for arm,mode in [('ar','ar'),('frozen','frozen'),('online','64')]for key in ['tokens','seconds']};v.update(continuous=True,full_cohort=True,resource_policy_ok=True,performance_run=True);values.append(v)
 proof.update(completed=True,full_audit_passed=True,metrics=metrics.confirmation_metrics(values))
except BaseException as e:proof.update(error=repr(e),traceback=traceback.format_exc())
(Q/'three-confirmation-audit.json').write_text(json.dumps(proof,indent=2)+'\n');(A/'spoc312-three-confirmation-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
