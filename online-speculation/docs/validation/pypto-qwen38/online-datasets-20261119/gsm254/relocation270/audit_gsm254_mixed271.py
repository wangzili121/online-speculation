"""核对两次 A 与一次 B 的冻结配方确认，保留执行位置变更记录。"""
from pathlib import Path
import json,subprocess,zlib,time,hashlib,importlib.util,traceback
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--validation-root',type=Path,required=True);args=parser.parse_args();A=args.artifacts;D=args.validation_root;f=json.loads((A/'gsm254-frozen-recipe.json').read_text());amendment=json.loads((A/'gsm254-relocation270-B-launch.json').read_text())['proof'];targets=[dict(host='159.138.5.111',root='online-studies',tag=f'a111-v248-gsm254-confirm{seed}',seed=seed,version='v248')for seed in [743,744]]+[dict(host='189.1.227.167',root='online-background',tag='b167-v245-gsm254-confirm745',seed=745,version='v245')];cache={};state=A/'gsm254-mixed271-watcher-state.json';ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
read=r'''from pathlib import Path
import json,zlib,sys,hashlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118')/P['root'];p=r/(P['tag']+'.json');d=json.loads(p.read_text())if p.exists()else{};out=dict(completed=d.get('completed')and d.get('clean_shutdown')is True,error=d.get('error'),stage=d.get('stage'))
status=r/('queue-gsm254-B745.status.json'if P['version']=='v245'else 'queue-gsm254-confirm'+str(P['seed'])+'.status.json')
if not out['completed']and not out.get('error')and status.exists():
 stat=json.loads(status.read_text());bad=next((c for c in stat.get('cases',[])if c.get('tag')==P['tag']and c.get('error')),None)
 if stat.get('execution_complete')and bad:out.update(error=bad['error'],launcher_status_text=status.read_text())
if out.get('completed')or d.get('error'):
 out['files']={P['tag']+e:(r/(P['tag']+e)).read_text()for e in ['.json','.source.json']}
 if d.get('completed')and d.get('clean_shutdown'):
  on=d['results']['128'];actual={}
  for part in on['exported_head']['manifest']['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as file:
    for chunk in iter(lambda:file.read(8<<20),b''):h.update(chunk)
   actual[part['name']]=h.hexdigest()
  out['actual_published_file_sha256']=actual
  out['asset_text']=(r/'assets'/('gsm254-confirm-order'+str(P['seed'])+'.json')).read_text()
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
def poll(t):
 x=subprocess.run([*ssh,'wangzili@'+t['host'],'python3 -'],input=('P='+repr(t)+'\n'+read).encode(),capture_output=True,timeout=60)
 if x.returncode:raise RuntimeError(x.stderr.decode()[-1000:])
 return json.loads(zlib.decompress(x.stdout))
while len(cache)<3:
 observations=[]
 for target in targets:
  if target['tag']in cache:continue
  try:
   d=poll(target);observations.append(dict(tag=target['tag'],completed=d.get('completed'),stage=d.get('stage'),error=d.get('error')))
   if d.get('completed')or d.get('error'):cache[target['tag']]=d
  except (subprocess.TimeoutExpired,RuntimeError,ValueError,zlib.error)as e:observations.append(dict(tag=target['tag'],read_error=repr(e),terminal=False))
 state.write_text(json.dumps(dict(stage='waiting_full_three',terminal_cases=list(cache),observations=observations,no_parameter_changes=True),indent=2)+'\n')
 if len(cache)<3:time.sleep(30)
proof=dict(completed=False,execution_amendment=amendment,resource='Each2physicalA2/TP2/learner0; twoA trials andoneBtrial',adjustments_from_confirmation=False,cases=[])
try:
 sp=importlib.util.spec_from_file_location('accounting',D/'rows255/audit_accounting.py');acct=importlib.util.module_from_spec(sp);sp.loader.exec_module(acct);sp=importlib.util.spec_from_file_location('metrics',D/'windows209/_study_metrics_windows209.py');metrics=importlib.util.module_from_spec(sp);sp.loader.exec_module(metrics)
 assignment_a=json.loads((A/'gsm254-relocation270-A-preflight.json').read_text());base=json.loads((A/'gsm254-confirm384.json').read_text());bcase=amendment['case'];acases=json.loads((D/'gsm254/gsm254-frozen-recipe.json').read_text());values=[];raws=[]
 canonical={}
 for name,sha in f['source_files'].items():
  if 'src-dense-v245/'in name:canonical['source/'+name.split('src-dense-v245/',1)[1]]=sha
  elif name=='orthrus_bench_dense_v245.py':canonical['benchmark']=sha
  elif name.startswith('pypto-serving-qwen3.8-27B/'):canonical[name]=sha
 for target in targets:
  item=cache[target['tag']];assert 'files'in item,('Launcher failed before model result',target['tag'],item);d=json.loads(item['files'][target['tag']+'.json']);assert d['completed']and d['clean_shutdown'];assert d['requests']==d['ar_checked_requests']==384 and d['physical_device_count']==2 and d['learner_device']==0 and len(set(d['visible_devices'].split(',')))==len(d['device_ids'])==2;assert d['head_step']==128 and d['head_shard_sha256']==f['initial_head_shards']and d['optimizer_init']=='fresh'and d['background']is False
  assert d['draft_width']==8 and d['candidates']==7 and d['paired_physical_rows']==16 and d.get('training_rows',32)==32 and d['last_layers']==1 and d['parameter_scope']=='full'and d['objective']=='tv'and d['learning_rate']==1e-5 and d['publish_every']==4 and d['teacher_tokens']==7 and d['warm_update_stride']==16 and d['warm_update_steps']==256 and d['warmup_updates']==8 and d['lr_schedule']=='constant'and d['replay_blocks']==1 and d['temperature']==0 and d['stop_on_eos'];assert d['seed']==target['seed'];assert d['holdout_sha256']=='bf6ccf463fa4d7b4cd27a5cdf054cdd911a21bd5dc31d2a8f49100b0fbbb3ba0'
  raw=item['asset_text'];assert hashlib.sha256(raw.encode()).hexdigest()==d['prompts_sha256'];import random;order=list(range(384));random.Random(target['seed']).shuffle(order);assert json.loads(raw)==[base[i]for i in order];source=json.loads(item['files'][target['tag']+'.source.json'])['files'];actual={};needle='src-dense-'+target['version']+'/'
  for name,sha in source.items():
   if needle in name:actual['source/'+name.split(needle,1)[1]]=sha
   elif name=='orthrus_bench_dense_'+target['version']+'.py':actual['benchmark']=sha
   elif name.startswith('pypto-serving-qwen3.8-27B/'):actual[name]=sha
  assert actual==canonical;on=d['results']['128'];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==128+on['published_version']and m['published_updates']==on['published_version'];assert on['optimizer_version']==len(on['updates'])and on['published_version']==on['stream']['stats']['publications']*4;assert item['actual_published_file_sha256']=={p['name']:p['sha256']for p in m['rank_files']};a=acct.audit(d,128);a.update(host=target['host'],seed=target['seed'],raw_sha256=hashlib.sha256(item['files'][target['tag']+'.json'].encode()).hexdigest(),actual_head_step=m['step'],initial_head_and_canonical_source_exact=True,actual_published_files_checked_on_host=True);proof['cases'].append(a);v={arm+'_'+field:d['results'][key]['stream'][field]for arm,key in [('ar','ar'),('frozen','frozen'),('online','128')]for field in ['tokens','seconds']};v.update(continuous=True,full_cohort=True,resource_policy_ok=True,performance_run=True);values.append(v)
  for name,text in item['files'].items():(A/name).write_text(text)
 proof.update(completed=True,full_audit_passed=True,metrics=metrics.confirmation_metrics(values),all_costs=True,continuous=True,scope='Frozen algorithm/config/head/order/source unchanged. Thirdtrial operationally movedbeforeanyNPUuse becauseA lackedidlepair; beforedispatch proof recorded. Eachandpooled strictdual5 unchanged. Ratios pairedwithinhost, no rawcrosshost equality assumed.')
except BaseException as e:proof.update(error=repr(e),traceback=traceback.format_exc())
(A/'gsm254-mixed271-confirmation-audit.json').write_text(json.dumps(proof,indent=2)+'\n');state.write_text(json.dumps(dict(stage='completed_audit',audit_completed=proof['completed'],metrics=proof.get('metrics'),error=proof.get('error')),indent=2)+'\n');print(json.dumps(proof),flush=True)
