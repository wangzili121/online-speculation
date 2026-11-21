"""Audit all three frozen confirmations; no parameter or source changes."""
from pathlib import Path
import json,time,hashlib,math,importlib.util
r=Path(__file__).parent
while not(r/'gsm254-cases.json').exists():time.sleep(5)
plan=json.loads((r/'gsm254-cases.json').read_text());assignment=json.loads((r.parent/'reserved-gsm254-assignment.json').read_text());f=assignment['recipe'];cases=plan['cases'];results=[]
for c in cases:
 p=r/(c['tag']+'.json');status=r/f'queue-gsm254-confirm{c["seed"]}.status.json'
 while True:
  if p.exists():
   try:d=json.loads(p.read_text())
   except ValueError:time.sleep(1);continue
   if d.get('completed')or d.get('error'):break
  if status.exists():
   s=json.loads(status.read_text())
   if s.get('execution_complete')and any(x.get('error')for x in s['cases']):d=dict(error=s['cases'][0]['error'],completed=False);break
  time.sleep(5)
 results.append(d)
proof=dict(completed=False,recipe=f['recipe_id'],entire_requests_per_flow=384,independent_resets=3,resource='Eachtrial2physicalA2/TP2/rank0; noextra learner',cases=[])
try:
 assert hashlib.sha256((r/'_study_metrics_windows209.py').read_bytes()).hexdigest()=='7756d4690a8cfdf1f5a76113ff14d3b01f4b4523b495670235850a293e6876ab';spec=importlib.util.spec_from_file_location('metrics',r/'_study_metrics_windows209.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics);values=[];base=json.loads((r/'assets/gsm254-confirm384.json').read_text());assert len(base)==len(set(base))==384
 for c,d in zip(cases,results):
  assert d.get('completed')and d.get('clean_shutdown'),(c['tag'],d.get('error'));assert d['requests']==d['ar_checked_requests']==384 and d['physical_device_count']==2 and d['learner_device']==0 and len(d['device_ids'])==len(set(d['visible_devices'].split(',')))==2;assert d['head_step']==128 and d['head_shard_sha256']==f['initial_head_shards']and d['optimizer_init']=='fresh';assert d['max_updates']is None and d['background']is False and not d['diagnostic_only']and not d['audit_replay'];assert d['draft_width']==8 and d['candidates']==7 and d['paired_physical_rows']==16 and d['last_layers']==1 and d['parameter_scope']=='full'and d['objective']=='tv'and d['learning_rate']==1e-5 and d['publish_every']==4 and d['teacher_tokens']==7 and d['warm_update_stride']==16 and d['warm_update_steps']==256 and d['warmup_updates']==8 and d['lr_schedule']=='constant'and d['replay_blocks']==1 and d['temperature']==0 and d['stop_on_eos']
  assert d['prompts_sha256']==c['prompts_sha256'];assert d['holdout_sha256']==assignment['audit']['sets']['gsm254-holdout32']['sha256'];asset=r/'assets'/c['args'][c['args'].index('--request-file')+1];assert json.loads(asset.read_text())==[base[i]for i in c['original_problem_order']];assert sorted(c['original_problem_order'])==list(range(384));source=json.loads((r/(c['tag']+'.source.json')).read_text())['files'];canon={}
  for name,sha in source.items():
   if 'src-dense-v248/'in name:canon['source/'+name.split('src-dense-v248/',1)[1]]=sha
   elif name=='orthrus_bench_dense_v248.py':canon['benchmark']=sha
   elif name.startswith('pypto-serving-qwen3.8-27B/'):canon[name]=sha
  assert canon==assignment['canonical_computational_source'];assert source['run_dense_v248_docker.py']==assignment['delivery_launcher_sha256']and source['docker_entry_v248.py']==assignment['delivery_entry_sha256'];on=d['results']['128'];u=on['updates'];assert len(u)>=4 and on['exact_greedy']and on['learned_replay']['exact_greedy'];assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])and x['gradient_norm']>=0 for x in u)and any(x['gradient_norm']>0 for x in u);assert on['update_schedule']['policy']=='continuous_two_stage';m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['published_updates']==on['published_version']and m['step']==128+on['published_version'];assert on['published_version']==on['stream']['stats']['publications']*4 and on['optimizer_version']==len(u);assert on['stream']['tokens']==d['results']['ar']['stream']['tokens']==d['results']['frozen']['stream']['tokens'];assert metrics.matched_request_timelines(384,2,d['results']['ar']['request_timeline'],d['results']['frozen']['request_timeline'],on['request_timeline'])
  for arm in ['ar','frozen']:
   assert all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])
  assert all(on['holdout']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])
  for part in m['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as fi:
    for chunk in iter(lambda:fi.read(8<<20),b''):h.update(chunk)
   assert h.hexdigest()==part['sha256']
  v={arm+'_'+field:d['results'][key]['stream'][field]for arm,key in [('ar','ar'),('frozen','frozen'),('online','128')]for field in ['tokens','seconds']};v.update(continuous=True,full_cohort=True,resource_policy_ok=True,performance_run=True);values.append(v);proof['cases'].append(dict(tag=c['tag'],seed=c['seed'],requests=384,ar_tok_s=d['results']['ar']['stream']['tok_s'],fixed_tok_s=d['results']['frozen']['stream']['tok_s'],online_tok_s=on['stream']['tok_s'],online_vs_ar=on['net_increment_vs_ar'],online_vs_fixed=on['net_increment'],updates=len(u),publications=on['stream']['stats']['publications'],actual_head_step=m['step'],final_seen_tok_s=on['learned_replay']['stream']['tok_s'],fresh_holdout_tok_s=on['holdout']['tok_s'],initial_holdout_tok_s=d['results']['frozen']['holdout']['tok_s'],greedy_AR_exact=True,published_export_exact=True,initial_head_and_source_exact=True,raw_sha256=hashlib.sha256((r/(c['tag']+'.json')).read_bytes()).hexdigest()))
 proof.update(completed=True,full_audit_passed=True,metrics=metrics.confirmation_metrics(values),all_costs=True,continuous=True,no_adjustments_from_confirmation=True,scope='Three full384 trulyunused cohortsamequestions/reordered, independentfreshAdam/head128, fixedalllearningargs andcanonicalbyteidentical computationalcode. Actual2cards pertrial, all3andpooled strictdual5. Seenfrozen andfresh32holdout separately recorded.')
except BaseException as e:proof['error']=repr(e)
(r/'gsm254-confirmation-audit-v2.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
