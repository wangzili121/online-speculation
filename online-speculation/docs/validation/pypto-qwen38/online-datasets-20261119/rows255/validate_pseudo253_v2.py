from pathlib import Path
import json,time,math,hashlib,importlib.util
r=Path(__file__).parent;plan=json.loads((r/'queue-pseudo253-confirm.json').read_text());frozen=json.loads((r/'pseudo253-frozen-recipe.json').read_text());cases=plan['cases'];rows=[]
for c in cases:
 p=r/(c['tag']+'.json')
 while True:
  if p.exists():
   try:d=json.loads(p.read_text())
   except ValueError:time.sleep(1);continue
   if d.get('completed')or d.get('error'):break
  time.sleep(5)
 rows.append(d)
proof=dict(completed=False,recipe=frozen['recipe_id'],independent_resets=3,resources='2physicalA2/TP2/rank0noextra',entire_requests_per_flow=24,all_costs=True,continuous=True,first_reserved_assignment=True,adjustments_from_confirmation=False,cases=[])
try:
 assert hashlib.sha256((r/'_study_metrics_windows209.py').read_bytes()).hexdigest()=='7756d4690a8cfdf1f5a76113ff14d3b01f4b4523b495670235850a293e6876ab';spec=importlib.util.spec_from_file_location('metrics',r/'_study_metrics_windows209.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics);values=[]
 for c,d in zip(cases,rows):
  assert d.get('completed')and d.get('clean_shutdown'),(c['tag'],d.get('error'));assert d['requests']==d['ar_checked_requests']==24 and d['physical_device_count']==2 and d['learner_device']==0 and len(d['device_ids'])==2 and len(set(d['visible_devices'].split(',')))==2;assert d['head_step']==135 and d['head_shard_sha256']==[x['sha256']for x in frozen['initial_head_rank_files']];assert d['optimizer_init']=='fresh'and d['max_updates']is None and d['background']is False and not d['diagnostic_only']and not d['audit_replay'];assert d['draft_width']==8 and d['candidates']==7 and d['paired_physical_rows']==16 and d['last_layers']==1 and d['parameter_scope']=='full'and d['objective']=='kl'and d['learning_rate']==3e-5 and d['publish_every']==4 and d['teacher_tokens']==7 and d['warm_update_stride']==16 and d['warm_update_steps']==64 and d['warmup_updates']==8 and d['lr_schedule']=='constant'and d['replay_blocks']==1 and d['temperature']==0
  assert d['prompts_sha256']==c['prompts_sha256'];assert d['holdout_sha256']=='8eb865531a386e75d6e9d1a6a5d96c79223bb6ed2730ce0a6c661685983fc922';source=json.loads((r/(c['tag']+'.source.json')).read_text());assert source['files']==frozen['source_files']['files']
  on=d['results']['64'];u=on['updates'];assert len(u)>=4 and on['exact_greedy']and on['learned_replay']['exact_greedy'];assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])and x['gradient_norm']>0 for x in u);assert on['update_schedule']['policy']=='continuous_two_stage';m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['published_updates']==on['published_version']and m['step']==d['head_step']+on['published_version'];assert on['published_version']==on['stream']['stats']['publications']*4 and on['optimizer_version']==len(u);assert on['stream']['tokens']==d['results']['ar']['stream']['tokens']==d['results']['frozen']['stream']['tokens'];assert metrics.matched_request_timelines(24,2,d['results']['ar']['request_timeline'],d['results']['frozen']['request_timeline'],on['request_timeline'])
  for arm in ['ar','frozen']:
   assert all(d['results'][arm]['stream']['stats'][key]==0 for key in ['updates','publications','feedback_ns','update_ns','publish_ns'])
  assert all(on['holdout']['stats'][key]==0 for key in ['updates','publications','feedback_ns','update_ns','publish_ns'])
  for part in m['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as file:
    for chunk in iter(lambda:file.read(8<<20),b''):h.update(chunk)
   assert h.hexdigest()==part['sha256']
  v={arm+'_'+field:d['results'][key]['stream'][field]for arm,key in [('ar','ar'),('frozen','frozen'),('online','64')]for field in ['tokens','seconds']};v.update(continuous=True,full_cohort=True,resource_policy_ok=True,performance_run=True);values.append(v)
  proof['cases'].append(dict(tag=c['tag'],seed=c['seed'],requests=24,physicalcards=2,ar_tok_s=d['results']['ar']['stream']['tok_s'],fixed_tok_s=d['results']['frozen']['stream']['tok_s'],online_tok_s=on['stream']['tok_s'],online_vs_ar=on['net_increment_vs_ar'],online_vs_fixed=on['net_increment'],updates=len(u),publications=on['stream']['stats']['publications'],actual_head_step=m['step'],published_updates=m['published_updates'],final_seen_tok_s=on['learned_replay']['stream']['tok_s'],fresh_holdout_tok_s=on['holdout']['tok_s'],initial_holdout_tok_s=d['results']['frozen']['holdout']['tok_s'],raw_sha256=hashlib.sha256((r/(c['tag']+'.json')).read_bytes()).hexdigest(),source_and_initial_head_exact=True,greedy_AR_exact=True,published_export_exact=True))
 proof.update(completed=True,full_audit_passed=True,metrics=metrics.confirmation_metrics(values),scope='All3complete24reservedrequest flows, independent process/Adam/source135reset; allmethods matchedwindows/generation/model, actual2cards pertrial, continuousallcosts. Finalfrozen seen andfresh2holdout diagnostics separate. Both individualandpooled +5 gates exact. No retuning fromreserved.')
except BaseException as e:proof['error']=repr(e)
(r/'pseudo253-confirmation-audit-v2.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
