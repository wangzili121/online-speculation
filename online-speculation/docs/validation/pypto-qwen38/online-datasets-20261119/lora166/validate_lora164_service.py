from pathlib import Path
import json,math
r=Path(__file__).resolve().parent
items=[('b167-v164-dense-regression-replay2-audit-r1','dense'),('b167-v164-lora64-last1-replay2-audit-r1','lora_merged'),('b167-v164-lora64-last4-replay2-audit-r1','lora_merged')]
# Forward terminal prerequisite failures so later gates do not wait for a file forever.
for index in range(1,len(items)):
 prev=r/(items[index-1][0]+'-validated.json');current=r/(items[index][0]+'-validated.json')
 if prev.exists()and not json.loads(prev.read_text()).get('completed')and not current.exists():
  current.write_text(json.dumps(dict(completed=False,tag=items[index][0],error='prior service correctness gate failed; not launched'),indent=2)+'\n')
for tag,mode in items:
 output=r/(tag+'-validated.json')
 if output.exists():continue
 path=r/(tag+'.json')
 if not path.exists():break
 d=json.loads(path.read_text())
 if not d.get('completed')and not d.get('error'):break
 try:
  assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
  assert d['physical_device_count']==2 and d['learner_device']==0 and d['parameterization']==mode
  assert d['optimizer_init']=='fresh'and d['max_updates']is None
  for k in ['ar','frozen']:
   assert d['results'][k]['stream']['stats']['updates']==0
  pairs=[v for k,v in d['results'].items()if k not in ['ar','frozen']];assert len(pairs)==1
  online=pairs[0];up=online['updates'];assert len(up)>=6
  assert online['exact_greedy']and online['learned_replay']['exact_greedy']
  assert online['exported_head']['manifest']['published_banks_exact']
  assert online['stream']['stats']['busy_updates']==0
  audits=[a for u in up for a in u['replay_audits']];assert audits
  assert max(a['served_replay_tv']for a in audits)<.01
  assert max(a['relative_l2']for a in audits)<.01
  assert all(math.isfinite(u['loss'])and math.isfinite(u['gradient_norm'])and u['gradient_norm']>0 for u in up)
  if mode=='dense':
   baseline=json.loads((r/'b167-v147-greedyteacher-last1-replay2-audit-r1.json').read_text())['results']['16']
   keys=['loss','gradient_norm','clip_scale','supervised','supervision_weight','replay_audits','learning_rate','version']
   assert [[u[k]for k in keys]for u in up]==[[u[k]for k in keys]for u in baseline['updates']]
   a=online['exported_head']['manifest'];b=baseline['exported_head']['manifest']
   assert a==b,'dense full exported manifest differs'
  else:
   assert d['adapter_rank']==64 and d['trainable_parameters']>0
   if d['last_layers']==1:assert d['trainable_parameters']==2621952
  result=dict(completed=True,tag=tag,mode=mode,updates=len(up),max_replay_TV=max(a['served_replay_tv']for a in audits),max_replay_L2=max(a['relative_l2']for a in audits),greedy_AR_exact=True,published_export_exact=True,dense_baseline_exact=mode=='dense',scope='Real27B actual2card short service correctness, not speed acceptance')
 except BaseException as err:result=dict(completed=False,tag=tag,error=repr(err))
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if not result['completed']:break
