from pathlib import Path
import json,time,math
r=Path(__file__).parent;rows=[];tags=['c23-v219-statistics-last4-'+m+'-audit-r1'for m in ['chained','merged']]
for tag in tags:
 p=r/(tag+'.json')
 while True:
  if p.exists():
   d=json.loads(p.read_text())
   if d.get('completed')or d.get('error'):break
  time.sleep(5)
 rows.append(d)
proof=dict(completed=False,tags=tags,scope='Actual27B same2card/rank0 last4 statistics path regression, audit timings are notperformance')
try:
 for d,mode in zip(rows,['chained','merged']):
  assert d['completed']and d['clean_shutdown'],d.get('error');assert d['statistics_implementation']==mode;assert d['physical_device_count']==2 and d['learner_device']==0 and d['last_layers']==4
  on=d['results']['16'];assert on['exact_greedy']and on['learned_replay']['exact_greedy'];assert on['exported_head']['manifest']['published_banks_exact'];assert len(on['updates'])>=4
  audits=[a for u in on['updates']for a in u['replay_audits']];assert audits and max(a['served_replay_tv']for a in audits)<.01 and max(a['relative_l2']for a in audits)<.01
  assert all(math.isfinite(u['loss'])and math.isfinite(u['gradient_norm'])and u['gradient_norm']>0 for u in on['updates'])
  assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 a,b=[d['results']['16']for d in rows];keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale'];assert[{k:u[k]for k in keys}for u in a['updates']]==[{k:u[k]for k in keys}for u in b['updates']];assert[p['sha256']for p in a['exported_head']['manifest']['rank_files']]==[p['sha256']for p in b['exported_head']['manifest']['rank_files']]
 proof.update(completed=True,updates=len(a['updates']),trajectory_bitwise=True,published_shards_bitwise=True,greedy_AR_exact=True,physicalcards=2,last_layers=4)
except BaseException as e:proof['error']=repr(e)
(r/'statistics219-validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
