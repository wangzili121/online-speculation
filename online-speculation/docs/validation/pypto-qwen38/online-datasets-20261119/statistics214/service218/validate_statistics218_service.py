from pathlib import Path
import time,json,math
r=Path(__file__).parent;tag='c23-v218-fused-statistics-last1-overlap-regression-r1';p=r/(tag+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,tag=tag)
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['physical_device_count']==2 and d['learner_device']==0
 on=d['results']['16'];old=json.loads((r/'c23-v210-default2-overlap-regression-compile-retry-r1.json').read_text())['results']['16'];keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale']
 assert on['exact_greedy']and on['learned_replay']['exact_greedy'];u=on['updates'];assert len(u)==8
 assert[{k:v[k]for k in keys}for v in u]==[{k:v[k]for k in keys}for v in old['updates']]
 assert[p['sha256']for p in on['exported_head']['manifest']['rank_files']]==[p['sha256']for p in old['exported_head']['manifest']['rank_files']]
 assert on['exported_head']['manifest']['published_banks_exact'];a=[v for x in u for v in x['replay_audits']];assert a and max(x['served_replay_tv']for x in a)<.01 and max(x['relative_l2']for x in a)<.01
 assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 proof.update(completed=True,updates=8,physicalcards=2,trajectory_bitwise=True,exported_shards_bitwise=True,greedy_AR_exact=True,replay_max_TV=max(x['served_replay_tv']for x in a),replay_max_relative_L2=max(x['relative_l2']for x in a))
except BaseException as e:proof['error']=repr(e)
(r/(tag+'-validated.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
