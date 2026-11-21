from pathlib import Path
import json,time,math
r=Path(__file__).parent;t='c23-v234-rowlayout-legacy32-regression-r1';p=r/(t+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,scope='Legacy32 default path after compactlayout refactor, real27B same2A2/TP2/rank0, same8update script asC218, unchangedbaselinefile.')
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['paired_physical_rows']==32 and d['physical_device_count']==2 and d['learner_device']==0
 old=json.loads((r/'c23-v218-fused-statistics-last1-overlap-regression-r1.json').read_text());a,b=[x['results']['16']for x in [old,d]]
 assert b['exact_greedy']and b['learned_replay']['exact_greedy'];assert b['exported_head']['manifest']['published_banks_exact']
 assert len(b['updates'])==len(a['updates'])==8
 keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale'];assert[{k:u[k]for k in keys}for u in a['updates']]==[{k:u[k]for k in keys}for u in b['updates']], 'scalar trajectory changed'
 assert[x['sha256']for x in a['exported_head']['manifest']['rank_files']]==[x['sha256']for x in b['exported_head']['manifest']['rank_files']], 'published shards differ'
 audits=[x for u in b['updates']for x in u['replay_audits']];assert audits and max(x['relative_l2']for x in audits)<.01 and max(x['served_replay_tv']for x in audits)<.01
 proof.update(completed=True,updates=8,trajectory_bitwise=True,published_shards_bitwise=True,greedy_AR_exact=True,replay_L2_max=max(x['relative_l2']for x in audits),replay_TV_max=max(x['served_replay_tv']for x in audits))
except BaseException as e:proof['error']=repr(e)
(r/'rows234-legacy-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
