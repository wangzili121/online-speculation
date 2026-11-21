from pathlib import Path
import json,math,time
r=Path(__file__).parent;tags=['c23-v210-default2-overlap-regression-compile-retry-r1','c23-v209-window4-kl-pseudo8-admission-audit-r1'];proofs={}
for tag in tags:
 p=r/(tag+'.json')
 while True:
  if p.exists():
   d=json.loads(p.read_text())
   if d.get('completed')or d.get('error'):break
  time.sleep(5)
 try:
  assert d['completed']and d['clean_shutdown'],d.get('error')
  assert d['physical_device_count']==2 and d['learner_device']==0 and d['execution_max_num_seqs']==2
  on=d['results']['16'];u=on['updates'];assert len(u)>=4 and on['exact_greedy']and on['learned_replay']['exact_greedy']and on['exported_head']['manifest']['published_banks_exact']
  assert all(math.isfinite(v['loss'])and math.isfinite(v['gradient_norm'])and v['gradient_norm']>0 for v in u)
  assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
  a=[x for v in u for x in v['replay_audits']];assert a and max(x['served_replay_tv']for x in a)<.01 and max(x['relative_l2']for x in a)<.01
  proof=dict(completed=True,tag=tag,updates=len(u),greedy_AR_exact=True,published_export_exact=True,physical_cards=2,execution_concurrency=2,stream_window=d['stream_window'])
  if d['stream_window']==2:
   old=json.loads((r/'c23-v200-overlap-last1-width8-step128-audit-r1.json').read_text())['results']['16'];keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale'];assert[{k:v[k]for k in keys}for v in u]==[{k:v[k]for k in keys}for v in old['updates']]
   assert[p['sha256']for p in on['exported_head']['manifest']['rank_files']]==[p['sha256']for p in old['exported_head']['manifest']['rank_files']];proof.update(default2_trajectory_bitwise=True,default2_published_shards_bitwise=True)
  else:
   assert d['stream_window']==4 and d['diagnostic_only']and d['audit_admission'];s=d['admission_summary'];assert s['maximum_active_requests']==2 and s['remaining_active']==0 and s['refill_with_unfinished_request']>0;proof['admission']=s
 except BaseException as e:proof=dict(completed=False,error=repr(e),tag=tag)
 proofs[tag]=proof;(r/(tag+'-validated.json')).write_text(json.dumps(proof,indent=2)+'\n')
out=dict(completed=all(v['completed']for v in proofs.values()),cases=proofs,scope='Defaultregression andqueuedwindowactualconcurrency/cache/teacher service validation; diagnostic timingsdo not qualify performance')
if not out['completed']:out['error']='Window servicevalidation failed; blockfullstreams'
(r/'window210-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
