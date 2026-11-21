from pathlib import Path
import time,json,math
r=Path(__file__).parent;tag='b167-v224-sampling1-overlap-step128-audit-r1';p=r/(tag+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,tag=tag,scope='Real27B/TP2 two-card co-located temperature1/NLL/overlapped update service correctness, diagnostic notperformance')
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['physical_device_count']==2 and d['learner_device']==0 and d['temperature']==1 and d['overlap_sampling']
 on=d['results']['8'];updates=on['updates'];assert len(updates)>=4;assert all(math.isfinite(u['loss'])and math.isfinite(u['gradient_norm'])and u['gradient_norm']>0 for u in updates)
 audits=[v for u in updates for v in u['replay_audits']];assert audits and max(v['served_replay_tv']for v in audits)<.01 and max(v['relative_l2']for v in audits)<.01
 assert on['exported_head']['manifest']['published_banks_exact'];assert on['stream']['stats']['accepted']>0 and on['stream']['stats']['rejected_rounds']>0
 assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 assert all(not row['learner']['pending_update']for row in on['request_timeline'])
 proof.update(completed=True,updates=len(updates),published_export_exact=True,replay_max_TV=max(v['served_replay_tv']for v in audits),replay_max_relative_L2=max(v['relative_l2']for v in audits),no_pending_at_window_end=True,sampling_accepted_and_rejected=True)
except BaseException as e:proof['error']=repr(e)
(r/'sampling224-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
