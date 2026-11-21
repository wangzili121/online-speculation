from pathlib import Path
import json,time,math
r=Path(__file__).parent;p=r/'c23-v239-rowlayout-compact16-width16-shadow-replay-r1.json'
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,scope='Real27B2A2/TP2/rank0compact16pair vs32padded shadow plus servedteacherreplay. Diagnostic only.')
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['draft_width']==16;assert d['paired_physical_rows']==16 and d['physical_device_count']==2 and d['learner_device']==0 and d['diagnostic_only']
 on=d['results']['16'];u=on['updates'];assert len(u)>=4 and on['exact_greedy']and on['learned_replay']['exact_greedy'];assert on['exported_head']['manifest']['published_banks_exact'];assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])and x['gradient_norm']>0 for x in u)
 audits=[a for x in u for a in x['replay_audits']];assert audits and max(a['relative_l2']for a in audits)<.01 and max(a['served_replay_tv']for a in audits)<.01
 rows=d['row_layout_audits'];assert rows and max(a['relative_L2']for a in rows)<.01 and max(a['maximum_TV']for a in rows)<.01
 proof.update(completed=True,updates=len(u),greedy_AR_exact=True,published_export_exact=True,replay_TV_max=max(a['served_replay_tv']for a in audits),replay_L2_max=max(a['relative_l2']for a in audits),row_layout_audits=rows)
except BaseException as e:proof['error']=repr(e)
(r/'rows239-compact-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
