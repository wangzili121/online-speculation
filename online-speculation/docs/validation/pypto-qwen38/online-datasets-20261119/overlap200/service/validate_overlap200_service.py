from pathlib import Path
import json,time,math
r=Path(__file__).parent;t='c23-v200-overlap-last1-width8-step128-audit-r1';p=r/(t+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
try:
 assert d['completed']and d['clean_shutdown'],d.get('error')
 assert d['head_step']==128 and d['physical_device_count']==2 and d['learner_device']==0
 assert d['draft_width']==8 and d['gdn_valid_rows']and d['objective']=='log_overlap'and d['teacher_targets']=='distribution'
 on=d['results']['16'];u=on['updates'];assert len(u)>=4 and on['exact_greedy']and on['learned_replay']['exact_greedy']
 assert on['exported_head']['manifest']['published_banks_exact']and on['published_version']==on['optimizer_version']==len(u)
 assert all(math.isfinite(v['loss'])and math.isfinite(v['gradient_norm'])and v['gradient_norm']>0 for v in u)
 assert all(d['results'][arm]['stream']['stats']['updates']==0 for arm in ['ar','frozen'])
 a=[x for v in u for x in v['replay_audits']];assert a and max(v['served_replay_tv']for v in a)<.01 and max(v['relative_l2']for v in a)<.01
 proof=dict(completed=True,tag=t,updates=len(u),head_step=128,physical_cards=2,greedy_AR_exact=True,published_export_exact=True,objective='log_overlap',max_replay_TV=max(v['served_replay_tv']for v in a),max_replay_L2=max(v['relative_l2']for v in a),audit_losses=[dict(before=x['batch_loss_before'],after=x['batch_loss_after'])for x in a],scope='Short actual27B independent new objective service correctness, not throughput qualification')
except BaseException as e:proof=dict(completed=False,error=repr(e),tag=t)
(r/(t+'-validated.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
