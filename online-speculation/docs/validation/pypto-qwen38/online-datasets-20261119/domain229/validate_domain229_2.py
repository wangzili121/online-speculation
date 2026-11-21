from pathlib import Path
import json,time,hashlib,math
r=Path(__file__).parent;tag='c23-v229-domain2-prewarm-kl-audit-r1';p=r/(tag+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,prewarm_requests=2,entire_evaluated_requests=56-2,initialization_only=True)
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['requests']==2 and d['physical_device_count']==2 and d['learner_device']==0
 on=d['results']['16'];u=on['updates'];assert len(u)>=1 and on['exact_greedy']and on['learned_replay']['exact_greedy'];assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])and x['gradient_norm']>0 for x in u)
 a=[x for v in u for x in v['replay_audits']];assert a and max(x['served_replay_tv']for x in a)<.01 and max(x['relative_l2']for x in a)<.01
 e=on['exported_head'];assert e['manifest']['published_banks_exact']and e['manifest']['step']==128+len(u)
 for part in e['manifest']['rank_files']:
  h=hashlib.sha256()
  with(Path(e['path'])/part['name']).open('rb')as f:
   for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
  assert h.hexdigest()==part['sha256']
 assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 proof.update(completed=True,updates=len(u),actual_head_step=e['manifest']['step'],manifest=e['manifest'],head=e['path'],published_export_exact=True,greedy_AR_exact=True)
except BaseException as e:proof['error']=repr(e)
(r/'domain229-head2-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
