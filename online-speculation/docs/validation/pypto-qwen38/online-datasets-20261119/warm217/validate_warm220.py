from pathlib import Path
import json,time,hashlib,math
r=Path(__file__).parent;tag='a111-v217-domain-prewarm8-kl-audit-r1';p=r/(tag+'.json')
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(5)
proof=dict(completed=False,initialization_only=True,excluded_from_48request_evaluation=True,tag=tag)
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');assert d['requests']==8 and d['physical_device_count']==2 and d['learner_device']==0
 on=d['results']['16'];assert on['exact_greedy']and on['learned_replay']['exact_greedy'];a=[x for u in on['updates']for x in u['replay_audits']];assert a and max(x['served_replay_tv']for x in a)<.01 and max(x['relative_l2']for x in a)<.01
 e=on['exported_head'];assert e['manifest']['published_banks_exact']and e['manifest']['step']==128+len(on['updates']);assert len(on['updates'])==24
 for part in e['manifest']['rank_files']:
  h=hashlib.sha256()
  with(Path(e['path'])/part['name']).open('rb')as f:
   for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
  assert h.hexdigest()==part['sha256']
 assert all(all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 proof.update(completed=True,initial_head_step=e['manifest']['step'],published_updates=len(on['updates']),head_path=e['path'],manifest=e['manifest'],physicalcards=2,prewarm_requests=8,evaluated_requests=48,greedy_AR_exact=True,published_export_exact=True,same_head_in_both_future_arms=True)
except BaseException as e:proof['error']=repr(e)
(r/'warm220-head-binding.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
