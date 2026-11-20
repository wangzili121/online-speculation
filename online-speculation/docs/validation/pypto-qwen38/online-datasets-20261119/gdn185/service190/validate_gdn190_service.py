from pathlib import Path
import json,time,math
r=Path(__file__).resolve().parent;tags=['b167-v190-width8-valid-shared-gdn-audit-r1','b167-v190-width32-padded-shared-gdn-audit-r1'];proofs={}
for tag in tags:
 while True:
  p=r/(tag+'.json')
  if p.exists():
   d=json.loads(p.read_text())
   if d.get('completed')or d.get('error'):break
  time.sleep(5)
 try:
  assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
  assert d['physical_device_count']==2 and d['learner_device']==0
  refpath=(r/'gdn190-width8-reference.json'if 'width8-'in tag else r/'b167-v179-lora64-last4-device-publish4-chained-audit-r1.json');ref=json.loads(refpath.read_text());on=d['results']['16'];old=ref['results']['16']
  assert d['head_shard_sha256']==ref['head_shard_sha256']and d['prompts_sha256']==ref['prompts_sha256']
  keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale']
  assert [{k:u[k]for k in keys}for u in on['updates']]==[{k:u[k]for k in keys}for u in old['updates']]
  assert [v['sha256']for v in on['exported_head']['manifest']['rank_files']]==[v['sha256']for v in old['exported_head']['manifest']['rank_files']]
  assert on['exact_greedy']and on['learned_replay']['exact_greedy']and on['exported_head']['manifest']['published_banks_exact']
  assert all(d['results'][arm]['stream']['stats']['updates']==0 for arm in ['ar','frozen'])
  proof=dict(completed=True,tag=tag,updates=len(on['updates']),trajectory_bitwise=True,published_shards_bitwise=True,greedy_AR_exact=True,published_export_exact=True,reference=str(refpath))
 except BaseException as e:proof=dict(completed=False,tag=tag,error=repr(e))
 proofs[tag]=proof;(r/(tag+'-validated.json')).write_text(json.dumps(proof,indent=2)+'\n')
out=dict(completed=all(p['completed']for p in proofs.values()),scope='Shared GDN factory whole-model refactor parity, actual2card short correctness only',cases=proofs)
if not out['completed']:out['error']='Shared factory refactor parity failed; do not claim equivalence'
(r/'gdn190-service-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
