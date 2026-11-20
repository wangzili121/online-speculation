from pathlib import Path
import json,math,time
r=Path(__file__).resolve().parent
MODE='width180'
TAGS=['c23-v180-width8-padded-gdn-audit-r1', 'c23-v180-width8-valid-gdn-audit-r1', 'c23-v180-width16-valid-gdn-audit-r1', 'c23-v180-width32-valid-gdn-audit-r1']
name='validation181-'+MODE+'.json'
def validate(tag):
 p=r/(tag+'.json')
 if not p.exists():return None
 d=json.loads(p.read_text())
 if not d.get('completed')and not d.get('error'):return None
 try:
  assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
  assert d['physical_device_count']==2 and d['learner_device']==0 and d['background']is False
  on=d['results']['16'];assert on['exact_greedy']and on['learned_replay']['exact_greedy']
  for arm in ['ar','frozen']:
   assert all(d['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])
  u=on['updates'];assert len(u)>=4 and all(math.isfinite(v['loss'])and math.isfinite(v['gradient_norm'])and v['gradient_norm']>0 for v in u)
  opt,pub=on['optimizer_version'],on['published_version'];assert pub==opt//4*4 and pub>=4
  m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==d['head_step']+pub
  assert all(a!=b['sha256']for a,b in zip(d['head_shard_sha256'],m['rank_files']))
  audits=[a for v in u for a in v['replay_audits']];assert audits and max(a['served_replay_tv']for a in audits)<.01 and max(a['relative_l2']for a in audits)<.01
  assert on['stream']['stats']['busy_updates']==0
  return dict(completed=True,tag=tag,updates=len(u),optimizer_version=opt,published_version=pub,greedy_AR_exact=True,published_export_exact=True,max_replay_TV=max(a['served_replay_tv']for a in audits),max_replay_L2=max(a['relative_l2']for a in audits),trajectory=[{k:v[k]for k in ['round','loss','supervised','learning_rate','gradient_norm','clip_scale']}for v in u],rank_sha256=[v['sha256']for v in m['rank_files']],draft_width=d.get('draft_width',32),gdn_valid_rows=d.get('gdn_valid_rows',False),merge_backend=d.get('adapter_merge_backend','chained'))
 except BaseException as err:return dict(completed=False,tag=tag,error=repr(err))
proofs={};deadline=time.monotonic()+43200
while len(proofs)<len(TAGS):
 for tag in TAGS:
  if tag in proofs:continue
  p=validate(tag)
  if p is not None:
   proofs[tag]=p;(r/(tag+'-validated181.json')).write_text(json.dumps(p,indent=2)+'\n')
 if time.monotonic()>deadline:break
 time.sleep(5)
result=dict(completed=len(proofs)==len(TAGS)and all(p['completed']for p in proofs.values()),mode=MODE,cases=proofs,scope='Short actual27B2card correctness only; no throughput qualification')
if result['completed']:
 a,b=[proofs[t]for t in TAGS[:2]]
 result['trajectory_bitwise']=a['trajectory']==b['trajectory'];result['published_shards_bitwise']=a['rank_sha256']==b['rank_sha256'];result['completed'] &= result['trajectory_bitwise']and result['published_shards_bitwise']
if not result['completed']:result['error']='Service correctness/equivalence failed or timed out; do not enable dependent experiments'
(r/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
