from pathlib import Path
import json,time,math,hashlib
r=Path(__file__).parent;p=r/'a111-v248-domain2-prewarm-kl-audit-r1.json'
while True:
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):break
 time.sleep(3)
proof=dict(completed=False,scope='Samefirst2naturalprewarm, expected exactC229/209preparedhead; newstats source233, nativeallPythonidentical; no reserved.')
try:
 assert d['completed']and d['clean_shutdown'],d.get('error');on=d['results']['16'];u=on['updates'];expected=json.loads((r/'rows248-expected-C229-head.json').read_text());assert on['exact_greedy']and on['learned_replay']['exact_greedy'];m=on['exported_head']['manifest'];assert m['published_banks_exact'];assert len(u)==expected['updates'] and m['step']==expected['actual_head_step'];assert[x['sha256']for x in m['rank_files']]==[x['sha256']for x in expected['manifest']['rank_files']], 'crosshost preparedhead differs';assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])for x in u)
 a=[a for x in u for a in x['replay_audits']];assert a and max(x['relative_l2']for x in a)<.01 and max(x['served_replay_tv']for x in a)<.01
 for part in m['rank_files']:
  h=hashlib.sha256()
  with(Path(on['exported_head']['path'])/part['name']).open('rb')as f:
   for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
  assert h.hexdigest()==part['sha256']
 proof.update(completed=True,updates=len(u),actual_head_step=m['step'],crosshost_prepared_head_bitwise=True,greedy_AR_exact=True,published_export_exact=True)
except BaseException as e:proof['error']=repr(e)
(r/'rows248-head-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
