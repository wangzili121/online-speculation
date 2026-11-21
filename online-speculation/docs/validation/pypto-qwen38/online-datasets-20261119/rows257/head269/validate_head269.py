"""只读核对重建起点；不接受同名但不同权重的 head。"""
from pathlib import Path
import json,time,hashlib,traceback
r=Path(__file__).parent;tag='b167-v265-domain2-reconstruct135-rows32';p=r/(tag+'.json');out=dict(completed=False,scope='Exacthead135reconstruction usingoriginalfirst2questions; no performancequalification orreserved use.')
try:
 while True:
  if p.exists():
   try:d=json.loads(p.read_text())
   except ValueError:time.sleep(1);continue
   if d.get('completed')or d.get('error'):break
  state=r/'queue-head269-prewarm.status.json'
  if state.exists():
   status=json.loads(state.read_text());bad=next((c for c in status['cases']if c.get('error')),None)
   if bad:raise RuntimeError(bad)
  time.sleep(5)
 assert d['completed']and d['clean_shutdown'];assert d['requests']==d['ar_checked_requests']==2 and d['head_step']==128 and d['training_rows']==32 and d['physical_device_count']==2 and d['learner_device']==0;on=d['results']['16'];assert len(on['updates'])==7 and on['exact_greedy']and on['learned_replay']['exact_greedy'];m=on['exported_head']['manifest'];expected=['711e1b71c0bbdd83eb816c6152061198c4c65f14b7cf51a242a8463d0e14aba2','b54c1fb72623d652f3d6dfec4dc3c6f355a9afd3717a15fb9de7ae457511c727'];assert m['step']==135 and m['published_banks_exact']and[x['sha256']for x in m['rank_files']]==expected
 for part in m['rank_files']:
  h=hashlib.sha256()
  with(Path(on['exported_head']['path'])/part['name']).open('rb')as file:
   for chunk in iter(lambda:file.read(8<<20),b''):h.update(chunk)
  assert h.hexdigest()==part['sha256']
 audits=[a for u in on['updates']for a in u['replay_audits']];assert audits and all(a['served_replay_tv']<.01 and a['relative_l2']<.01 for a in audits);out.update(completed=True,actual_head_step=135,updates=7,rank_files=m['rank_files'],actual_weight_bytes_match_original_C229_A248=True,greedy_AR_exact=True,published_export_exact=True,max_replay_tv=max(a['served_replay_tv']for a in audits),max_replay_l2=max(a['relative_l2']for a in audits))
except BaseException as e:out.update(error=repr(e),traceback=traceback.format_exc())
(r/'head269-exact-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
