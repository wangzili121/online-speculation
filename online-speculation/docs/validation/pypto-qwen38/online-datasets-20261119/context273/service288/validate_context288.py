"""只读验证两条长历史两卡链路，以及 KL32/16 的真实更新轨迹。"""
from pathlib import Path
import json,time,math,hashlib,traceback
r=Path(__file__).parent;plan=json.loads((r/'queue-context288-service.json').read_text());proof=dict(completed=False,cases=[],scope='Actual27B/context1024/2physicalcardsTP2/learner0, syntheticoldquestionlongcontextdiagnostics. No newpool orperformancequalification. KL32/16samehead128/freshAdam/source/order; feedback/rowlayout/greedy/export exact.')
try:
 results=[];sources=[]
 for c in plan['cases']:
  p=r/(c['tag']+'.json')
  while True:
   if p.exists():
    try:d=json.loads(p.read_text())
    except ValueError:time.sleep(1);continue
    if d.get('completed')or d.get('error'):break
   status=r/'queue-context288-service.status.json'
   if status.exists():
    state=json.loads(status.read_text());bad=next((x for x in state['cases']if x['tag']==c['tag']and x.get('error')),None)
    if bad:raise RuntimeError((c['tag'],bad))
   time.sleep(5)
  assert d['completed']and d['clean_shutdown'],(c['tag'],d.get('error'));assert d['requests']==d['ar_checked_requests']==6 and d['physical_device_count']==2 and d['learner_device']==0 and len(set(d['visible_devices'].split(',')))==2;assert d['head_step']==128 and d['optimizer_init']=='fresh'and d['max_updates']is None and d['background']is False;assert d['max_model_len']==1024 and d['diagnostic_only'];assert d['training_rows']==int(c['args'][c['args'].index('--training-rows')+1]);assert d['prompts_sha256']==c['prompts_sha256'];assert d['draft_width']==8 and d['candidates']==7 and d['teacher_tokens']==7 and d['last_layers']==1
  assert d['row_layout_audits']and all(x['logits_bitwise_equal']for x in d['row_layout_audits']);assert any(any(capacity==1024 for capacity,width in x['key'])for x in d['row_layout_audits']);source=json.loads((r/(c['tag']+'.source.json')).read_text());sources.append(source['files']);on=d['results']['16'];assert on['exact_greedy']and on['learned_replay']['exact_greedy'];assert len(on['updates'])>=4 and all(math.isfinite(u['loss'])and math.isfinite(u['gradient_norm'])and u['gradient_norm']>=0 for u in on['updates']);assert any(u['gradient_norm']>0 for u in on['updates']);audits=[a for u in on['updates']for a in u['replay_audits']];assert audits and all(a['served_replay_tv']<.01 and a['relative_l2']<.01 for a in audits);m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==128+on['published_version']and on['published_version']==on['optimizer_version']==len(on['updates'])
  for part in m['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as file:
    for chunk in iter(lambda:file.read(8<<20),b''):h.update(chunk)
   assert h.hexdigest()==part['sha256']
  proof['cases'].append(dict(tag=c['tag'],training_rows=d['training_rows'],objective=d['objective'],updates=len(on['updates']),greedy_AR_exact=True,published_export_exact=True,max_replay_tv=max(a['served_replay_tv']for a in audits),max_replay_l2=max(a['relative_l2']for a in audits),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),published_shards=m['rank_files']));results.append(d)
 assert all(s==sources[0]for s in sources);old,new=[d['results']['16']for d in results[:2]];assert results[0]['head_shard_sha256']==results[1]['head_shard_sha256'];keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale','interval'];a=[{k:u[k]for k in keys}for u in old['updates']];b=[{k:u[k]for k in keys}for u in new['updates']];assert a==b,('trajectory mismatch',[(i,x,y)for i,(x,y)in enumerate(zip(a,b))if x!=y][:2]);assert old['exported_head']['manifest']['rank_files']==new['exported_head']['manifest']['rank_files']
 proof.update(completed=True,KL32_16_update_trajectory_bitwise=True,KL32_16_served_shards_exact=True,source_unchanged=True,both_longcontext_service_gates_passed=True)
except BaseException as e:proof.update(error=repr(e),traceback=traceback.format_exc())
(r/'context288-service-validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
