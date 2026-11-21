"""旧题的双卡上下文边界检查，不计算或选择新题性能结论。"""
from pathlib import Path
import json,time,math,hashlib,traceback
r=Path(__file__).parent;tag='a111-v303-outputbudget-boundary1024-replay';proof=dict(completed=False,scope='2A2/TP2/rank0, oldsynthetic boundary prompts992/1000, shared nominal512 clipped32/24, allarms ignoreEOS. Diagnostic teacher replay and publication only, no performance qualification.')
try:
 while True:
  p=r/(tag+'.json');d=json.loads(p.read_text())if p.exists()else{};sp=r/'queue-budget303-boundary.status.json';ctrl=json.loads(sp.read_text())if sp.exists()else{};case=next((x for x in ctrl.get('cases',[])if x['tag']==tag),{})
  if d.get('error')or case.get('error'):raise RuntimeError(d.get('error')or case['error'])
  if case.get('completed')and d.get('completed')and d.get('clean_shutdown'):break
  time.sleep(5)
 assert d['requests']==d['ar_checked_requests']==2 and d['physical_device_count']==2 and d['learner_device']==0 and not d['background'];assert d['audit_replay']and not d['stop_on_eos'];assert d['max_model_len']==1024 and d['training_rows']==16 and d['head_step']==128 and d['optimizer_init']=='fresh';assert d['output_budget_mode']=='context_clipped'and d['output_budget_limit']==512 and d['request_prompt_tokens']==[992,1000]and d['request_output_budgets']==[32,24];assert all(a+b==1024 for a,b in zip(d['request_prompt_tokens'],d['request_output_budgets']));assert d['prompts_sha256']==json.loads((r/'budget303-boundary-assets.json').read_text())['sha256']
 for key in ['ar','frozen','4']:
  arm=d['results'][key];timeline=arm['request_timeline'];assert len(timeline)==1 and timeline[0]['requests']==[1,2]and timeline[0]['output_budgets']==timeline[0]['output_tokens']==[32,24];s=arm['stream'];assert s['tokens']==56 and math.isclose(s['seconds'],timeline[0]['seconds']+s['tail_seconds'],rel_tol=1e-10)
  if key!='4':assert all(s['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])
 on=d['results']['4'];assert on['exact_greedy']and on['learned_replay']['exact_greedy']and on['learned_replay']['stream']['tokens']==56;assert len(on['updates'])>=2 and on['published_version']==on['optimizer_version']==len(on['updates'])
 assert all(math.isfinite(u['loss'])and math.isfinite(u['gradient_norm'])and u['gradient_norm']>=0 for u in on['updates']);assert any(u['gradient_norm']>0 for u in on['updates']);audits=[x for u in on['updates']for x in u['replay_audits']];assert audits and all(x['served_replay_tv']<.01 and x['relative_l2']<.01 for x in audits)
 for prompt,budget in zip(d['holdout_prompt_tokens'],d['holdout_output_budgets']):assert budget==min(512,1024-prompt)
 assert on['holdout']['tokens']==sum(d['holdout_output_budgets'])and all(on['holdout']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])
 m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==128+on['published_version'];actual={}
 for part in m['rank_files']:
  h=hashlib.sha256()
  with(Path(on['exported_head']['path'])/part['name']).open('rb')as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  assert h.hexdigest()==part['sha256'];actual[part['name']]=h.hexdigest()
 new=json.loads((r/(tag+'.source.json')).read_text())['files'];old=json.loads((r/'a111-v292-context1024-trainingrows16-KL-replay.source.json').read_text())['files'];left={n.split('src-dense-v292/',1)[1]:s for n,s in old.items()if'src-dense-v292/'in n};right={n.split('src-dense-v303/',1)[1]:s for n,s in new.items()if'src-dense-v303/'in n};assert left==right;pre=json.loads((r/'budget303-CPU-preflight.json').read_text());assert new['orthrus_bench_dense_v303.py']==pre['benchmark_sha256'];assert {n:s for n,s in old.items()if n.startswith('pypto-serving-qwen3.8-27B/')}=={n:s for n,s in new.items()if n.startswith('pypto-serving-qwen3.8-27B/')}
 proof.update(completed=True,greedy_AR_and_replay_exact=True,complete_prompts_reached_1024=True,all_arms_same_per_request_budget=True,updates=len(on['updates']),max_replay_tv=max(x['served_replay_tv']for x in audits),max_replay_l2=max(x['relative_l2']for x in audits),supervised_rows=[u['supervised']for u in on['updates']],actual_published_shards=actual,source_model_kernels_unchanged=True,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),paired_draft_calls=on['stream']['stats']['draft_pair_batches'],single_draft_calls=on['stream']['stats']['draft_batches']-on['stream']['stats']['draft_pair_batches'])
except BaseException as error:proof.update(error=repr(error),traceback=traceback.format_exc())
(r/'budget303-device-validation.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof),flush=True)
