from pathlib import Path
import sys,json,time,math
r=Path(__file__).resolve().parent;tag=sys.argv[1];out=r/'timing150-service-validation.json';deadline=time.monotonic()+36000
try:
 while True:
  p=r/(tag+'.json');d=json.loads(p.read_text())if p.exists()else{}
  if d.get('completed')or d.get('error'):break
  if time.monotonic()>deadline:raise TimeoutError('timing gate observation timeout; no restart implied')
  time.sleep(10)
 assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
 assert d['physical_device_count']==2 and d['learner_device']==0 and d['feedback_timing']=='near_update'and d['teacher_targets']=='greedy'and d['temperature']==0
 g=[v for v in d['results'].values()if'updates'in v];us=[u for v in g for u in v['updates']];aa=[a for u in us for a in u['replay_audits']];assert len(us)>=3 and aa
 assert all(math.isfinite(u[k])for u in us for k in ['loss','gradient_norm','learning_rate'])
 assert all(v['exact_greedy']and v['exported_head']['manifest']['published_banks_exact']for v in g)
 assert all(a['served_replay_tv']<.01 and a['relative_l2']<.01 for a in aa)
 assert all(len(u['feedback_age_rounds'])==u['feedback_blocks']and all(0<=age<=u['interval']+2 for age in u['feedback_age_rounds'])for u in us)
 late=[u for u in us if u['interval']==32 and all((u['round']-age)%32>=16 for age in u['feedback_age_rounds'])];assert late,'No real near-update capture was exercised'
 result=dict(completed=True,scope='Actual27B physical2card sharedrank0 near-update collection; diagnostic, not speed acceptance',tag=tag,updates=len(us),normal32_near_update_events=len(late),actual_feedback_ages=[u['feedback_age_rounds']for u in us],greedyARexact=True,published_export_exact=True,max_TV=max(a['served_replay_tv']for a in aa),max_L2=max(a['relative_l2']for a in aa))
except BaseException as e:result=dict(completed=False,error=repr(e),scope='Timing correctness gate failed or wait expired; retain NN state/raw, no automatic restart')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
