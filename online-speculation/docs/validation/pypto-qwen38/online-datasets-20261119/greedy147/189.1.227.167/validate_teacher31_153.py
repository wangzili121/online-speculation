from pathlib import Path
import sys,json,time,math
r=Path(__file__).resolve().parent;tag=sys.argv[1];deadline=time.monotonic()+36000
try:
 while True:
  f=r/(tag+'.json');d=json.loads(f.read_text())if f.exists()else{}
  if d.get('completed')or d.get('error'):break
  if time.monotonic()>deadline:raise TimeoutError('Teacher31 gate observation timeout; not NN termination/restart')
  time.sleep(10)
 assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
 assert d['teacher_targets']=='greedy'and d['teacher_tokens']==31 and d['physical_device_count']==2 and d['learner_device']==0
 gs=[v for v in d['results'].values()if'updates'in v];us=[u for g in gs for u in g['updates']];aa=[a for u in us for a in u['replay_audits']];assert us and aa
 assert any(u['full_feedback_blocks']>=1 and u['supervised']>=31 for u in us),'No complete31 teacher window exercised'
 assert all(math.isfinite(u[k])for u in us for k in ['loss','gradient_norm','learning_rate'])
 assert all(a['served_replay_tv']<.01 and a['relative_l2']<.01 for a in aa)
 assert all(g['exact_greedy']and g['exported_head']['manifest']['published_banks_exact']for g in gs)
 o=dict(completed=True,scope='Real27B physical2card hardteacher31 partial/EOS/fullwindow/replay2/update/export; not throughput',tag=tag,updates=len(us),supervised_rows=sorted({u['supervised']for u in us}),full31_blocks=sum(u['full_feedback_blocks']for u in us),greedyARexact=True,published_export_exact=True,max_replay_TV=max(a['served_replay_tv']for a in aa),max_relative_L2=max(a['relative_l2']for a in aa))
except BaseException as e:o=dict(completed=False,error=repr(e),scope='Hardteacher31 gate failed or observationexpired; preserve NN/raw, no automatic restart')
(r/'teacher31_153-service-validation.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))
