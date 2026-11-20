from pathlib import Path
import json,math,time
r=Path(__file__).resolve().parent;t='c23-v164-lora64-last4-device-publish4-audit-r1';out=r/(t+'-validated.json')
while not out.exists():
 p=r/(t+'.json')
 if p.exists():
  d=json.loads(p.read_text())
  if d.get('completed')or d.get('error'):
   try:
    assert d.get('completed')and d.get('clean_shutdown'),d.get('error')
    assert d['physical_device_count']==2 and d['learner_device']==0
    assert d['last_layers']==4 and d['parameterization']=='lora_merged'and d['publish_every']==4
    assert d['publication_backend']=='device'and d['learner_memory_plan']['publication_bytes']>0
    assert all(d['results'][k]['stream']['stats']['updates']==0 for k in ['ar','frozen'])
    on=d['results']['16'];u=on['updates'];assert len(u)>=6
    assert on['exact_greedy']and on['learned_replay']['exact_greedy']
    assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])and x['gradient_norm']>0 for x in u)
    opt,pub=on['optimizer_version'],on['published_version'];assert opt>pub and pub==(opt//4)*4
    m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['published_updates']==pub and m['step']==d['head_step']+pub
    assert all(a!=b['sha256']for a,b in zip(d['head_shard_sha256'],m['rank_files']))
    audits=[a for x in u for a in x['replay_audits']];assert audits and max(x['served_replay_tv']for x in audits)<.01 and max(x['relative_l2']for x in audits)<.01
    assert on['stream']['stats']['busy_updates']==0 and on['stream']['stats']['publications']==pub//4
    proof=dict(completed=True,tag=t,updates=len(u),optimizer_version=opt,published_version=pub,unpublished_tail=opt-pub,exported_step=m['step'],device_publication=True,greedy_AR_exact=True,actual_published_export_exact=True,max_replay_TV=max(x['served_replay_tv']for x in audits),max_replay_L2=max(x['relative_l2']for x in audits),scope='Actual27B2card last4 LoRA cadence4 correctness with unpublished tail; not speed acceptance')
   except BaseException as err:proof=dict(completed=False,error=repr(err),tag=t)
   out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True);break
 time.sleep(5)
