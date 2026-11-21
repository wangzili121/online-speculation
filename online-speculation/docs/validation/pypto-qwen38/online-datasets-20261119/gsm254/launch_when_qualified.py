"""Only a complete qualifying GSM384 flow can consume the new reserved cohort."""
from pathlib import Path
import json,subprocess,hashlib,zlib,time,traceback
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-dir',type=Path,required=True)
args=parser.parse_args()
A=args.artifact_dir.resolve();state=A/'gsm254-launcher-state.json';tag='b167-v245-triplet-compact16-step128-gsm384-tv-staged256-r1'
def save(d):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(state)
def ssh(h,code,timeout=55):
 return subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@'+h,'python3 -'],input=code.encode(),capture_output=True,timeout=timeout)
read=r'''from pathlib import Path
import json,zlib,sys
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-background');tag='b167-v245-triplet-compact16-step128-gsm384-tv-staged256-r1';d=json.loads((r/(tag+'.json')).read_text());out=dict(completed=d.get('completed'),error=d.get('error'),stage=d.get('stage'),clean_shutdown=d.get('clean_shutdown'))
if d.get('completed')or d.get('error'):out['files']={tag+ext:(r/(tag+ext)).read_text()for ext in ['.json','.source.json']};out['plan']=json.loads((r/'queue-rows245-gsm384.json').read_text())
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
try:
 assert json.loads((A/'gsm254-unused-reserve-audit.json').read_text())['completed']
 while True:
  try:
   x=ssh('189.1.227.167',read)
   if x.returncode:raise RuntimeError(x.stderr.decode()[-1000:])
   d=json.loads(zlib.decompress(x.stdout))
  except (subprocess.TimeoutExpired,RuntimeError,ValueError,zlib.error)as e:
   save(dict(stage='verified_read_wait',terminal=False,observed_error=repr(e),job_tag=tag));time.sleep(20);continue
  if d.get('error'):
   save(dict(stage='qualifier_failed',qualification=False,error=d['error'],no_reserved_executed=True));raise SystemExit(0)
  if d.get('completed'):
   assert d['clean_shutdown'];break
  save(dict(stage='waiting_actual_live_qualifier',job_tag=tag,remote_stage=d.get('stage'),no_reserved_executed=True));time.sleep(20)
 raw=d['files'][tag+'.json'];q=json.loads(raw);on=q['results']['128'];assert q['completed']and q['clean_shutdown']and q['requests']==q['ar_checked_requests']==384 and q['physical_device_count']==2 and q['learner_device']==0 and q['head_step']==128 and q['optimizer_init']=='fresh'and q['max_updates']is None and on['exact_greedy']and on['learned_replay']['exact_greedy']and on['exported_head']['manifest']['published_banks_exact'];assert on['update_schedule']['policy']=='continuous_two_stage';assert not q['diagnostic_only']and not q['audit_replay'];assert all(all(q['results'][arm]['stream']['stats'][k]==0 for k in ['updates','publications','feedback_ns','update_ns','publish_ns'])for arm in ['ar','frozen'])
 for n,s in d['files'].items():(A/n).write_text(s)
 if not(on['net_increment_vs_ar']>.05+1e-12 and on['net_increment']>.05+1e-12):
  save(dict(stage='qualifier_not_dual5',qualification=False,online_vs_ar=on['net_increment_vs_ar'],online_vs_fixed=on['net_increment'],no_reserved_executed=True));raise SystemExit(0)
 frozen=dict(recipe_id='gsm254-step128-compact16-K8-TV1e5-staged256-pub4',qualifying_tag=tag,qualifying_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),online_vs_ar=on['net_increment_vs_ar'],online_vs_fixed=on['net_increment'],selected_case=d['plan']['cases'][0],source_files=json.loads(d['files'][tag+'.source.json'])['files'],initial_head_step=128,initial_head_shards=q['head_shard_sha256'],confirmation=dict(seeds=[743,744,745],requests=384,holdout=32,resource='Eachtrialexact2physicalA2TP2/rank0noextra',execution='SameA111delivery for3independent processes, idlepair pertrial, startupserializedbyreadinessgates; simultaneousflowsnever share optimizer orupdatedweights',criterion='Eachandpooled >5vsARandfixed. All3rununchangedregardless gains. No reservedtuning.',source_delivery='ExistingA248 products+native+benchmark mustbyte-matchB245 canonicalfile mapping. Only noncomputational launcher/Docker paths may differ; Aactualsource recorded andfrozen beforestart.'))
 fp=A/'gsm254-frozen-recipe.json'
 with fp.open('x')as f:json.dump(frozen,f,indent=2)
 P=dict(frozen=frozen,audit=json.loads((A/'gsm254-unused-reserve-audit.json').read_text()),assets={name:(A/(name+'.json')).read_text()for name in ['gsm254-confirm384','gsm254-holdout32']})
 dispatch=r'''from pathlib import Path
import json,hashlib,copy,random,subprocess
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');f=P['frozen'];assert P['audit']['completed']
# Computational code must match the qualifying B flow; A's pinned Docker launcher is a delivery adapter.
canonical={}
for name,sha in f['source_files'].items():
 if 'src-dense-v245/'in name:canonical['source/'+name.split('src-dense-v245/',1)[1]]=sha
 elif name=='orthrus_bench_dense_v245.py':canonical['benchmark']=sha
 elif name.startswith('pypto-serving-qwen3.8-27B/'):canonical[name]=sha
assert canonical and 'benchmark'in canonical
actual={}
for name,expected in canonical.items():
 p=r/'src-dense-v248'/name[len('source/'):]if name.startswith('source/')else r/'orthrus_bench_dense_v248.py'if name=='benchmark'else r.parent/name;sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==expected,name;actual[name]=sha
head=r.parent/'kl-prewarm-suffix-tp2';m=json.loads((head/'manifest.json').read_text());assert m['step']==128 and[x['sha256']for x in m['rank_files']]==f['initial_head_shards']
for part in m['rank_files']:
 h=hashlib.sha256()
 with(head/part['name']).open('rb')as fi:
  for chunk in iter(lambda:fi.read(8<<20),b''):h.update(chunk)
 assert h.hexdigest()==part['sha256']
for name,raw in P['assets'].items():
 p=r/'assets'/(name+'.json');assert hashlib.sha256(raw.encode()).hexdigest()==P['audit']['sets'][name]['sha256']
 if p.exists():assert p.read_text()==raw
 else:p.write_text(raw)
assignment=dict(recipe=f,audit=P['audit'],canonical_computational_source=actual,initial_head_actual_manifest=m,delivery_launcher='run_dense_v248_docker.py',delivery_launcher_sha256=hashlib.sha256((r/'run_dense_v248_docker.py').read_bytes()).hexdigest(),delivery_entry_sha256=hashlib.sha256((r/'docker_entry_v248.py').read_bytes()).hexdigest())
lock=r.parent/'reserved-gsm254-assignment.json'
with lock.open('x')as file:json.dump(assignment,file,indent=2)
(r/'gsm254-frozen-recipe.json').write_text(json.dumps(f,indent=2)+chr(10));(r/'gsm254-unused-reserve-audit.json').write_text(json.dumps(P['audit'],indent=2)+chr(10));(r/'gsm254-preflight.json').write_text(json.dumps(dict(completed=True,canonical_source_matches=True,head128_exact=True,canonical_computational_source=actual,initial_shards=m['rank_files'],configuration_unchanged=True),indent=2)+chr(10));base=json.loads(P['assets']['gsm254-confirm384']);cases=[]
for seed in [743,744,745]:
 order=list(range(384));random.Random(seed).shuffle(order);name=f'gsm254-confirm-order{seed}.json';p=r/'assets'/name
 with p.open('x')as file:file.write(json.dumps([base[i]for i in order],ensure_ascii=False)+chr(10))
 c=copy.deepcopy(f['selected_case']);c.update(tag=f'a111-v248-gsm254-confirm{seed}',launcher='run_dense_v248_docker.py',confirmation_group=f['recipe_id'],seed=seed,requires=[dict(file='gsm254-preflight.json')]+([]if seed==743 else[dict(file=f'gsm254-ready{seed-1}.json')]),max_attempts=1,minimum_available_host_gib=96,allow_parallel_startup=False,purpose='Frozen completequalifier recipe, all384newstrictunused officialGSM trainquestions, freshAdam/head128, 2physicalA2/TP2/rank0/allcontinuouscosts, finalseen+fresh32holdout. Onlyrequestorder/seed andnoncomputational hostdelivery differ. All3unchanged evenifnegative, noreservedtuning.')
 for key,val in [('--head','kl-prewarm-suffix-tp2'),('--request-file',name),('--holdout-file','gsm254-holdout32.json'),('--ar-check-requests',384),('--seed',seed),('--devices','0,1,2,3,4,5,6,7')]:c['args'][c['args'].index(key)+1]=str(val)
 assert not c.get('confirm')and '--audit-replay'not in c['args']and '--background'not in c['args'];c['prompts_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();c['original_problem_order']=order;cases.append(c)
# Readiness allows the next startup only after the prior worker is initialized, never from a lock alone.
watch=r"""from pathlib import Path
import json,time,subprocess,os
r=Path(__file__).parent
for seed in [743,744]:
 p=r/(f'a111-v248-gsm254-confirm{seed}.json')
 while True:
  if p.exists():
   try:d=json.loads(p.read_text())
   except ValueError:time.sleep(1);continue
   if d.get('stage')=='measuring'or d.get('completed')or d.get('error'):break
  time.sleep(5)
 ok=not d.get('error');proof=dict(completed=ok,prior_seed=seed,scope='Prior startup finished, allow another independentidlepair; not performanceacceptance.')
 if not ok:proof['error']=d['error']
 (r/f'gsm254-ready{seed}.json').write_text(json.dumps(proof,indent=2)+chr(10))
"""
compile(watch,'watch_gsm254_startup.py','exec');(r/'watch_gsm254_startup.py').write_text(watch);log=(r/'gsm254-startup-watcher.log').open('x');w=subprocess.Popen(['python3','watch_gsm254_startup.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);launched=[]
for c in cases:
 name=f'queue-gsm254-confirm{c["seed"]}.json'
 with(r/name).open('x')as file:json.dump(dict(cases=[c],automatic_reserved=False,explicit_frozen_confirmation=True,reserved_assignment=str(lock)),file,indent=2)
 log=(r/f'gsm254-confirm{c["seed"]}-controller.log').open('x');p=subprocess.Popen(['python3','study_queue_windows217.py','--plan',name],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);launched.append(dict(pid=p.pid,plan=name,seed=c['seed']))
(r/'gsm254-cases.json').write_text(json.dumps(dict(cases=cases,readiness_watcher=w.pid,controllers=launched),indent=2)+chr(10));print(json.dumps(dict(launched=launched,readiness_watcher=w.pid,canonical_source_match=True,actual_head128=True,first_reserved384_consumption=True)))
'''
 compile(dispatch,'gsm254-dispatch','exec');save(dict(stage='dispatching_frozen_confirmation',qualification=True,frozen_sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
 x=ssh('159.138.5.111','import json\nP=json.loads('+repr(json.dumps(P,ensure_ascii=True))+')\n'+dispatch)
 if x.returncode:raise RuntimeError('Dispatchfailed, inspectremoteactualstate beforeanyretry: '+x.stderr.decode()[-4000:])
 out=json.loads(x.stdout);(A/'gsm254-confirm-launch.json').write_text(json.dumps(out,indent=2)+'\n');save(dict(stage='launched',qualification=True,launch=out,scope='No models restarted fromobservation timeout; eachcontroller watches actualidlepairs, readinessserializes initialization.'));print(json.dumps(out),flush=True)
except BaseException as e:
 if isinstance(e,SystemExit):raise
 save(dict(stage='needs_inspection_no_automatic_redispatch',error=repr(e),traceback=traceback.format_exc()));raise
