"""Stage separate three-card SPoC controls using the frozen two-card numerical recipe."""
from pathlib import Path
import json,copy,subprocess,hashlib,gzip,zlib
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();D=a.validation_root;Q=D/'aux333';Q.mkdir(exist_ok=True)
base=json.loads((D/'spoc312/queue-spoc312-confirm743.json').read_text())['cases'][0]
gate=json.loads((D/'relocate318/queue-spoc323B.json').read_text())['cases'][0]
explore=next(x for x in json.loads((D/'context273/recovery301/queue-explore301-spoc.json').read_text())['cases']if'kl-after301'in x['tag'])
freeze=json.loads((D/'spoc312/frozen-recipe.json').read_text());cases=[]
for old,tag,phase in [(gate,'a111-v292-spoc333-aux3card-context1024-boundary','gate'),(explore,'a111-v292-spoc333-aux3card-explore64','explore'),(base,'a111-v292-spoc333-aux3card-confirmed256-order743','auxiliary')]:
 c=copy.deepcopy(old);c.update(tag=tag,max_attempts=1,minimum_available_host_gib=96,allow_parallel_startup=False,requires=[dict(file='spoc333-preflight.json')],expected_source_manifest='spoc312-frozen-source.json',expected_base_manifest='spoc312-base-files.json',expected_image_tag='quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc',expected_image_id=freeze['expected_image_id'],expected_initial_head='kl-prewarm-suffix-tp2',expected_initial_head_shards=freeze['initial_head_shards'])
 for key in ['confirm','confirmation_group','unless_achieved']:c.pop(key,None)
 args=c['args']
 for flag in ['--interleaved','--interleave-ar']:
  if flag in args:args.remove(flag)
 for flag,value in [('--learner-device','2'),('--devices','2,3,5,6,7'),('--wait','28800')]:args[args.index(flag)+1]=value
 if '--background'not in args:args.append('--background')
 if phase!='gate':c['requires'].append(dict(file='a111-v292-spoc333-aux3card-context1024-boundary.json',greedy=True,audit=True))
 if phase=='auxiliary':c['requires'].append(dict(file='a111-v292-spoc333-aux3card-explore64.json',greedy=True))
 c['purpose']='Three physical A2, TP2 target/draft plus separate learner. All AR/fixed/continuous/finalpublishedseen/holdout stages reserve the same three physical cards. source292 and numerical model/head/Adam/generation/update recipe unchanged; only learner placement/background execution and separate-stream timing differ. Gate uses old synthetic fixture; exploratory64 uses old data; auxiliary256 replays already-assigned primary cohort and is NOT a fresh-question confirmation or a replacement for any two-card proof. No retuning, no auto confirmation, no stopping other tasks.'
 cases.append(c)
plan=dict(cases=cases,automatic_reserved=False,primary_qualification=False,physical_cards=3)
remote=r'''from pathlib import Path
import json,hashlib,subprocess
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');assert not(r/'queue-spoc333aux.json').exists()
for seed in [743,744]:
 d=json.loads((r/('a111-v292-spoc312-confirm'+str(seed)+'.json')).read_text());assert d['completed']and d['clean_shutdown']and not d.get('error')
actual={}
for name,sha in P['source'].items():
 f=(r if '/'not in name else r.parent)/name;actual[name]=hashlib.sha256(f.read_bytes()).hexdigest();assert actual[name]==sha,name
image=json.loads(subprocess.check_output(['docker','image','inspect',P['plan']['cases'][0]['expected_image_tag']],text=True))[0]['Id'];assert image==P['plan']['cases'][0]['expected_image_id']
head=r.parent/'kl-prewarm-suffix-tp2'
for i,sha in enumerate(P['plan']['cases'][0]['expected_initial_head_shards']):assert hashlib.sha256((head/f'rank{i}.safetensors').read_bytes()).hexdigest()==sha
assets={}
for c in P['plan']['cases']:
 for flag in ['--request-file','--holdout-file']:
  name=c['args'][c['args'].index(flag)+1];assets[name]=hashlib.sha256((r/'assets'/name).read_bytes()).hexdigest()
pre=dict(completed=True,source_byte_exact=True,initial_head_rehashed=True,image_id=image,NN_math_changed=False,physical_cards=3,primary_two_card_trials_unchanged=True,assigned256_is_auxiliary_replay=True)
with(r/'spoc333-preflight.json').open('x')as f:json.dump(pre,f,indent=2)
with(r/'queue-spoc333aux.json').open('x')as f:json.dump(P['plan'],f,indent=2)
with(r/'queue-spoc333aux.json.controller.log').open('x')as log:p=subprocess.Popen(['python3','study_queue_spoc312.py','--plan','queue-spoc333aux.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,preflight=pre,source=actual,assets=assets,base=json.loads((r/'spoc312-base-files.json').read_text()))))
'''
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@159.138.5.111','python3 -'],input=('P='+repr(dict(plan=plan,source=freeze['source_files']))+'\n'+remote).encode(),capture_output=True,timeout=40);assert x.returncode==0,x.stderr.decode();result=json.loads(x.stdout)
for name,data in [('launch.json',result),('queue-spoc333aux.json',plan)]:(Q/name).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(controller_pid=result['pid'],physical_cards=3,cases=[c['tag']for c in cases],primary_confirmation_unchanged=True)))
