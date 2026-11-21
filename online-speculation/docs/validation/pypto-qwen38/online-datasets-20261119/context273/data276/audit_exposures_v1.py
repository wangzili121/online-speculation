from pathlib import Path
import subprocess,json,zlib,hashlib
from concurrent.futures import ThreadPoolExecutor
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);a=parser.parse_args().artifacts
scan=r'''from pathlib import Path
import json,hashlib,zlib,sys
r=Path(ROOT);asset_index={};by_name={};unknown=[];used_names=set();used_hashes=set();tags=set();direct=[]
for p in (r/'assets').glob('*.json'):
 try:d=json.loads(p.read_text())
 except (ValueError,OSError):continue
 if isinstance(d,list)and all(isinstance(x,str)for x in d):
  item=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),texts_sha256=[hashlib.sha256(x.encode()).hexdigest()for x in d],normalized_sha256=[hashlib.sha256(' '.join(x.split()).encode()).hexdigest()for x in d]);by_name[p.name]=item;asset_index.setdefault(item['sha256'],[]).append(item)
for p in r.glob('queue*.json'):
 try:plan=json.loads(p.read_text())
 except (ValueError,OSError):continue
 for c in plan.get('cases',[]):
  if not isinstance(c,dict)or not isinstance(c.get('args'),list):continue
  tag=c.get('tag')
  if not isinstance(tag,str)or not((r/(tag+'.source.json')).exists()or(r/(tag+'.json')).exists()):continue
  tags.add(tag);args=c['args']
  for flag in ['--request-file','--holdout-file']:
   if flag not in args:continue
   n=args[args.index(flag)+1]
   if n not in by_name:unknown.append(dict(tag=tag,flag=flag,asset=n,reason='launchedplanasset missing/nontext'));continue
   used_names.add(n)
for p in r.glob('*.json'):
 try:d=json.loads(p.read_text())
 except (ValueError,OSError):continue
 if not isinstance(d,dict)or not isinstance(d.get('prompts_sha256'),str):continue
 tags.add(p.stem)
 for key in ['prompts_sha256','holdout_sha256']:
  sha=d.get(key)
  if not sha:continue
  if sha not in asset_index:unknown.append(dict(tag=p.stem,field=key,sha256=sha,reason='rawreporthash not in currenttextasset index'));continue
  for item in asset_index[sha]:used_names.add(item['file'])
 direct.append(p.stem)
for n in used_names:
 used_hashes.update(by_name[n]['texts_sha256']);used_hashes.update(by_name[n]['normalized_sha256'])
pre=r.parent.parent/'online-study-20261111/kl-answer-last4-k32-128-r1.json';parent=[]
def strings(x):
 if isinstance(x,str):parent.extend([hashlib.sha256(x.encode()).hexdigest(),hashlib.sha256(' '.join(x.split()).encode()).hexdigest()])
 elif isinstance(x,list):
  for v in x:strings(v)
 elif isinstance(x,dict):
  for v in x.values():strings(v)
if pre.exists():strings(json.loads(pre.read_text()));used_hashes.update(parent)
out=dict(completed=not unknown,host_root=str(r),launched_tags=len(tags),raw_reports=len(direct),used_assets=[{k:v for k,v in by_name[n].items()if k not in ['texts_sha256','normalized_sha256']}for n in sorted(used_names)],used_question_sha256=sorted(used_hashes),unknown_assets=unknown,parent128_present=pre.exists(),parent128_sha256=hashlib.sha256(pre.read_bytes()).hexdigest()if pre.exists()else None,scope='Read-only allqueuedlaunchedsource/result assets plusALLrawreports bearingpromptSHA; includeholdouts andparent128strings. Noanswers/generation/performance selection. No newreservedassignment.')
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
def call(item):
 host,root=item;code=scan.replace('ROOT',repr('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/'+root));x=subprocess.run([*ssh,'wangzili@'+host,'python3 -'],input=code.encode(),capture_output=True,timeout=55);assert x.returncode==0,x.stderr.decode();d=json.loads(zlib.decompress(x.stdout));print(json.dumps(dict(host=host,completed=d['completed'],tags=d['launched_tags'],raw=d['raw_reports'],assets=len(d['used_assets']),unknown=d['unknown_assets'])),flush=True);return host,d
with ThreadPoolExecutor(max_workers=3)as pool:hosts=dict(pool.map(call,[('159.138.5.111','online-studies'),('189.1.227.167','online-background'),('159.138.54.61','online-tp2-test')]))
report=dict(completed=all(d['completed']for d in hosts.values()),hosts=hosts,consumed_pseudo253_reserved24_excluded=True,no_newNNrequests=True)
(a/'pseudo274-crosshost-exposure-audit.json').write_text(json.dumps(report,indent=2)+'\n')
