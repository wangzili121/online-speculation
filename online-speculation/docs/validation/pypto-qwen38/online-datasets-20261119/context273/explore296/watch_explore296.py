"""只读收集完整探索结果；不启动模型，也不派发预留题。"""
from pathlib import Path
import argparse,subprocess,json,zlib,time,hashlib,importlib.util,traceback,gzip
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--validation-root',type=Path,required=True);args=parser.parse_args();A,D=args.artifacts,args.validation_root
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
q=D/'context273/explore296';q.mkdir(exist_ok=True)
plans=[json.loads((D/'context273/service292'/('queue-explore296-'+group+'.json')).read_text())for group in ['longproc','spoc']];cases=[c for plan in plans for c in plan['cases']];cache={};observations=[]
spec=importlib.util.spec_from_file_location('accounting',D/'rows255/audit_accounting.py');acct=importlib.util.module_from_spec(spec);spec.loader.exec_module(acct)
read=r'''from pathlib import Path
import json,zlib,sys,hashlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');p=r/(P['tag']+'.json');d=json.loads(p.read_text())if p.exists()else{};state=r/('queue-explore296-'+P['group']+'.status.json');ctrl=json.loads(state.read_text())if state.exists()else{};case=next((c for c in ctrl.get('cases',[])if c['tag']==P['tag']),{});out=dict(tag=P['tag'],stage=d.get('stage'),error=d.get('error')or case.get('error'),terminal=bool(case.get('completed')or case.get('error')),model_complete=bool(d.get('completed')and d.get('clean_shutdown')),progress={k:(v[-1]if isinstance(v,list)and v else v)for k,v in d.get('progress',{}).items()})
if out['terminal']:
 out['files']={P['tag']+e:(r/(P['tag']+e)).read_text()for e in ['.json','.source.json','.log']if(r/(P['tag']+e)).exists()};out['actual_published_shards']={}
 if out['model_complete']:
  on=d['results']['64']
  for part in on['exported_head']['manifest']['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
   out['actual_published_shards'][part['name']]=h.hexdigest()
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
def check(c,item):
 assert item['model_complete']and not item.get('error'),item.get('error');raw=item['files'][c['tag']+'.json'];d=json.loads(raw);p=acct.audit(d,64);assert d['physical_device_count']==2 and d['learner_device']==0 and not d['background']and len(set(d['visible_devices'].split(',')))==2;assert d['max_model_len']==1024 and d['training_rows']==16 and d['last_layers']==1 and d['parameter_scope']=='full';assert d['prompts_sha256']==c['prompts_sha256'];assert d['head_step']==(135 if'head135'in c['tag']else 128)and d['optimizer_init']=='fresh';assert d['draft_width']==8 and d['candidates']==7 and d['teacher_tokens']==7;assert d['publish_every']==4 and d['warm_update_stride']==16 and d['warm_update_steps']==64 and d['temperature']==0 and d['stop_on_eos'];assert d['requests']==(30 if'longproc30'in c['tag']else 64);on=d['results']['64'];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==d['head_step']+on['published_version'];assert item['actual_published_shards']=={x['name']:x['sha256']for x in m['rank_files']};src=json.loads(item['files'][c['tag']+'.source.json']);ref=json.loads((D/'context273/service292/a111-v292-context1024-trainingrows16-KL-replay.source.json').read_text());assert src['files']==ref['files'];p.update(completed=True,tag=c['tag'],raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),source_exact=True,actual_published_shards=item['actual_published_shards'],both_5_exploration_gate=p['online_vs_ar']>.05 and p['online_vs_fixed']>.05);return p
while len(cache)<len(cases):
 observations=[]
 for c in cases:
  if c['tag']in cache:continue
  try:
   P=dict(tag=c['tag'],group='longproc'if'longproc'in c['tag']else'spoc');x=subprocess.run([*ssh,'wangzili@159.138.5.111','python3 -'],input=('P='+repr(P)+'\n'+read).encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode()[-1200:];item=json.loads(zlib.decompress(x.stdout));observations.append({k:v for k,v in item.items()if k!='files'})
   if item['terminal']:
    for n,s in item['files'].items():
     b=s.encode();suffix='.gz'if len(b)>2<<20 else'';(q/(n+suffix)).write_bytes(gzip.compress(b,mtime=0)if suffix else b)
    try:proof=check(c,item)
    except BaseException as e:proof=dict(completed=False,tag=c['tag'],error=repr(e),traceback=traceback.format_exc(),model_error=item.get('error'))
    cache[c['tag']]=proof;(q/(c['tag']+'.audit.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
  except (subprocess.TimeoutExpired,AssertionError,ValueError,zlib.error)as e:observations.append(dict(tag=c['tag'],read_error=repr(e),terminal=False))
 state=dict(completed=len(cache)==len(cases),terminal_cases=list(cache),results=list(cache.values()),observations=observations,automatic_reserved=False,scope='Independent full exploration only; no confirmation dispatched or recipe changed by this read-only watcher.')
 (A/'context296-exploration-watcher-state.json').write_text(json.dumps(state,indent=2)+'\n')
 if len(cache)<len(cases):time.sleep(30)
(q/'exploration-audit.json').write_text(json.dumps(state,indent=2)+'\n');(A/'context296-exploration-audit.json').write_text(json.dumps(state,indent=2)+'\n')
