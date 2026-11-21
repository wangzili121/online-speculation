"""只读归档双卡 32/16 行训练布局的完整流对照。"""
from pathlib import Path
import argparse,json,subprocess,zlib,time,hashlib,importlib.util,traceback
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();A,D=a.artifacts,a.validation_root;Q=D/'rows257/service300';Q.mkdir(exist_ok=True);cache={};tags=[f'b167-v265-full-pseudo54-KL-trainingrows{rows}'for rows in [32,16]]
read=r'''from pathlib import Path
import json,zlib,hashlib,sys
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-background');s=r/'queue-training269-fullflows.status.json';state=json.loads(s.read_text())if s.exists()else{};c=next((c for c in state.get('cases',[])if c['tag']==P),{});p=r/(P+'.json');d=json.loads(p.read_text())if p.exists()else{};out=dict(tag=P,terminal=bool(c.get('completed')or c.get('error')),completed=bool(d.get('completed')and d.get('clean_shutdown')),error=c.get('error')or d.get('error'),stage=d.get('stage'))
if out['terminal']:
 out['files']={P+ext:(r/(P+ext)).read_text()for ext in ['.json','.source.json','.log']if(r/(P+ext)).exists()};out['actual_published_shards']={}
 if out['completed']:
  on=d['results']['64']
  for part in on['exported_head']['manifest']['rank_files']:
   h=hashlib.sha256()
   with(Path(on['exported_head']['path'])/part['name']).open('rb')as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
   out['actual_published_shards'][part['name']]=h.hexdigest()
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''
while len(cache)<2:
 observations=[]
 for tag in tags:
  if tag in cache:continue
  try:
   x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -'],input=('P='+repr(tag)+'\n'+read).encode(),capture_output=True,timeout=60);assert x.returncode==0,x.stderr.decode()[-1200:];item=json.loads(zlib.decompress(x.stdout));observations.append({k:v for k,v in item.items()if k!='files'})
   if item['terminal']:
    cache[tag]=item
    for name,s in item['files'].items():(Q/name).write_text(s)
    (Q/(tag+'.actual-shards.json')).write_text(json.dumps(item['actual_published_shards'],indent=2)+'\n')
  except (subprocess.TimeoutExpired,AssertionError,ValueError,zlib.error)as e:observations.append(dict(tag=tag,read_error=repr(e),terminal=False))
 (A/'training300-watcher-state.json').write_text(json.dumps(dict(terminal_cases=list(cache),observations=observations,read_only=True),indent=2)+'\n')
 if len(cache)<2:time.sleep(30)
proof=dict(completed=False,cases=[],scope='Two physicalA2/TP2/learner0, original complete54 exploratory codequestions. Source265/defaultcontext512/head135/freshAdam/configsame, only trainingrows32versus16. Allcosts, not new reserved confirmations.')
try:
 spec=importlib.util.spec_from_file_location('accounting',D/'rows255/audit_accounting.py');acct=importlib.util.module_from_spec(spec);spec.loader.exec_module(acct);raws=[];sources=[]
 for rows,tag in zip([32,16],tags):
  item=cache[tag];assert item['completed']and not item.get('error'),item.get('error');d=json.loads(item['files'][tag+'.json']);v=acct.audit(d,64);assert d['requests']==54 and d['physical_device_count']==2 and d['learner_device']==0 and not d['background'];assert d['training_rows']==rows and d['head_step']==135 and d['optimizer_init']=='fresh'and d['last_layers']==1 and d['objective']=='kl'and d['teacher_targets']=='greedy';assert d.get('max_model_len',512)==512 and d['draft_width']==8 and d['candidates']==7 and d['paired_physical_rows']==16;on=d['results']['64'];m=on['exported_head']['manifest'];assert m['published_banks_exact']and m['step']==135+on['published_version'];assert item['actual_published_shards']=={p['name']:p['sha256']for p in m['rank_files']};v.update(tag=tag,training_rows=rows,raw_sha256=hashlib.sha256(item['files'][tag+'.json'].encode()).hexdigest(),actual_published_shards=item['actual_published_shards'],mean_update_ms=on['stream']['stats']['update_ns']/len(on['updates'])/1e6,updates_seconds=on['stream']['stats']['update_ns']/1e9);proof['cases'].append(v);raws.append(d);sources.append(json.loads(item['files'][tag+'.source.json'])['files'])
 assert sources[0]==sources[1];assert raws[0]['head_shard_sha256']==raws[1]['head_shard_sha256'];assert raws[0]['prompts_sha256']==raws[1]['prompts_sha256']and raws[0]['holdout_sha256']==raws[1]['holdout_sha256'];on=[d['results']['64']for d in raws];keys=['round','loss','supervised','learning_rate','gradient_norm','clip_scale','interval'];trace=[[{k:u[k]for k in keys}for u in v['updates']]for v in on];same=trace[0]==trace[1];same_shards=cache[tags[0]]['actual_published_shards']==cache[tags[1]]['actual_published_shards'];proof.update(completed=True,both_full_chain_checks_passed=True,update_trajectory_bitwise_equal=same,actual_published_shards_equal=same_shards,source_exact=True,online_rate_ratio=on[1]['stream']['tok_s']/on[0]['stream']['tok_s'],mean_update_ratio=proof['cases'][1]['mean_update_ms']/proof['cases'][0]['mean_update_ms'],direct_AB_only=True)
 if not same:proof['trajectory_difference_summary']=dict(lengths=[len(x)for x in trace],first_differences=[dict(index=i,old=x,new=y)for i,(x,y)in enumerate(zip(*trace))if x!=y][:3],interpretation='Different trajectories are not a proof of mathematical layout inequivalence: inspect feedback/admission and parameter gradients before assigning cause. Do not claim bitwise wholeflow parity.')
except BaseException as e:proof.update(error=repr(e),traceback=traceback.format_exc())
(Q/'fullflow-validation.json').write_text(json.dumps(proof,indent=2)+'\n');(A/'training300-fullflow-validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
