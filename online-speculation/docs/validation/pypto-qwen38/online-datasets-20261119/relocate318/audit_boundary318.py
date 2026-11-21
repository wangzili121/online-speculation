"""只读核对 B 机旧合成输入诊断；不会启动模型或激活预留题。"""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,subprocess,zlib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--validation-root',type=Path,required=True)
p.add_argument('--artifacts',type=Path,required=True)
p.add_argument('--stage-file',default='stage.json');p.add_argument('--plan-file',default='queue-spoc318B.json')
a=p.parse_args();D=a.validation_root;Q=D/'relocate318';stage=json.loads((Q/a.stage_file).read_text())
plan=json.loads((Q/a.plan_file).read_text());case=plan['cases'][0];tag=case['tag']
read=r"""from pathlib import Path
import hashlib,json,subprocess,sys,zlib
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies')
s=r/P['status'];status=json.loads(s.read_text())if s.exists()else{}
c=next((x for x in status.get('cases',[])if x['tag']==P['tag']),{})
out=dict(status=status,completed=bool(c.get('completed')),error=c.get('error'))
if out['completed']:
 out['files']={P['tag']+ext:(r/(P['tag']+ext)).read_text()for ext in ['.json','.source.json','.log']}
 d=json.loads(out['files'][P['tag']+'.json']);exp=d['results']['4']['exported_head']
 out['actual_published_shards']={f['name']:hashlib.sha256((Path(exp['path'])/f['name']).read_bytes()).hexdigest()for f in exp['manifest']['rank_files']}
 bind=json.loads((r/'spoc318B-base-files.json').read_text());model=Path(bind['model'])
 out['current_base_binding']=dict(metadata_sha256={n:hashlib.sha256((model/n).read_bytes()).hexdigest()for n in bind['metadata_sha256']},weight_files_stat={})
 for n in bind['weight_files_stat']:
  s=(model/n).stat();out['current_base_binding']['weight_files_stat'][n]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
 out['current_revision_marker_sha256']=hashlib.sha256((model/'.cache/huggingface/download/config.json.metadata').read_bytes()).hexdigest()
 out['current_image_id']=json.loads(subprocess.check_output(['docker','image','inspect',P['image']],text=True))[0]['Id']
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
"""
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -'],input=('P='+repr(dict(tag=tag,image=case['expected_image_id'],status=Path(a.plan_file).with_suffix('.status.json').name))+'\n'+read).encode(),capture_output=True,timeout=90)
assert x.returncode==0,x.stderr.decode()[-2000:]
reply=json.loads(zlib.decompress(x.stdout));assert not reply.get('error'),reply.get('error')
if not reply['completed']:
 (a.artifacts/'boundary318-pending.json').write_text(json.dumps(dict(completed=False,status=reply['status']),indent=2)+'\n')
 print(json.dumps(dict(completed=False,status=reply['status'])))
 raise SystemExit(0)
for name,text in reply['files'].items():
 raw=text.encode();suffix='.gz'if name.endswith('.log')or len(raw)>2<<20 else''
 (Q/(name+suffix)).write_bytes(gzip.compress(raw,mtime=0)if suffix else raw)
d=json.loads(reply['files'][tag+'.json'])
bundle=json.loads(gzip.decompress((Q/'source292_A_bundle.json.gz').read_bytes()))
for flag,key in [('--request-file','prompts_sha256'),('--holdout-file','holdout_sha256')]:
 name=case['args'][case['args'].index(flag)+1]
 assert d[key]==hashlib.sha256(bundle['assets'][name].encode()).hexdigest(),name

sp=importlib.util.spec_from_file_location('accounting318',D/'rows255/audit_accounting.py');acc=importlib.util.module_from_spec(sp);sp.loader.exec_module(acc)
proof=acc.audit(d,4,allow_replay=True)
assert d['requests']==d['ar_checked_requests']==2
assert d['physical_device_count']==2 and len(set(d['visible_devices'].split(',')))==2 and d['device_ids']==[0,1] and d['learner_device']==0 and d['background'] is False
assert d['head_step']==128 and d['head_shard_sha256']==case['expected_initial_head_shards']and d['optimizer_init']=='fresh'
for k,v in dict(training_rows=16,max_model_len=1024,output_tokens_per_request=24,last_layers=1,parameter_scope='full',parameterization='dense',draft_width=8,candidates=7,teacher_tokens=7,objective='kl',learning_rate=3e-5,warmup_updates=4,warm_update_stride=2,warm_update_steps=4,lr_schedule='constant',replay_blocks=1,publish_every=1,inner_steps=1,teacher_targets='greedy',teacher_temperature=1,teacher_scaling='host',feedback_mode='delayed',feedback_timing='early',publication_backend='device',temperature=0,stop_on_eos=False,paired_physical_rows=16,draft_convolution='parallel128',gdn_valid_rows=True).items():
 assert d[k]==v,(k,d[k],v)
assert json.loads(reply['files'][tag+'.source.json'])['files']==stage['source']
assert reply['status']['controller_sha256']==hashlib.sha256((D/'spoc312/study_queue_spoc312.py').read_bytes()).hexdigest()
assert reply['current_base_binding']=={k:stage['base'][k]for k in ['metadata_sha256','weight_files_stat']}
assert reply['current_image_id']==case['expected_image_id']
if stage.get('marker_restoration'):assert reply['current_revision_marker_sha256']==stage['marker_restoration']['sha256']
for proofname in ['model-runtime-equivalence.json','extended-runtime-equivalence.json']:
 assert json.loads((Q/proofname).read_text())['passed']
on=d['results']['4'];m=on['exported_head']['manifest']
assert m['published_banks_exact']and m['step']==128+on['published_version']and on['optimizer_version']==len(on['updates'])
assert on['published_version']==on['stream']['stats']['publications']
assert reply['actual_published_shards']=={v['name']:v['sha256']for v in m['rank_files']}
replays=[x for u in on['updates']for x in u.get('replay_audits',[])]
assert replays and all(x['served_replay_tv']<=.01 and x['relative_l2']<=.01 for x in replays)
for arm in ['ar','frozen','4']:
 assert [n for row in d['results'][arm]['request_timeline']for n in row['output_tokens']]==[24,24]
proof.update(completed=True,tag=tag,execution_host='189.1.227.167',raw_sha256=hashlib.sha256(reply['files'][tag+'.json'].encode()).hexdigest(),source_base_head_runtime_checked=True,actual_published_shards=reply['actual_published_shards'],replay_audits=replays,performance_qualification=False)
(Q/'boundary-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
