"""Read-only full checkpoint and runtime identity for a possible unstarted-trial relocation."""
from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,zlib,base64,datetime
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();D=a.validation_root;A=a.artifacts;Q=D/'relocate318';Q.mkdir(exist_ok=True)
fingerprint=r'''from pathlib import Path
import json,hashlib,importlib.util,importlib.metadata as m,os,shutil,sys
spec=importlib.util.find_spec('pypto');root=Path(spec.origin).parent
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()and(p.suffix=='.py'or'.so'in p.name)and'__pycache__'not in p.parts}
versions={}
for n in ['torch','transformers','numpy','safetensors','pypto']:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
bin=shutil.which('ptoas');selected={}
for directory in os.environ.get('LD_LIBRARY_PATH','').split(':'):
 p=Path(directory)
 if not p.is_dir():continue
 for f in p.glob('*'):
  if f.is_file()and'.so'in f.name and any(k in f.name.lower()for k in ['simpler','pto_runtime']):selected[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(dict(python=sys.version.split()[0],versions=versions,pypto_files=files,ptoas_sha256=hashlib.sha256(Path(bin).read_bytes()).hexdigest()if bin else None,extra_runtime_libraries=selected)))
'''
inside="from pathlib import Path\nimport sys,subprocess,os\nb=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment\ne=environment();e['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'\nx=subprocess.run([str(PYTHON),'-c',"+repr(fingerprint)+"],env=e,capture_output=True,text=True);sys.stdout.write(x.stdout);sys.stderr.write(x.stderr);raise SystemExit(x.returncode)\n"
remote=r'''from pathlib import Path
import json,hashlib,subprocess,zlib,sys,datetime
started=datetime.datetime.now(datetime.timezone.utc).isoformat();b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');model=Path('/data/disk/models/Qwen3.8-27B');idx=json.loads((model/'model.safetensors.index.json').read_text());names=sorted(set(idx['weight_map'].values()))+['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','generation_config.json','vocab.json','merges.txt','chat_template.jinja'];hashes={}
for n in names:
 h=hashlib.sha256()
 with(model/n).open('rb')as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 hashes[n]=h.hexdigest()
image='sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2';args=['docker','run','--rm','--network','none','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-v',str(b)+':'+str(b)+':ro','-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3',image,'-c',P['inside']];x=subprocess.run(args,capture_output=True,text=True,timeout=120);assert x.returncode==0,x.stderr[-3000:];runtime=json.loads(x.stdout);result=dict(completed=True,started_UTC=started,finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),all26_loading_files_sha256=hashes,runtime=runtime,image_id=image,no_NPU_devices_mounted=True,scope='CPU-only Docker fingerprint, model bytes read only, no NPU context or model execution. Existing studies unchanged.');sys.stdout.buffer.write(zlib.compress(json.dumps(result).encode()))
'''
def one(host):
 code='P='+repr(dict(inside=inside))+'\n'+remote
 x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@'+host,'ionice -c 3 python3 -'],input=code.encode(),capture_output=True,timeout=600);assert x.returncode==0,(host,x.stderr.decode()[-3500:]);d=json.loads(zlib.decompress(x.stdout));(Q/(host+'-identity.json')).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(host=host,completed=True,files=len(d['all26_loading_files_sha256']),runtime_files=len(d['runtime']['pypto_files']))),flush=True);return host,d
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:hosts=dict(pool.map(one,['159.138.5.111','189.1.227.167']))
x,y=hosts.values();proof=dict(completed=True,all26_model_files_byte_exact=x['all26_loading_files_sha256']==y['all26_loading_files_sha256'],runtime_byte_exact=x['runtime']==y['runtime'],same_pinned_image=x['image_id']==y['image_id'],no_NPU_execution=True,scope='Preflight only; does not cancel A745 or activate B reserved questions.');proof['passed']=all(proof[k]for k in ['all26_model_files_byte_exact','runtime_byte_exact','same_pinned_image']);(Q/'model-runtime-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
if proof['passed']:
 code="from pathlib import Path\nimport json\nr=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies')\nwith(r/'spoc318B-model-runtime-equivalence.json').open('x')as f:json.dump(P,f,indent=2)\nprint('B diagnostic gate released; existing auxiliary must finish first; reserved NN still waits A handoff')\n"
 x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -'],input=('P='+repr(proof)+'\n'+code).encode(),capture_output=True,timeout=30);assert x.returncode==0,x.stderr;print(x.stdout.decode(),flush=True)
