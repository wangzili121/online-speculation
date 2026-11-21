from pathlib import Path
import subprocess,concurrent.futures,json,base64,zlib
import argparse
p=argparse.ArgumentParser();p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();Q=a.validation_root/'relocate318'
finger=r'''from pathlib import Path
import json,subprocess,hashlib,os
import pypto.runtime as rt
root=Path(rt.__file__).parent.resolve();files={};seen=set()
for folder,dirs,names in os.walk(root,followlinks=True):
 p=Path(folder);real=p.resolve()
 if real in seen:dirs[:]=[];continue
 seen.add(real);dirs[:]=[d for d in dirs if d!='__pycache__']
 for n in names:
  f=p/n
  if f.is_file()and(f.suffix=='.py'or'.so'in f.name):files[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
git=subprocess.run(['git','-C',str(root),'rev-parse','--show-toplevel'],capture_output=True,text=True);code={};commit=None
if git.returncode==0:
 repo=Path(git.stdout.strip());commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 for name in subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().split(chr(0)):
  f=repo/name
  if name and f.is_file()and(f.suffix in ['.py','.cpp','.h','.hpp','.cc','.cmake']or f.name=='CMakeLists.txt'):code[name]=hashlib.sha256(f.read_bytes()).hexdigest()
loaded={}
for row in Path('/proc/self/maps').read_text().splitlines():
 fields=row.split()
 if len(fields)<6:continue
 p=Path(fields[-1])
 if p.is_file()and'.so'in p.name and ('runtime'in str(p).lower()or'simpler'in str(p).lower()):loaded[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(runtime_module=str(rt.__file__),runtime_files=files,source_commit=commit,tracked_runtime_source=code,loaded_runtime_libraries=loaded)))
'''
inside="from pathlib import Path\nimport sys,subprocess\nb=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment\ne=environment();e['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'\nx=subprocess.run([str(PYTHON),'-c',"+repr(finger)+"],env=e,capture_output=True,text=True);sys.stdout.write(x.stdout);sys.stderr.write(x.stderr);raise SystemExit(x.returncode)"
remote="""from pathlib import Path
import subprocess,json,zlib,sys
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');image='sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2';cmd=['docker','run','--rm','--network','none','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-v',str(b)+':'+str(b)+':ro','-v',str(b/'cann-9.1.1')+':/usr/local/Ascend/cann-9.1.1:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/python3',image,'-c',P['inside']];x=subprocess.run(cmd,capture_output=True,text=True,timeout=90);assert x.returncode==0,x.stderr[-2500:];sys.stdout.buffer.write(zlib.compress(x.stdout.encode()))
"""
def call(host):
 wire='P='+repr(dict(inside=inside))+'\n'+remote;x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@'+host,'python3 -'],input=wire.encode(),capture_output=True,timeout=120);assert x.returncode==0,(host,x.stderr.decode()[-3000:]);d=json.loads(zlib.decompress(x.stdout));(Q/(host+'-extended-runtime.json')).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(host=host,module=d['runtime_module'],runtime_files=len(d['runtime_files']),source_commit=d['source_commit'],source_files=len(d['tracked_runtime_source']),libraries=len(d['loaded_runtime_libraries']))),flush=True);return d
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as p:values=list(p.map(call,['159.138.5.111','189.1.227.167']))
proof=dict(completed=True,passed=values[0]==values[1],all_resolved_runtime_sources_and_libraries_equal=values[0]==values[1],no_NPU_devices_mounted=True);(Q/'extended-runtime-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
