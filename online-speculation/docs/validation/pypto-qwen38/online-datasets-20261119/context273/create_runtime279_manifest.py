from pathlib import Path
import subprocess,json,zlib
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);a=parser.parse_args().artifacts
code=r'''from pathlib import Path
import sys,subprocess,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
s=r"""from pathlib import Path
import json,hashlib,importlib.metadata,pypto
from pypto.runtime.runtime_pin import check_runtime_pin
from pypto.backend._ptoas_locate import find_ptoas_binary
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');root=Path(pypto.__file__).resolve().parents[2];p=check_runtime_pin();assert p.matches and not p.off_pin
files=list((root/'python').rglob('*.py'))+list((root/'runtime/python').rglob('*.py'))+list((root/'python/pypto').glob('*.so'))+list((b/'host-runtime-1/packages/simpler_setup').rglob('*.so'))+[Path(find_ptoas_binary())]
assert p.extension_path is not None;files.append(Path(p.extension_path));files=sorted(set(x.resolve()for x in files));shas={}
for file in files:
 h=hashlib.sha256()
 with file.open('rb')as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 shas[str(file)]=h.hexdigest()
d=dict(format='pypto-export-artifacts-v1',pypto_root=str(root),files=shas,packages={n:importlib.metadata.version(n)for n in ['pypto','ptoas']},simpler_commit=p.installed,scope='Canonical exported runtime onC; compare actual imported package/core native binding/Simpler sharedlibs/PTOAS entry and Python sources bySHA onA. No pin check bypass or Git metadata fabrication.')
with(b/'context279-runtime-manifest.json').open('x')as file:json.dump(d,file,indent=2)
print(json.dumps(d))
"""
x=subprocess.run([str(PYTHON),'-c',s],env=environment(),capture_output=True,text=True);assert x.returncode==0,x.stderr;print(x.stdout)
'''
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@159.138.54.61','python3 -'],input=code.encode(),capture_output=True,timeout=50);assert x.returncode==0,(x.stdout,x.stderr);d=json.loads(x.stdout);(a/'context279-runtime-manifest.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(files=len(d['files']),packages=d['packages'],simpler=d['simpler_commit'])))
