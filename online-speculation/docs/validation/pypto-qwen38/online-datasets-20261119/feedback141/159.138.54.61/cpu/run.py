from pathlib import Path
import sys,subprocess,json,hashlib
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-tp2-test');q=b/'scope-v141-review';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
e=environment();e['PYTHONPATH']=str(r/'src-dense-v141')+':'+str(r.parent/'pypto-serving-qwen3.8-27B')+':'+e['PYTHONPATH'];e['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
(q/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in list((r/'src-dense-v141').rglob('*.py'))+list((q/'tests').rglob('*.py'))+list((q/'scripts').glob('*.py'))},indent=2)+'\n')
with(q/'tests.log').open('x')as f:x=subprocess.run([str(PYTHON),'-m','pytest','-q',str(q/'tests'),'-o','addopts='],cwd=q,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=180)
d=dict(completed=x.returncode==0,returncode=x.returncode,log=(q/'tests.log').read_text());(q/'report.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
