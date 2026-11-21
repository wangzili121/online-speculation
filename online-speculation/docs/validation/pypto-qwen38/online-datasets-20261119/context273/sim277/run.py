from pathlib import Path
import sys,subprocess,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-tp2-test';q=b/'context-v277-sim';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
e=environment();e['PYTHONPATH']=':'.join([str(b/'context-v275-compile'),str(r/'src-dense-v273'),str(r.parent/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH']]);e.update(PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1');status=dict(completed=False,npu_execution=False,cases=[])
try:
 for page in [16,128]:
  command=[str(PYTHON),str(b/'context-v275-compile/validate_qwen38_pypto_attention.py'),'--backend','paged','--context','1024','--q-heads','12','--kv-heads','2','--page-size',str(page),'--query-tile','16','--runtime-artifacts',str(b/'context275-runtime-manifest.json'),'--simulate','--device','0','--repeats','3','--output',str(q/('attention-page'+str(page)+'.json'))]
  rc=subprocess.run(command,cwd=q,env=e,timeout=900).returncode;status['cases'].append(dict(page_size=page,returncode=rc));(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if rc:raise RuntimeError('simulator gate failed')
 status['completed']=True
finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
