from pathlib import Path
import sys,subprocess,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-tp2-test';q=b/'context-v275-compile';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
# Compile-only does not create a worker, reserve memory or acquire an NPU.
e=environment();e['PYTHONPATH']=':'.join([str(q),str(r/'src-dense-v273'),str(r.parent/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH']]);e.update(PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1')
cases=[('attention-page16','validate_qwen38_pypto_attention.py',['--backend','paged','--context','1024','--page-size','16','--q-heads','12','--kv-heads','2','--runtime-artifacts',str(b/'context275-runtime-manifest.json')]),('attention-page128','validate_qwen38_pypto_attention.py',['--backend','paged','--context','1024','--page-size','128','--q-heads','12','--kv-heads','2','--runtime-artifacts',str(b/'context275-runtime-manifest.json')]),('training1024','validate_qwen38_pypto_training_rows.py',['--context','1024','--model',str(b/'inference-tuning-20261116/model-view'),'--head',str(r.parent/'kl-prewarm-suffix-tp2')])]
status=dict(completed=False,npu_execution=False,scope='CPU compile-only, not device validation or throughput',cases=[])
try:
 for name,script,args in cases:
  print('BEGIN',name,flush=True);child=subprocess.run([str(PYTHON),str(q/script),*args,'--device','0','--compile-only','--output',str(q/(name+'.json'))],cwd=q,env=e,timeout=1800);status['cases'].append(dict(name=name,returncode=child.returncode));(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if child.returncode:raise RuntimeError('compile gate failed: '+name)
 status['completed']=True
finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
