from pathlib import Path
import sys,subprocess,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-tp2-test';q=b/'context-v291-contract';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
e=environment();e['PYTHONPATH']=':'.join([str(r/'src-dense-v291'),str(r.parent/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH']]);e.update(PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1');status=dict(completed=False,npu_execution=False,cases=[])
try:
 commands=[('page-contract',[str(PYTHON),str(q/'validate_qwen38_pypto_page_contract.py'),'--output',str(q/'page-contract.json')]),('CPU-regressions',[str(PYTHON),'-m','pytest','-q',str(b/'scope-v285-review/tests'),'-o','addopts=']),('full-TP2-entry',[str(PYTHON),str(b/'context-v278-assembly/benchmark_qwen38_oneserving_tp2_online.py'),'--model',str(b/'inference-tuning-20261116/model-view'),'--head',str(r.parent/'kl-prewarm-suffix-tp2'),'--prompts',str(r/'assets/domain229-prewarm2.json'),'--holdout',str(r/'assets/pseudo-holdout.json'),'--compile-only','--max-model-len','1024','--draft-width','8','--candidates','7','--teacher-tokens','7','--last-layers','1','--learner-device','0','--gather-feedback','--training-rows','16','--compact-paired-rows','--kv-pages','32','--output',str(q/'full-TP2-entry.json')])]
 for name,cmd in commands:
  print('BEGIN',name,flush=True);rc=subprocess.run(cmd,cwd=q,env=e,timeout=1200).returncode;status['cases'].append(dict(name=name,returncode=rc));(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if rc:raise RuntimeError('CPU gate failed: '+name)
 status['completed']=True
finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
