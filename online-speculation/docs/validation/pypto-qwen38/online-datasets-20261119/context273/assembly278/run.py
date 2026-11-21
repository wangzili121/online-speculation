from pathlib import Path
import sys,subprocess,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-tp2-test';q=b/'context-v278-assembly';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
e=environment();e['PYTHONPATH']=':'.join([str(q),str(b/'context-v275-compile'),str(r/'src-dense-v278'),str(r.parent/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH']]);e.update(PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1')
base=['--context','1024','--device','0'];cases=[('attention-page16','validate_qwen38_pypto_attention.py',base+['--assemble-only','--backend','paged','--page-size','16','--q-heads','12','--kv-heads','2','--runtime-artifacts',str(b/'context275-runtime-manifest.json')]),('attention-page128','validate_qwen38_pypto_attention.py',base+['--assemble-only','--backend','paged','--page-size','128','--q-heads','12','--kv-heads','2','--runtime-artifacts',str(b/'context275-runtime-manifest.json')]),('training1024','validate_qwen38_pypto_training_rows.py',base+['--assemble-only','--model',str(b/'inference-tuning-20261116/model-view'),'--head',str(r.parent/'kl-prewarm-suffix-tp2')]),('full-TP2-entry','benchmark_qwen38_oneserving_tp2_online.py',['--model',str(b/'inference-tuning-20261116/model-view'),'--head',str(r.parent/'kl-prewarm-suffix-tp2'),'--prompts',str(r/'assets/domain229-prewarm2.json'),'--holdout',str(r/'assets/pseudo-holdout.json'),'--compile-only','--max-model-len','1024','--draft-width','8','--candidates','7','--teacher-tokens','7','--last-layers','1','--learner-device','0','--gather-feedback','--training-rows','16','--compact-paired-rows','--kv-pages','32'])]
status=dict(completed=False,npu_execution=False,scope='ONBOARD compiler binary assembly without NPU worker; fullmodel metadata compilation only. No performance measure, no newrequests.',cases=[])
try:
 for name,script,args in cases:
  print('BEGIN',name,flush=True);rc=subprocess.run([str(PYTHON),str(q/script),*args,'--output',str(q/(name+'.json'))],cwd=q,env=e,timeout=1800).returncode;status['cases'].append(dict(name=name,returncode=rc));(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if rc:raise RuntimeError('assembly/entry gate failed: '+name)
 status['completed']=True
finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
