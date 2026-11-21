from pathlib import Path
import sys,subprocess,os,json
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-studies';q=b/'context-v273';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment,prepare
prepare();e=environment();e.update(ASCEND_RT_VISIBLE_DEVICES=sys.argv[1],ASCEND_AICPU_PATH='/usr/local/Ascend/cann-9.1.1',PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1');e['PYTHONPATH']=':'.join([str(q),str(r/'src-dense-v273'),str(r.parent/'pypto-serving-qwen3.8-27B'),str(r),e['PYTHONPATH']])
cases=[('attention-page16',['validate_attention.py','--backend','paged','--context','1024','--q-heads','12','--kv-heads','2','--page-size','16','--repeats','5']),('attention-page128',['validate_attention.py','--backend','paged','--context','1024','--q-heads','12','--kv-heads','2','--page-size','128','--repeats','5']),('training1024',['validate_training.py','--context','1024','--model','/data/disk/models/Qwen3.8-27B','--head',str(r.parent/'kl-prewarm-suffix-tp2')])]
status=dict(completed=False,cases=[])
try:
 for name,args in cases:
  bootstrap="from _guard_bootstrap import Guard; import runpy; g=Guard();\ntry:runpy.run_path("+repr(str(q/args[0]))+",run_name='__main__')\nfinally:g.close()"
  command=[str(PYTHON),'-c',bootstrap,*args[1:],'--device','0','--output',str(q/(name+'.json'))]
  print('BEGIN',name,flush=True);rc=subprocess.call(command,cwd=q,env=e);status['cases'].append(dict(name=name,returncode=rc))
  (q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if rc:raise RuntimeError('component gate failed: '+name)
 status['completed']=True
finally:
 (q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 for file in q.glob('*.json'):
  if file.is_file()and not file.is_symlink():os.chown(file,int(sys.argv[2]),int(sys.argv[3]))
