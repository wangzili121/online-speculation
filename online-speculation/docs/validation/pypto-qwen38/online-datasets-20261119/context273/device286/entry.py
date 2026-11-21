from pathlib import Path
import sys,subprocess,json,os
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'oneserving-v109-b99e118/online-studies';q=b/'context-v286';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment,prepare
prepare();e=environment();e.update(ASCEND_RT_VISIBLE_DEVICES=sys.argv[1],ASCEND_AICPU_PATH='/usr/local/Ascend/cann-9.1.1',PYPTO_CACHE_DIR=str(r/'jit'),PYPTO_CACHE='1');e['PYTHONPATH']=':'.join([str(q),str(r/'src-dense-v286'),str(r.parent/'pypto-serving-qwen3.8-27B'),str(r),e['PYTHONPATH']]);rc=subprocess.call([str(PYTHON),str(q/'sequence.py')],cwd=q,env=e)
for file in q.glob('*.json'):
 if file.is_file()and not file.is_symlink():os.chown(file,int(sys.argv[2]),int(sys.argv[3]))
raise SystemExit(rc)
