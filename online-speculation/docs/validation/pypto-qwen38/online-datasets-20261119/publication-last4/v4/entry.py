from pathlib import Path
import sys,subprocess
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment,prepare
prepare();e=environment();e['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1];e['PUB_PHYSICAL_CARD']=sys.argv[1];e['ASCEND_AICPU_PATH']='/usr/local/Ascend/cann-9.1.1';e['PYPTO_CACHE_DIR']=str(b/'pub-suffix-npu-v4/jit');e['PYPTO_CACHE']='1'
raise SystemExit(subprocess.call([str(PYTHON),str(b/'pub-suffix-npu-v4/micro.py')],env=e,cwd=b/'pub-suffix-npu-v4'))
