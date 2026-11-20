from pathlib import Path
import sys,subprocess
root=Path(__file__).resolve().parent;base=root.parent.parent
sys.path.insert(0,str(base));from host_runtime import PYTHON,environment,prepare
prepare();env=environment();env['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1];env['ASCEND_AICPU_PATH']='/usr/local/Ascend/cann-9.1.1'
env['PYTHONPATH']=str(root.parent/'pypto-serving-qwen3.8-27B')+':'+str(root/'recurrent-v117/src')+':'+str(root)+':'+env['PYTHONPATH'];env['PYPTO_CACHE_DIR']=str(root/'jit');env['PYPTO_CACHE']='1'
raise SystemExit(subprocess.call([str(PYTHON),*sys.argv[2:]],cwd=root,env=env))
