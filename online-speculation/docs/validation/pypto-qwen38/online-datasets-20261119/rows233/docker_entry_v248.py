from pathlib import Path
import sys,subprocess
root=Path(__file__).resolve().parent;base=root.parent.parent
sys.path.insert(0,str(base));from host_runtime import PYTHON,environment,prepare
prepare();env=environment();env['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1];env['ASCEND_AICPU_PATH']='/usr/local/Ascend/cann-9.1.1'
env['PYTHONPATH']=str(root/'rows246-probe')+':'+str(root.parent/'pypto-serving-qwen3.8-27B')+':'+str(root/'src-dense-v248')+':'+str(root)+':'+env['PYTHONPATH'];env['PYPTO_CACHE_DIR']=str(root/'jit');env['PYPTO_CACHE']='1'
rc=subprocess.call([str(PYTHON),*sys.argv[2:]],cwd=root,env=env)
# 只将本次新建的结果和导出文件交还启动用户；保持现有权限和权重字节。
import os
for flag in ['--output','--export-heads']:
    if flag not in sys.argv:continue
    p=Path(sys.argv[sys.argv.index(flag)+1])
    if p.parent!=root or not p.name.startswith('a111-v248-'):raise RuntimeError('unexpected result path')
    if not p.exists():continue
    values=[p,*p.rglob('*')] if p.is_dir() else[p]
    for v in values:
        if v.is_symlink():raise RuntimeError('unexpected export symlink')
        os.chown(v,1015,1016)
raise SystemExit(rc)

