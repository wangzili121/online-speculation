"""等待已派发实验的完整审核文件，再只读核对事件计划；不启动模型。"""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,time,traceback
from audit_schedule import audit
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();D=a.validation_root;Q=D/'schedule326'
specs=[dict(directory='spoc312',tag='a111-v292-spoc312-confirm'+str(seed),stride=64,observer=80840,observer_script='watch_spoc320.py')for seed in [743,744,745]]
specs += [dict(directory='explore321',tag=tag,stride=64,observer=76336,observer_script='watch_explore321.py')for tag in ['c61-v314-explore-longproc30-kl-staged64-v321-output512','c61-v314-explore-spoc64-kl-staged64-v321-output192']]
cache={}
while len(cache)<len(specs):
    pending=[]
    for spec in specs:
        tag=spec['tag']
        if tag in cache:continue
        folder=D/spec['directory'];proof=folder/(tag+'.audit.json')
        try:
            if not proof.exists():
                x=subprocess.run(['ps','-p',str(spec['observer']),'-o','command='],capture_output=True,text=True)
                live=x.returncode==0 and spec['observer_script']in x.stdout
                pending.append(dict(tag=tag,read_only_observer_live=live))
                continue
            prior=json.loads(proof.read_text())
            if not prior.get('completed'):
                cache[tag]=dict(completed=False,tag=tag,prior_audit_error=prior);continue
            rawpath=folder/(tag+'.json')
            raw=rawpath.read_bytes()if rawpath.exists()else gzip.decompress(rawpath.with_suffix('.json.gz').read_bytes())
            if hashlib.sha256(raw).hexdigest()!=prior['raw_sha256']:
                raise ValueError('raw report differs from primary audit')
            result=audit(json.loads(raw),spec['stride']);result.update(tag=tag,raw_sha256=prior['raw_sha256'],primary_audit_passed=True)
            cache[tag]=result;(Q/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        except (json.JSONDecodeError,FileNotFoundError):
            pending.append(dict(tag=tag,awaiting_atomic_file_completion=True))
        except Exception as e:
            cache[tag]=dict(completed=False,tag=tag,error=repr(e),traceback=traceback.format_exc())
    state=dict(completed=len(cache)==len(specs),all_audits_passed=len(cache)==len(specs)and all(v['completed']for v in cache.values()),cases=list(cache.values()),pending=pending,no_NN_dispatch=True)
    (a.artifacts/'schedule326-watcher-state.json').write_text(json.dumps(state,indent=2)+'\n')
    if len(cache)<len(specs):time.sleep(30)
(Q/'new-completed-reports-audit.json').write_text(json.dumps(state,indent=2)+'\n')
