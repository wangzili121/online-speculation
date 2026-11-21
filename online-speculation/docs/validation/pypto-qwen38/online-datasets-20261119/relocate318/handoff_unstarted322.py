"""仅迁移从未启动模型的等待任务；实机诊断和不可变配置检查先行。"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();D=a.validation_root.resolve();Q=D/'relocate318';A=a.artifacts.resolve()
assert not(Q/'placement.json').exists(),'Already handed off; do not repeat'
proof=json.loads((Q/'boundary-audit.json').read_text());assert proof['completed']and proof['source_base_head_runtime_checked']and proof['tag']=='a111-v292-spoc318B-boundary-mountfix322-metadata323'
for name in ['model-runtime-equivalence.json','extended-runtime-equivalence.json']:assert json.loads((Q/name).read_text())['passed']
cann=json.loads((Q/'cann-library-equivalence.json').read_text());assert set(cann['differing'])=={'mstx.so','libmspti.so','libms_tools_ext.so'}
original=json.loads((D/'spoc312/queue-spoc312-confirm745.json').read_text())['cases'][0]
recovery=json.loads((Q/'queue-spoc323B.json').read_text());formal=recovery['cases'][1];stage=json.loads((Q/'recovery323-stage.json').read_text())
def numerical_args(args):
 out=[];i=0
 while i<len(args):
  if args[i]in ['--devices','--wait']:i+=2;continue
  out.append(args[i]);i+=1
 return out
assert original['tag']==formal['tag']and numerical_args(original['args'])==numerical_args(formal['args'])
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
# 暂停原只读观察器，避免把已证明的主机迁移误记为试验缺失。
observer=54877;cmd=subprocess.check_output(['ps','-p',str(observer),'-o','command='],text=True).strip();assert str(D/'spoc312/watch_spoc312.py')in cmd
os.kill(observer,signal.SIGSTOP);stopped=False
remote=r'''from pathlib import Path
import json,os,signal,subprocess,time,datetime
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');cp,lp=1820622,2397650;tag=P['tag'];record=r/'spoc312-confirm745-relocated-unstarted322B.json'
assert not record.exists()
def args(pid):return(Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(bytes([0])).decode().split(chr(0))
def check_process(pid,expected):
 p=Path('/proc')/str(pid);assert p.stat().st_uid==os.getuid()==1015;assert args(pid)==expected,(pid,args(pid),expected);assert(p/'cwd').resolve()==r
check_process(cp,['python3','study_queue_spoc312.py','--plan','queue-spoc312-confirm745.json'])
check_process(lp,['python3','run_dense_v292_docker.py','--tag',tag,*P['args']])
def check_no_model():
 assert not(r/(tag+'.json')).exists()and not(r/(tag+'.log')).exists(),'A model already started; do not migrate'
 rows=[x.split(None,4)for x in subprocess.check_output(['ps','-eo','pid,ppid,uid,stat,args'],text=True).splitlines()[1:]];procs={int(x[0]):dict(ppid=int(x[1]),uid=int(x[2]),state=x[3],args=x[4])for x in rows};tree={cp,lp}
 while True:
  new={pid for pid,v in procs.items()if v['ppid']in tree}-tree
  if not new:break
  tree.update(new)
 for pid in tree-{cp,lp}:
  v=procs[pid];query=v['args']=='/usr/local/bin/npu-smi info';reaped_query=v['state'].startswith('Z') and v['args']=='[npu-smi] <defunct>';assert v['uid']==1015 and v['ppid']==lp and(query or reaped_query),v
 assert procs[lp]['ppid']==cp
 return{str(k):procs[k]for k in tree if k in procs}
before=check_no_model();paused=[]
try:
 for pid in [cp,lp]:os.kill(pid,signal.SIGSTOP);paused.append(pid)
 time.sleep(.05);check_process(cp,['python3','study_queue_spoc312.py','--plan','queue-spoc312-confirm745.json']);check_process(lp,['python3','run_dense_v292_docker.py','--tag',tag,*P['args']]);after=check_no_model()
 for pid in [cp,lp]:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 time.sleep(.1)
 for pid in [cp,lp]:
  f=Path('/proc')/str(pid)/'stat';assert not f.exists()or f.read_text().split(') ',1)[1].split()[0]=='Z',pid
 proof=dict(completed=True,tag=tag,no_reserved_NN_started_on_A=True,original_A_waiting_only_stopped=True,controller_pid=cp,launcher_pid=lp,uid=1015,before=before,after_pausing=after,stopped_only_own_waiters=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with record.open('x')as f:json.dump(proof,f,indent=2)
 print(json.dumps(proof))
except BaseException:
 for pid in paused:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 raise
'''
try:
 x=subprocess.run([*ssh,'wangzili@159.138.5.111','python3 -'],input=('P='+repr(dict(tag=original['tag'],args=original['args']))+'\n'+remote).encode(),capture_output=True,timeout=30)
 assert x.returncode==0,x.stderr.decode();stopproof=json.loads(x.stdout);stopped=True;(Q/'A-stopped-unstarted322.json').write_text(json.dumps(stopproof,indent=2)+'\n')
except BaseException:
 os.kill(observer,signal.SIGCONT);raise
# 先持久化交接证明；B 在拿到激活文件前仍不能执行预留题。
placement=dict(completed=True,tag=original['tag'],seed=745,execution_host='189.1.227.167',original_A_waiting_only_stopped=True,no_reserved_NN_started_on_A=True,stage_manifest='recovery323-stage.json',controller_status='queue-spoc323B.status.json',controller_log='queue-spoc323B.json.controller.log',controller_pid=stage['pid'],same_numeric_args=True,allowed_operational_changes=['host','physical idle-card candidates','wait duration','Docker same digest pin','result ownership','host CANN mount source'],CANN_library_differences=cann['differing'],CANN_full_tree_byte_exact=False,model_PyPTO_PTOAS_runtime_identity_proofs=['model-runtime-equivalence.json','extended-runtime-equivalence.json'],boundary_raw_sha256=proof['raw_sha256'],A_stop_sha256=hashlib.sha256((Q/'A-stopped-unstarted322.json').read_bytes()).hexdigest(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
(Q/'placement.json').write_text(json.dumps(placement,indent=2)+'\n')
activate=r'''from pathlib import Path
import json,hashlib,os
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');tag=P['tag'];assert not(r/(tag+'.json')).exists()and not(r/(tag+'.log')).exists()
raw=r/'a111-v292-spoc318B-boundary-mountfix322-metadata323.json';d=json.loads(raw.read_text());assert d['completed']and d['clean_shutdown']and hashlib.sha256(raw.read_bytes()).hexdigest()==P['boundary_raw_sha256']
proc=Path('/proc')/str(P['controller_pid']);assert proc.stat().st_uid==os.getuid();assert'study_queue_spoc312.py --plan queue-spoc323B.json'in(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
with(r/'spoc322-activated.json').open('x')as f:json.dump(P,f,indent=2)
print(json.dumps(dict(activated=True,tag=tag,controller_pid=P['controller_pid'])))
'''
x=subprocess.run([*ssh,'wangzili@189.1.227.167','python3 -'],input=('P='+repr(placement)+'\n'+activate).encode(),capture_output=True,timeout=30)
(Q/'activation-result.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout.decode(),stderr=x.stderr.decode()),indent=2)+'\n')
# 旧观察器已暂停且身份已核对；只结束这一只读进程，替换为按主机收集。
os.kill(observer,signal.SIGTERM);os.kill(observer,signal.SIGCONT)
with(A/'spoc312-mixed320-watcher.log').open('xb')as log:
 w=subprocess.Popen(['python3',str(Q/'watch_spoc320.py'),'--validation-root',str(D),'--artifacts',str(A)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(Q/'observer-launch.json').write_text(json.dumps(dict(pid=w.pid,replaced_pid=observer,command=cmd),indent=2)+'\n')
assert x.returncode==0,x.stderr.decode()
print(json.dumps(dict(placement=placement,observer_pid=w.pid)))
