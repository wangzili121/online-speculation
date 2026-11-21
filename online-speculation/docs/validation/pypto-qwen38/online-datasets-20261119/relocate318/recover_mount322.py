from pathlib import Path
import json,copy,subprocess,zlib,gzip,datetime
import argparse
p=argparse.ArgumentParser();p.add_argument('--validation-root',type=Path,required=True);a=p.parse_args();D=a.validation_root;Q=D/'relocate318';old=json.loads((Q/'queue-spoc318B.json').read_text());plan=copy.deepcopy(old)
gate=plan['cases'][0];oldtag=gate['tag'];gate['tag']=oldtag+'-mountfix322';gate['expected_source_manifest']='spoc322-frozen-source.json'
formal=plan['cases'][1];formal['expected_source_manifest']='spoc322-frozen-source.json'
for s in formal['requires']:
 if s['file']==oldtag+'.json':s['file']=gate['tag']+'.json'
formal['requires']=[s for s in formal['requires']if s['file']!='spoc318B-activated.json']+[dict(file='spoc322-activated.json')]
remote=r'''from pathlib import Path
import json,subprocess,os,signal,time,hashlib,zlib,sys
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/online-studies');pid=2105411;proc=Path('/proc')/str(pid)
args=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert proc.stat().st_uid==os.getuid()and'study_queue_spoc312.py --plan queue-spoc318B.json'in args
assert not(r/'a111-v292-spoc312-confirm745.json').exists()and not(r/'a111-v292-spoc312-confirm745.log').exists()
os.kill(pid,signal.SIGSTOP)
try:
 time.sleep(.05);children=[s for s in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]if int(s.split(None,2)[1])==pid];assert not children,children
 proof=dict(pid=pid,uid=os.getuid(),command=args,children=children,only_waiting_activation=True,no_reserved_report_or_log=True)
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
except BaseException:
 os.kill(pid,signal.SIGCONT);raise
failed={n:(r/n).read_text()for n in [P['oldtag']+ext for ext in ['.json','.source.json','.log','.launch.log']]+['queue-spoc318B.status.json','queue-spoc318B.json.controller.log']}
launch=r/'run_dense_v292_docker.py';s=launch.read_text();needle='str(project/"cann-9.1.1")+":/usr/local/Ascend/cann-9.1.1:ro"';assert s.count(needle)==1
assert Path('/usr/local/Ascend/cann-9.1.1/lib64/libascendcl.so').is_file()
s=s.replace(needle,'"/usr/local/Ascend/cann-9.1.1:/usr/local/Ascend/cann-9.1.1:ro"');compile(s,str(launch),'exec');launch.write_text(s)
source=json.loads((r/'spoc318B-frozen-source.json').read_text())['files'];source['run_dense_v292_docker.py']=hashlib.sha256(s.encode()).hexdigest()
with(r/'spoc322-frozen-source.json').open('x')as f:json.dump(dict(files=source),f,indent=2)
with(r/'queue-spoc322B.json').open('x')as f:json.dump(P['plan'],f,indent=2)
with(r/'queue-spoc322B.json.controller.log').open('x')as log:p=subprocess.Popen(['python3','study_queue_spoc312.py','--plan','queue-spoc322B.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,source=source,base=json.loads((r/'spoc318B-base-files.json').read_text()),previous_failed_files=failed,stopped_waiting_controller=proof,NN_code_changed=False,host_CANN_mount_fixed=True,B_reserved_waits='spoc322-activated.json',A745_unchanged=True)))
'''
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@189.1.227.167','python3 -'],input=('P='+repr(dict(plan=plan,oldtag=oldtag))+'\n'+remote).encode(),capture_output=True,timeout=30);assert x.returncode==0,x.stderr.decode();out=json.loads(x.stdout)
failed=out.pop('previous_failed_files');fdir=Q/'failed-preNN318';fdir.mkdir(exist_ok=True)
for n,text in failed.items():
 b=text.encode();suffix='.gz'if n.endswith('.log')else'';(fdir/(n+suffix)).write_bytes(gzip.compress(b,mtime=0)if suffix else b)
(Q/'recovery322-stage.json').write_text(json.dumps(out,indent=2)+'\n');(Q/'queue-spoc322B.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(pid=out['pid'],gate=gate['tag'],A745_unchanged=True,formal_B_blocked=True)))
