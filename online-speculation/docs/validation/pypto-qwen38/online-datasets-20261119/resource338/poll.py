"""只读查看补充资源重复的真实控制器/launcher和报告；不启动或重启NN。"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess


REMOTE = r'''
from pathlib import Path
import json,subprocess,os
r=Path(P['root']);cmd=Path('/proc')/str(P['pid'])/'cmdline'
command=cmd.read_bytes().replace(bytes([0]),b' ').decode()if cmd.exists()else''
live=P['controller']in command and P['plan']in command
ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
def read(p):
 try:return json.loads(p.read_text())if p.exists()else{}
 except json.JSONDecodeError:return{'observation_incomplete':True}
state=read(r/Path(P['plan']).with_suffix('.status.json'))
rows=[]
for c in P['cases']:
 d=read(r/(c['tag']+'.json'));s=next((v for v in state.get('cases',[])if v['tag']==c['tag']),{})
 launchers=[int(line.split(None,1)[0])for line in ps if c['launcher']+' --tag '+c['tag']+' 'in line]
 rows.append(dict(tag=c['tag'],launcher_pids=launchers,stage=d.get('stage'),completed=s.get('completed',False),error=s.get('error')or d.get('error'),observation_incomplete=d.get('observation_incomplete',False),progress={k:(v[-1]if isinstance(v,list)and v else v)for k,v in d.get('progress',{}).items()},results={k:v['stream']['tok_s']for k,v in d.get('results',{}).items()if'stream'in v}))
print(json.dumps(dict(root=str(r),controller_pid=P['pid'],controller_live=live,cases=rows)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    Q = args.validation_root / "resource338"
    jobs = []
    for host, address in (("A", "159.138.5.111"), ("B", "189.1.227.167")):
        stage = json.loads((Q / f"stage-{host}.json").read_text())
        for item in stage["launched"]:
            name = "A" if host == "A" else "B_gsm" if item["root"].endswith("online-background") else "B_spoc"
            jobs.append((address, item | dict(cases=json.loads((Q / f"queue-{name}.json").read_text())["cases"])))

    def one(job):
        host, payload = job
        try:
            value = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                                    "-o", "ConnectTimeout=10", "wangzili@" + host, "python3 -"],
                                   input=("P=" + repr(payload) + "\n" + REMOTE).encode(), capture_output=True, timeout=40)
            return dict(host=host, returncode=value.returncode, observation=json.loads(value.stdout)
                        if value.returncode == 0 else value.stderr.decode()[-2000:])
        except subprocess.TimeoutExpired:
            return dict(host=host, observation_timeout=True, terminal=False)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for value in pool.map(one, jobs):
            print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
