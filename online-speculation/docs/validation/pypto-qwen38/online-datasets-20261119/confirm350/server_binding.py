"""在服务器本地保存运行时设备绑定；只读观察，不调用模型或启动器。"""
from datetime import datetime, timezone
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from common import module, read, sha


BODY = r'''
from pathlib import Path
import contextlib,datetime,hashlib,io,json,os,subprocess,tempfile,time
os.umask(0o077)
root=Path(P['root']);folder=root/'resource-observations-confirm350';folder.mkdir(exist_ok=True)
digest=hashlib.sha256(LIVE_SOURCE.encode()).hexdigest();captured=set();deadline=time.monotonic()+36*3600
def save(path,value):
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False)as f:
  json.dump(value,f,indent=2);f.write('\n');name=f.name
 os.replace(name,path)
def controller_alive():
 proc=Path('/proc')/str(P['controller_pid'])
 try:
  ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
  return ticks==P['controller_start_ticks']and P['plan']in(proc/'cmdline').read_bytes().decode()
 except FileNotFoundError:return False
while True:
 observations=[]
 for case in P['cases']:
  tag=case['tag'];path=folder/(tag+'.resources.json')
  if path.exists():
   previous=json.loads(path.read_text())
   assert previous['capture_program_sha256']==digest and previous['tag']==tag and previous['captured']
   captured.add(tag);continue
  if tag in captured:continue
  try:
   output=io.StringIO()
   with contextlib.redirect_stdout(output):exec(LIVE_SOURCE,{'P':dict(root=str(root),host=P['host'],**case)})
   snap=json.loads(output.getvalue());observations.append({k:snap.get(k)for k in('tag','stage','completed','captured','error')})
   if snap['captured']:
    snap.update(capture_program_sha256=digest,server_local_observer_pid=os.getpid(),read_only=True)
    save(path,snap);captured.add(tag)
    print(json.dumps(dict(tag=tag,captured=True,physical_cards=snap['physical_cards'])),flush=True)
  except Exception as error:observations.append(dict(tag=tag,observation_error=repr(error),NN_restarted=False))
 live=controller_alive();done=len(captured)==len(P['cases']);terminal=done or not live or time.monotonic()>=deadline
 save(folder/'observer-state.json',dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      all_bindings_captured=done,terminal=terminal,captured=sorted(captured),controller_live=live,
      observations=observations,read_only=True,NN_started_or_restarted=False,goal_complete=False))
 if terminal:break
 time.sleep(30)
'''


DEPLOY = r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
os.umask(0o077)
directory=Path(P['root']).parent/'observations-confirm350';directory.mkdir(exist_ok=True)
program=directory/'capture.py';marker=directory/'launch.json';digest=hashlib.sha256(P['program'].encode()).hexdigest()
if marker.exists():
 result=json.loads(marker.read_text());assert result['script_sha256']==digest
else:
 if program.exists():assert program.read_text()==P['program']
 else:program.write_text(P['program'])
 with(directory/'capture.log').open('xb')as handle:
  proc=subprocess.Popen([sys.executable,str(program)],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=int((Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
 result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=proc.pid,start_ticks=ticks,
             script=str(program),script_sha256=digest,root=P['root'],read_only=True,NN_started_or_restarted=False)
 with marker.open('x')as handle:json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
'''


def context():
    q = Path(__file__).resolve().parent
    capture = module(q.parent / "resource338/capture_resources.py", "server_capture350")
    launch = read(q / "launch-B.json")
    recovery = read(q / "access-recovery-20261122T0817.json")
    payload = dict(root=launch["root"], host="189.1.227.167", controller_pid=launch["pid"],
                   controller_start_ticks=recovery["controller_start_ticks"], plan=launch["plan"],
                   cases=[dict(tag=c["tag"], launcher=c["launcher"]) for c in read(q / "queue-B.json")["cases"]])
    program = "P=" + repr(payload) + "\nLIVE_SOURCE=" + repr(capture.LIVE_REMOTE) + "\n" + BODY
    compile(program, "server-observer", "exec")
    return q, capture, payload, program


def selfcheck():
    q, capture, payload, program = context()
    binding = read(q / (payload["cases"][0]["tag"] + ".resources.json"))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folder = root / "resource-observations-confirm350"
        folder.mkdir()
        binding = copy.deepcopy(binding)
        binding.update(root=str(root), capture_program_sha256=sha(capture.LIVE_REMOTE.encode()))
        path = folder / (binding["tag"] + ".resources.json")
        request = dict(root=str(root), tag=binding["tag"], host=binding["host"])
        def run(value):
            path.write_text(json.dumps(value))
            output = io.StringIO()
            with patch.object(subprocess, "check_output", side_effect=AssertionError("live probe attempted")), \
                 contextlib.redirect_stdout(output):
                exec(capture.REMOTE, {"P": request})
            return json.loads(output.getvalue())
        assert run(binding) == binding
        for replacement in (dict(capture_program_sha256="wrong"), dict(tag="wrong-tag")):
            try:
                run(binding | replacement)
            except AssertionError:
                pass
            else:
                raise AssertionError("Invalid persisted binding accepted")
    return dict(passed=True, preserved_live_capture_sha256=sha(capture.LIVE_REMOTE.encode()),
                persisted_snapshot_loaded_without_live_NPU_probe=True, wrong_source_or_tag_rejected=True,
                compiled_server_observer=True, NPU_or_model_calls=False,
                scope="CPU persistence/consumer checks only; server capture must still be observed live.")


def deploy():
    q, capture, payload, program = context()
    transport = module(q.parent / "resource338/ssh_connection.py", "server_transport350")
    call = subprocess.run(
        transport.command("189.1.227.167"),
        input=("P=" + repr(dict(root=payload["root"], program=program)) + "\n" + DEPLOY).encode(),
        capture_output=True, timeout=40, check=True,
    )
    result = json.loads(call.stdout)
    assert result["script_sha256"] == sha(program.encode())
    result.update(capture_program_sha256=sha(capture.LIVE_REMOTE.encode()),
                  recorded_locally_UTC=datetime.now(timezone.utc).isoformat(),
                  observation_only=True, model_source_or_frozen_recipe_changed=False)
    (q / "server-binding-observer.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--deploy"]:
        print(json.dumps(deploy()))
    else:
        q = Path(__file__).resolve().parent
        result = selfcheck()
        (q / "server-binding-selfcheck.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
