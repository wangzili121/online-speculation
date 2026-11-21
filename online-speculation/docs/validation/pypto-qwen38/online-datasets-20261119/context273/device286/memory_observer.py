from pathlib import Path
import subprocess,json,time,datetime
q=Path(__file__).resolve().parent;deadline=time.monotonic()+3600
with(q/'memory-observer.jsonl').open('x')as out:
 while time.monotonic()<deadline:
  smi=subprocess.run(['/usr/local/bin/npu-smi','info'],capture_output=True,text=True,timeout=15);top=subprocess.run(['docker','top','orthrus-context286-single','-eo','pid,ppid,user,comm'],capture_output=True,text=True,timeout=10);out.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),npu_smi=smi.stdout,npu_smi_returncode=smi.returncode,own_container_processes=top.stdout,own_container_present=top.returncode==0))+'\n');out.flush()
  if(q/'exit.json').exists():break
  time.sleep(5)
