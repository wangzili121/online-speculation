from pathlib import Path
import time,json,subprocess,hashlib
r=Path(__file__).parent;p=r/'rows238-components-device.json'
while True:
 if p.exists():break
 log=r/'rows238-components-controller.log'
 if log.exists()and '"returncode":'in log.read_text():
  p.write_text(json.dumps(dict(completed=False,error='Probe process exited without result; inspect rows238-components-device.log'))+chr(10));break
 time.sleep(3)
d=json.loads(p.read_text());d.update(gate_for='rows235',retry_of='rows234-components-compile-failure.json',probe_sha256=hashlib.sha256((r/'rows238-probe/validate_qwen38_pypto_row_layout.py').read_bytes()).hexdigest());(r/'rows234-components-device.json').write_text(json.dumps(d,indent=2)+chr(10));print(json.dumps(d),flush=True)
