from pathlib import Path
import runpy,io,subprocess,threading,json
A=Path(__file__).resolve().parent;Guard=runpy.run_path(str(A/'guarded_entry_v28.py'))['Guard'];checks=[]
class Process:
 def __init__(self,failures):self.stdin=io.BytesIO();self.failures=failures;self.events=[];self.pid=87654;self.code=None
 def poll(self):return self.code
 def wait(self,timeout):
  self.events.append(('wait',timeout))
  if self.failures:self.failures-=1;raise subprocess.TimeoutExpired(['own-helper'],timeout)
  self.code=0;return 0
 def terminate(self):self.events.append(('terminate',self.pid))
 def kill(self):self.events.append(('kill',self.pid))
def make(failures,reply=None):
 g=object.__new__(Guard);g.process=Process(failures);g.lock=threading.Lock();g.closed=False;g.devices={0};g.released=set();g.read=lambda: reply or {'released':[0]};return g
for f,term,kill in [(0,False,False),(1,True,False),(2,True,True)]:
 g=make(f);g.close();before=list(g.process.events);g.close();assert g.process.events==before;assert g.closed and g.released=={0} and g.process.stdin.closed;assert any(x[0]=='terminate' for x in before)==term;assert any(x[0]=='kill' for x in before)==kill;checks.append(dict(wait_failures=f,owned_helper_reaped=True,idempotent=True))
g=make(1,{'released':[]})
try:g.close();raise AssertionError('missing release accepted')
except RuntimeError as e:assert 'acknowledge' in str(e)
assert g.process.stdin.closed and any(x[0]=='terminate' for x in g.process.events);checks.append(dict(incomplete_release_rejected=True,owned_helper_reaped=True))
(A/'guard28-cpu-validation.json').write_text(json.dumps(dict(completed=True,checks=checks),indent=2)+'\n');print('guard cleanup cases passed')
