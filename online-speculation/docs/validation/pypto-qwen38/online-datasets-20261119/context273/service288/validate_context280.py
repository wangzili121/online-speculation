from pathlib import Path
import time,json,traceback
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');q=b/'context-v280';r=b/'oneserving-v109-b99e118/online-studies';proof=dict(completed=False,scope='Actual1NPU componentgates before2card model. No performance threshold orreservedquestions.')
try:
 while not(q/'exit.json').exists():time.sleep(10)
 exit=json.loads((q/'exit.json').read_text());status=json.loads((q/'status.json').read_text());assert exit['completed']and status['completed'],(exit,status)
 for page in [16,128]:
  d=json.loads((q/f'attention-page{page}.json').read_text());assert d['passed']and d['npu_execution']and len(d['cases'])==19
  assert all(c['passed']and c['inputs_byte_exact']and c['poisoned_inactive_inputs']for c in d['cases']);assert max(c['past']for c in d['cases'])==1022
 d=json.loads((q/'training1024.json').read_text());assert d['completed']and d['npu_execution']and d['inputs_byte_exact']and len(d['cases'])==6
 assert all(c['active_logits_bitwise']and all(g['bitwise']for g in c['gradients'].values())for c in d['cases']);assert max(c['past']for c in d['cases'])==1008
 proof.update(completed=True,poisoned_paged_readonly_cases=38,realweight_forward_backward_cases=6,active_logits_and_gradients_bitwise=True)
except BaseException as e:proof.update(error=repr(e),traceback=traceback.format_exc())
(r/'context280-components-validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
