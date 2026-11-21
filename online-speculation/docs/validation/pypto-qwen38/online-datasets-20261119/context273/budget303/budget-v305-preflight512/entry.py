P={'old_lengths': [453, 707, 529, 847, 467, 582, 531, 598, 560, 478, 538, 458, 491, 610, 565, 483, 597, 504, 510, 676, 614, 501, 533, 535, 455, 562, 533, 533, 487, 531], 'old_sha': '4e218ceb98cdb1ee9c8c786f3b9e1120aa6be65f234e688e0f05a095f8662db5'}
from pathlib import Path
import importlib.util,json,hashlib
from transformers import AutoTokenizer
q=Path.cwd();b=q.parent;sp=importlib.util.spec_from_file_location('bench',b/'budget-v303-cpu/scripts/benchmark_qwen38_oneserving_tp2_online.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=False);cases=[]
for name,path,oldlimit in [('LongProc30',b/'longproc-online-study-20261119/pool276/pseudo276-explore.json',128),('SPoC64',b/'spoc-online-study-20261121/pool283/spoc283-explore64.json',192)]:
 raw=path.read_bytes();qs=json.loads(raw);budgets,lengths=m.chat_output_budgets(tok,qs,512,1024);assert all(lengths[x]+budgets[x]<=1024 and budgets[x]>=oldlimit for x in qs)
 if name=='LongProc30':assert hashlib.sha256(raw).hexdigest()==P['old_sha']and [lengths[x]for x in qs]==P['old_lengths']
 cases.append(dict(dataset=name,sha256=hashlib.sha256(raw).hexdigest(),requests=len(qs),old_budget=oldlimit,prompt_tokens=[lengths[x]for x in qs],output_budgets=[budgets[x]for x in qs],nominal_total_output=sum(budgets.values()),all_questions_preserved=True))
(q/'budget-preflight.json').write_text(json.dumps(dict(completed=True,npu_execution=False,context=1024,requested=512,cases=cases,source='Exact native apply_chat_template, no answers or model execution, same budget per question in all three arms.'),indent=2)+chr(10))
