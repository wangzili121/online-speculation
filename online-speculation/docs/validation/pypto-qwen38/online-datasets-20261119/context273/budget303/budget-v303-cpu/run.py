from pathlib import Path
import subprocess,json,sys,traceback,hashlib
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');q=b/'budget-v303-cpu';s=b/'oneserving-v109-b99e118';sys.path.insert(0,str(b));from host_runtime import PYTHON,environment
e=environment();e['PYTHONPATH']=':'.join([str(s/'online-tp2-test/src-dense-v291'),str(s/'pypto-serving-qwen3.8-27B'),e['PYTHONPATH']]);e['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';status=dict(completed=False,npu_execution=False)
try:
 rc=subprocess.run([str(PYTHON),'-m','pytest','-q',str(q/'tests'),'-o','addopts='],cwd=q,env=e,timeout=1200).returncode;status['tests_returncode']=rc;assert rc==0
 code="""from pathlib import Path
import json,sys,importlib.util,hashlib
from transformers import AutoTokenizer
q=Path.cwd();b=q.parent;s=b/'oneserving-v109-b99e118';r=b/'longproc-online-study-20261119/pool276';tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=False);sp=importlib.util.spec_from_file_location('bench',q/'scripts/benchmark_qwen38_oneserving_tp2_online.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);cases=[]
for name,p,oldbudget in [('LongProc30',r/'pseudo276-explore.json',128),('SPoC64',b/'spoc-online-study-20261121/spoc283-explore64.json',192)]:
 text=p.read_bytes();questions=json.loads(text);budgets,lengths=m.chat_output_budgets(tok,questions,256,1024);assert all(budgets[x]>=oldbudget and lengths[x]+budgets[x]<=1024 for x in questions);cases.append(dict(dataset=name,requests=len(questions),sha256=hashlib.sha256(text).hexdigest(),old_budget=oldbudget,maximum_requested=256,context=1024,prompt_lengths=[lengths[x]for x in questions],output_budgets=[budgets[x]for x in questions],nominal_total_output=sum(budgets.values()),same_questions_kept=True))
(q/'budget-preflight.json').write_text(json.dumps(dict(completed=True,npu_execution=False,cases=cases,source_of_lengths='Exact native engine apply_chat_template: tokenize=True/add_generation_prompt=True/enable_thinking=False; no answers, outputs or speed used.'),indent=2)+'\\n')
"""
 rc=subprocess.run([str(PYTHON),'-c',code],cwd=q,env=e,timeout=120).returncode;status['real_tokenizer_returncode']=rc;assert rc==0;status['completed']=True
except BaseException as error:status.update(error=repr(error),traceback=traceback.format_exc())
finally:(q/'status.json').write_text(json.dumps(status,indent=2)+'\n')
