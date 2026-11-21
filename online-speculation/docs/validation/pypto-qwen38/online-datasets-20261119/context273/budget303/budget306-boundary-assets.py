from pathlib import Path
import json,hashlib
from transformers import AutoTokenizer
r=Path.cwd();b=r.parent.parent;tok=AutoTokenizer.from_pretrained('/data/disk/models/Qwen3.8-27B',local_files_only=True,trust_remote_code=False);old=json.loads((r/'assets/context284-synthetic6.json').read_text());questions=[];lengths=[]
def size(text):
 value=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False);return len(value['input_ids']if hasattr(value,'keys')else value)
for original,target in zip(old[:2],[992,1000]):
 text=original+'\n';n=max(0,target-size(text));text+=' x'*n
 while size(text)<target:text+=' x'
 while size(text)>target:text=text[:-2]
 assert size(text)==target;questions.append(text);lengths.append(target)
path=r/'assets/budget303-boundary2.json';text=json.dumps(questions,ensure_ascii=False)+chr(10)
with path.open('x')as f:f.write(text)
proof=dict(completed=True,npu_execution=False,questions_from='first2ofoldsynth6',context=1024,prompt_lengths=lengths,requested=512,output_budgets=[1024-x for x in lengths],sha256=hashlib.sha256(text.encode()).hexdigest(),scope='Only functional contextboundary diagnostic, ignoreEOS inallarms, no new dataset or performance claim.')
(r/'budget303-boundary-assets.json').write_text(json.dumps(proof,indent=2)+chr(10))
