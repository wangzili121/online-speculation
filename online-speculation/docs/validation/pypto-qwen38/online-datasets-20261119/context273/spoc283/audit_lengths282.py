from pathlib import Path
import csv,io,zipfile,hashlib,json,yaml
from collections import OrderedDict
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'spoc-online-study-20261121';longproc=b/'longproc-online-study-20261119';source=r/'spoc.zip';assert hashlib.sha256(source.read_bytes()).hexdigest()=='9a4439f7c61ee948a6522f93bcfa5495ac6cdfa6ffcd9b2807a093ed7130e7d1';used=set(json.loads((r/'used-question-hashes282.json').read_text()));oldids={str(x['problem_id'])for file in ['source.json','source2k.json']for x in json.loads((longproc/file).read_text())};template=yaml.safe_load((longproc/'prompts.yaml').read_text())['USER_PROMPT'];tokenizer=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),trust_remote_code=True,local_files_only=True);reports=[]
with zipfile.ZipFile(source)as archive:
 for name in ['train/spoc-train.tsv','test/spoc-testp.tsv','test/spoc-testw.tsv']:
  groups=OrderedDict();data=archive.read(name);reader=csv.DictReader(io.StringIO(data.decode()),delimiter='\t',quoting=csv.QUOTE_NONE)
  for row in reader:
   assert None not in row,(name,'malformedTSV');key=(row['probid'],row['subid'],row['workerid']);groups.setdefault(key,[]).append((int(row['line']),int(row['indent']),row['text']))
  eligible={(c,o):dict(programs=0,problems=set())for c in [512,1024]for o in [128,192]};all_problem_ids=set();matched=set()
  for (problem,submission,worker),rows in groups.items():
   all_problem_ids.add(problem)
   if problem in oldids:continue
   rows=sorted(rows);assert len({x[0]for x in rows})==len(rows);lines=['    '*indent+text for _,indent,text in rows if text.strip()]
   if not lines:continue
   text=template.format(pseudocode='\n'.join(lines));sha=hashlib.sha256(text.encode()).hexdigest();normalized=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
   if sha in used or normalized in used:matched.add(problem);continue
   tokens=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False);tokens=tokens.input_ids if hasattr(tokens,'input_ids')else tokens
   for (context,output),values in eligible.items():
    if len(tokens)+output<=context:values['programs']+=1;values['problems'].add(problem)
  reports.append(dict(member=name,member_sha256=hashlib.sha256(data).hexdigest(),all_annotated_programs=len(groups),all_problem_ids=len(all_problem_ids),excluded_all_LongProc_problem_ids=len(oldids&all_problem_ids),additional_rendered_text_matched_problem_ids=sorted(matched),eligibility=[dict(context=c,output_budget=o,eligible_annotated_programs=v['programs'],eligible_unique_problems=len(v['problems']))for(c,o),v in eligible.items()]))
report=dict(completed=True,source_manifest=json.loads((r/'source-manifest.json').read_text()),ignored_answer_field='code',rendering='Onlyhumantext andprovidedindent, emptytextlines skipped. No reference code copied into prompt. LongProc tasktemplate retained; thisis a separateSPoC-derived candidate stream, not unchangedLongProc.',excluded_all_LongProc_problem_ids=len(oldids),sets=reports,scope='CPU eligibility counts only. No NPU, no generatedanswers, no recipe selection ornewreservedNNassignment; final future split must beproblemID disjoint acrossexploration/confirmation/holdout.')
(r/'lengths282.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
