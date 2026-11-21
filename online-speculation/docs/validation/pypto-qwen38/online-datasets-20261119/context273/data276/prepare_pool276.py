from pathlib import Path
import subprocess,json,zlib,hashlib
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);a=parser.parse_args().artifacts;audit=json.loads((a/'pseudo274-v2-crosshost-exposure-audit.json').read_text());assert audit['completed'];used=sorted(set().union(*(set(h['used_question_sha256'])for h in audit['hosts'].values())))
code=r'''from pathlib import Path
import json,hashlib,yaml,sys,zlib
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'longproc-online-study-20261119';q=r/'pool276';q.mkdir();manifest=json.loads((r/'pseudo-study-manifest.json').read_text());source=r/'source.json';long=r/'source2k.json';assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256'];assert hashlib.sha256(long.read_bytes()).hexdigest()=='2e19e3503000f2d76c1da123319ceb8474d07edf5ed92386d34e0b562759053f';assert hashlib.sha256((r/'prompts.yaml').read_bytes()).hexdigest()==manifest['template_sha256'];template=yaml.safe_load((r/'prompts.yaml').read_text())['USER_PROMPT'];tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=True);seen=set(json.loads((r/'fresh266-eligibility-count.json').read_text())['registered_seen_problem_ids']);used=set(P['used']);rows=[];matched=set()
for file in [source,long]:
 for index,row in enumerate(json.loads(file.read_text())):
  problem=str(row['problem_id']);text=template.format(pseudocode='\n'.join(row['pseudocode_lines']));sha=hashlib.sha256(text.encode()).hexdigest();norm=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
  if sha in used or norm in used:matched.add(problem)
  rows.append(dict(problem_id=problem,text=text,question_sha256=sha,normalized_sha256=norm,source_file=file.name,source_index=index))
seen.update(matched);selected=[];ids=set();normalized=set()
for row in rows:
 if row['problem_id']in seen or row['problem_id']in ids or row['normalized_sha256']in normalized:continue
 tokens=tok.apply_chat_template([dict(role='user',content=row['text'])],tokenize=True,add_generation_prompt=True,enable_thinking=False);tokens=tokens.input_ids if hasattr(tokens,'input_ids')else tokens
 if len(tokens)+128>1024:continue
 row['prompt_tokens']=len(tokens);selected.append(row);ids.add(row['problem_id']);normalized.add(row['normalized_sha256'])
count=(len(selected)-32)//2*2;assert count>=24,len(selected);groups={'pseudo276-explore':selected[:count],'pseudo276-confirm':selected[count:count+24],'pseudo276-holdout':selected[count+24:count+32]};files={};sets={}
for name,values in groups.items():
 raw=json.dumps([x['text']for x in values],ensure_ascii=False)+'\n';(q/(name+'.json')).write_text(raw);files[name+'.json']=raw;sets[name]=dict(requests=len(values),sha256=hashlib.sha256(raw.encode()).hexdigest(),problem_ids=[x['problem_id']for x in values],rows=[{k:v for k,v in x.items()if k!='text'}for x in values])
result=dict(completed=True,context=1024,output_budget=128,source_commit='673ec4c230876e941d674116d66d28783c186b15',source_sha256=manifest['source_sha256'],source2k_sha256=hashlib.sha256(long.read_bytes()).hexdigest(),template_sha256=manifest['template_sha256'],crosshost_exposure_audit_sha256=P['audit_sha256'],matched_previous_text_problem_ids=sorted(matched),conservative_seen_problem_ids=sorted(seen),remaining_eligible=len(selected),unassigned_problem_ids=[x['problem_id']for x in selected[count+32:]],sets=sets,no_newNNrequests=True,performance_recipe_frozen=False,scope='Firsteligibleofficialvariant perproblem inoriginal0.5k then2k order. Excludeallregistered130IDs andallvariants matchedagainst3host launchedrequest/holdout/head128hash audit. Firstevenexploration then24reserve then8independentholdout; allpaircounts even. Noanswer/output/speed selection. Confirmation NN onlyafterfuturefullqualification at1024; noreserveduse now.')
(q/'selection-manifest.json').write_text(json.dumps(result,indent=2)+'\n');sys.stdout.buffer.write(zlib.compress(json.dumps(dict(manifest=result,assets=files)).encode()))
'''
# Tokenization is CPU only; no model loading, no worker and no device reservation.
payload_code='P='+repr(dict(used=used,audit_sha256=hashlib.sha256((a/'pseudo274-v2-crosshost-exposure-audit.json').read_bytes()).hexdigest()))+'\n'+code
outer="from pathlib import Path\nimport sys,subprocess\nb=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment\nx=subprocess.run([str(PYTHON),'-'],input="+repr(payload_code.encode())+",env=environment());raise SystemExit(x.returncode)\n"

x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@159.138.54.61','python3 -'],input=outer.encode(),capture_output=True,timeout=70);assert x.returncode==0,(x.stdout[-2000:],x.stderr[-3000:]);result=json.loads(zlib.decompress(x.stdout));(a/'pseudo276-selection-manifest.json').write_text(json.dumps(result['manifest'],indent=2)+'\n')
for n,t in result['assets'].items():(a/n).write_text(t)
print(json.dumps(dict(remaining=result['manifest']['remaining_eligible'],sets={k:v['requests']for k,v in result['manifest']['sets'].items()},unassigned=result['manifest']['unassigned_problem_ids'],noNN=True)))
