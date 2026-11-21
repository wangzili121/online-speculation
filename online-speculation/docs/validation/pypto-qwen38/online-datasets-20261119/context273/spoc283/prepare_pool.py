from pathlib import Path
import subprocess,json,zlib
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);a=parser.parse_args().artifacts
script=r'''from pathlib import Path
import io,csv,zipfile,hashlib,json,yaml,sys,zlib
from collections import OrderedDict
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'spoc-online-study-20261121';longproc=b/'longproc-online-study-20261119';source=r/'spoc.zip';assert hashlib.sha256(source.read_bytes()).hexdigest()=='9a4439f7c61ee948a6522f93bcfa5495ac6cdfa6ffcd9b2807a093ed7130e7d1';oldids={str(x['problem_id'])for file in ['source.json','source2k.json']for x in json.loads((longproc/file).read_text())};used=set(json.loads((r/'used-question-hashes282.json').read_text()));raw_template=(longproc/'prompts.yaml').read_bytes();assert hashlib.sha256(raw_template).hexdigest()=='d0739b6becb28c16fad99291f9b32e15bdba7aaf5ef21f789d414ee947c22b16';template=yaml.safe_load(raw_template)['USER_PROMPT'];tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),trust_remote_code=True,local_files_only=True)
with zipfile.ZipFile(source)as z:data=z.read('train/spoc-train.tsv')
assert hashlib.sha256(data).hexdigest()=='925cc1badb32282184842cc6f8b9d34adf46c802fb2f2088728205d3a129b460';groups=OrderedDict()
for row in csv.DictReader(io.StringIO(data.decode()),delimiter='\t',quoting=csv.QUOTE_NONE):
 assert None not in row;groups.setdefault((row['probid'],row['subid'],row['workerid']),[]).append((int(row['line']),int(row['indent']),row['text']))
selected=[];ids=set(oldids);normalized=set()
for (problem,submission,worker),rows in groups.items():
 if problem in ids:continue
 rows=sorted(rows);assert len({x[0]for x in rows})==len(rows);lines=['    '*indent+text for _,indent,text in rows if text.strip()]
 if not lines:continue
 text=template.format(pseudocode='\n'.join(lines));sha=hashlib.sha256(text.encode()).hexdigest();norm=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
 if sha in used or norm in used or norm in normalized:continue
 tokens=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False);tokens=tokens.input_ids if hasattr(tokens,'input_ids')else tokens
 if len(tokens)+192>1024:continue
 ids.add(problem);normalized.add(norm);selected.append(dict(problem_id=problem,submission_id=submission,worker_id=worker,prompt_tokens=len(tokens),question_sha256=sha,normalized_sha256=norm,text=text))
assert len(selected)>=352,len(selected);q=r/'pool283';q.mkdir();files={};sets={}
for name,values in [('spoc283-explore64',selected[:64]),('spoc283-confirm256',selected[64:320]),('spoc283-holdout32',selected[320:352])]:
 raw=json.dumps([x['text']for x in values],ensure_ascii=False)+'\n';(q/(name+'.json')).write_text(raw);files[name+'.json']=raw;sets[name]=dict(requests=len(values),sha256=hashlib.sha256(raw.encode()).hexdigest(),rows=[{k:v for k,v in x.items()if k!='text'}for x in values])
manifest=dict(completed=True,source_url='https://sumith1896.github.io/spoc/data/spoc.zip',source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),member='train/spoc-train.tsv',member_sha256=hashlib.sha256(data).hexdigest(),task='SPoC-derived Pseudocode-to-C++ supplementary stream; NOT the original fixedLongProc benchmark',selection='Firsteligibleoriginalprogramannotation perproblem in officialsource order. Onlylength, ID/textde-duplication andprovenancefilters, noanswer/output/speed selection.',template_sha256=hashlib.sha256(raw_template).hexdigest(),ignored_answer_field='code',excluded_all_LongProc_problem_ids=sorted(oldids),eligible_unused_problems=len(selected),max_model_len=1024,output_budget=192,sets=sets,unassigned_problem_ids=[x['problem_id']for x in selected[352:]],noNNrequests=True,performance_recipe_frozen=False,scope='64explore/256newconfirm/32independentholdout, mutuallyproblemID-disjoint. No generation, training or qualification yet. Three confirmations onlyaftera future fullsame-resource exploratory qualification andfrozenrecipe; no sourcecodeaccuracy claim.')
(q/'selection-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');sys.stdout.buffer.write(zlib.compress(json.dumps(dict(manifest=manifest,assets=files)).encode()))
'''
outer="from pathlib import Path\nimport sys,subprocess\nb=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');sys.path.insert(0,str(b));from host_runtime import PYTHON,environment\nx=subprocess.run([str(PYTHON),'-'],input="+repr(script.encode())+",env=environment());raise SystemExit(x.returncode)\n"
x=subprocess.run(['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10','wangzili@159.138.54.61','python3 -'],input=outer.encode(),capture_output=True,timeout=100);assert x.returncode==0,(x.stdout[-1000:],x.stderr[-3000:]);d=json.loads(zlib.decompress(x.stdout));(a/'spoc283-selection-manifest.json').write_text(json.dumps(d['manifest'],indent=2)+'\n')
for n,text in d['assets'].items():(a/n).write_text(text)
print(json.dumps(dict(completed=True,eligible=d['manifest']['eligible_unused_problems'],sets={k:v['requests']for k,v in d['manifest']['sets'].items()},noNN=True)))
