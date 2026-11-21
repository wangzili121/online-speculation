"""官方2K子集的长度/问题ID盘点；不生成、不训练、不分配新的确认题。"""
from pathlib import Path
import hashlib,json,yaml,urllib.request
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'longproc-online-study-20261119';commit='673ec4c230876e941d674116d66d28783c186b15';url=f'https://raw.githubusercontent.com/princeton-pli/LongProc/{commit}/data/pseudo_to_code/pseudo_to_code_2k.json';raw=urllib.request.urlopen(url,timeout=45).read();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest();assert blob=='d56b9d94296970bcee8c78cfc5210467eff0095f';p=r/'source2k.json'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
old=json.loads((r/'fresh266-eligibility-count.json').read_text());seen=set(old['registered_seen_problem_ids']);short=json.loads((r/'source.json').read_text());long=json.loads(raw);template=yaml.safe_load((r/'prompts.yaml').read_text())['USER_PROMPT'];tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=True);eligible={(c,o):{}for c in [512,1024,2048,4096]for o in [128,256,512]}
for row in short+long:
 problem=str(row['problem_id']);text=template.format(pseudocode='\n'.join(row['pseudocode_lines']));ids=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False);ids=ids.input_ids if hasattr(ids,'input_ids')else ids
 for (context,output),values in eligible.items():
  if len(ids)+output<=context:values.setdefault(problem,len(ids))
report=dict(completed=True,source_url=url,git_blob_sha1=blob,sha256=hashlib.sha256(raw).hexdigest(),official_bin='2k refers to the original maximum output length; not a context size',records2k=len(long),problems2k=len({str(x['problem_id'])for x in long}),union_problems=len({str(x['problem_id'])for x in short+long}),registered_seen=len(seen),counts=[dict(context=c,output_cap=o,eligible_problems=len(v),remaining_after_registered_ID_exclusion=len(set(v)-seen))for(c,o),v in eligible.items()],scope='Counts only. First eligible officialvariant perproblem; no answers/generation/speed selection. No GPU use ornewreservedassignment. Fullcrosshost request/holdout exposure audit still required beforeclaiming unused cohort.')
(r/'fresh268-official2k-lengths.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
