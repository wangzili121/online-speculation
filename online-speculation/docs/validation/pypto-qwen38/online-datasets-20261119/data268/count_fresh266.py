from pathlib import Path
import json,hashlib,sys,yaml
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');r=b/'longproc-online-study-20261119';assets=b/'oneserving-v109-b99e118/online-tp2-test/assets';old=json.loads((r/'pseudo-study-manifest.json').read_text());source=r/'source.json';template=r/'prompts.yaml';assert hashlib.sha256(source.read_bytes()).hexdigest()==old['source_sha256']and hashlib.sha256(template.read_bytes()).hexdigest()==old['template_sha256'];excluded=set()
for name in ['pseudo-explore','pseudo-holdout','pseudo-confirm']:excluded.update(old[name]['problem_ids'])
for name in ['pseudo-prefix64-manifest.json','pseudo-confirm-holdout-manifest.json']:excluded.update(json.loads((assets/name).read_text())['problem_ids'])
prompt=yaml.safe_load(template.read_text())['USER_PROMPT'];tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=True);eligible={64:{},128:{}};rows=json.loads(source.read_text())
for row in rows:
 problem=str(row['problem_id']);text=prompt.format(pseudocode='\n'.join(row['pseudocode_lines']));ids=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False);ids=ids.input_ids if hasattr(ids,'input_ids')else ids
 for output in [64,128]:
  if len(ids)+output<=512:eligible[output].setdefault(problem,len(ids))
report=dict(completed=True,source_sha256=old['source_sha256'],template_sha256=old['template_sha256'],max_context=512,registered_seen_problem_ids=sorted(excluded),counts={str(output):dict(total_eligible=len(values),remaining_after_registered_ID_exclusions=len(set(values)-excluded),remaining_problem_ids=sorted(set(values)-excluded))for output,values in eligible.items()},scope='Advisory length eligibility and registeredID exclusion only; no inference/training, no answers/output/speed selection, no reserved cohort assignment. Full crosshost launched-request/holdout exposureaudit still required before certifying any freshcohort.')
(r/'fresh266-eligibility-count.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
