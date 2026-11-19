"""仅按来源分区、去重和 token 长度选择长学习流；不看答案或生成效果。"""
from pathlib import Path
import hashlib,json,random
from transformers import AutoTokenizer
b=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');a=b/'oneserving-v109-b99e118/online-tp2-test/assets'
f=b/'online-effect-gsm8k/train.jsonl';data=f.read_bytes();sha=hashlib.sha256(data).hexdigest();assert sha=='17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
rows=[json.loads(s) for s in data.splitlines()]
# 从 train 的不同索引区间取样，并排除本次已测的 384 题。
used=set(json.loads((a/'gsm-train384.json').read_text()))
tok=AutoTokenizer.from_pretrained(str(b/'inference-tuning-20261116/model-view'),local_files_only=True,trust_remote_code=True)
sets={}
for name,indices,count,seed in [('gsm-train1024',range(2048,5000),1024,147743),('gsm-reserved1024',range(5000,len(rows)),1024,147744),('gsm-reserved-holdout32',range(5000,len(rows)),32,147745)]:
 indices=list(indices);random.Random(seed).shuffle(indices);texts=[];chosen=[];sizes=[]
 for index in indices:
  q=rows[index]['question']
  if q in used:continue
  ids=tok.apply_chat_template([dict(role='user',content=q)],tokenize=True,add_generation_prompt=True,enable_thinking=False);ids=ids.input_ids if hasattr(ids,'input_ids') else ids
  if len(ids)>224:continue
  texts.append(q);chosen.append(index);sizes.append(len(ids));used.add(q)
  if len(texts)==count:break
 assert len(texts)==count,(name,len(texts))
 path=a/(name+'.json');assert not path.exists();path.write_text(json.dumps(texts,ensure_ascii=False)+'\n')
 m=dict(source='openai/grade-school-math',source_sha256=sha,split='train',index_partition=([2048,5000] if name=='gsm-train1024' else [5000,len(rows)]),seed=seed,indices=chosen,prompt_tokens=sizes,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_total_context=512,max_new_tokens=192,answers_used=False,purpose='exploration' if name=='gsm-train1024' else 'reserved independent problems; throughput study, not official test accuracy',excludes='previous train384, all earlier partitions below2048, other sets constructed in this script')
 path.with_name(name+'-manifest.json').write_text(json.dumps(m,indent=2)+'\n');sets[name]=dict(count=count,sha256=m['sha256'],max_prompt_tokens=max(sizes))
print(json.dumps(sets,indent=2))
