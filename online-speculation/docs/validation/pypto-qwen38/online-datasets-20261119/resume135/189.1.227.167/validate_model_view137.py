from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

BASE=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105')
ROOT=BASE/'oneserving-v109-b99e118/online-background'
deadline=time.monotonic()+360
while not(ROOT/'resume136-model-view.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Verified private model view was not published')
    time.sleep(5)
receipt=json.loads((ROOT/'resume136-model-view.json').read_text())
assert receipt['completed']and receipt['weight_files_copied']==0 and not receipt['public_directory_modified']
sys.path.insert(0,str(BASE))
from host_runtime import PYTHON,environment
code=r'''from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,'/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/pypto-serving-qwen3.8-27B')
import torch
torch.set_num_threads(2)
from serving.weight import TextWeights,EXPECTED_REVISION,EXPECTED_DIGESTS
from transformers import AutoTokenizer
r=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/inference-tuning-20261116/model-view')
meta=(r/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()
assert meta[0]==EXPECTED_REVISION
config=(r/'config.json').read_bytes()
blob=hashlib.sha1(b'blob '+str(len(config)).encode()+b'\0'+config).hexdigest()
assert meta[1]==blob
weights=TextWeights(r)
q=weights.read('model.language_model.layers.63.self_attn.q_norm.weight')
assert tuple(q.shape)==(256,)and q.dtype==torch.bfloat16
tokenizer=AutoTokenizer.from_pretrained(r,local_files_only=True)
ids=tokenizer.apply_chat_template([{'role':'user','content':'Compute 1 + 1.'}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
assert isinstance(ids,list)and len(ids)>0
index=json.loads((r/'model.safetensors.index.json').read_text())['weight_map']
assert len(set(index.values()))==18 and all((r/name).is_symlink()for name in set(index.values()))
print(json.dumps(dict(completed=True,cpu_only=True,actual_native_TextWeights_passed=True,actual_safetensors_tensor_read=True,actual_tokenizer_passed=True,metadata_revision=EXPECTED_REVISION,metadata_config_blob_verified=True,expected_digests=EXPECTED_DIGESTS,view=str(r),public_model='/data/disk/models/Qwen3.8-27B',weight_files_copied=0,public_directory_modified=False)))
'''
x=subprocess.run([str(PYTHON),'-c',code],env=environment(),capture_output=True,text=True,timeout=120)
report=dict(completed=x.returncode==0,returncode=x.returncode,log=x.stdout+x.stderr,
            verification_receipt_sha256=hashlib.sha256((ROOT/'resume136-model-view.json').read_bytes()).hexdigest(),
            scope='Native weight loader, config Git blob metadata, tokenizer and tensor CPU read; real NPU gate still required')
if x.returncode:report['error']='Native model-view CPU preflight failed'
(ROOT/'resume137-model-view-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(x.returncode)
