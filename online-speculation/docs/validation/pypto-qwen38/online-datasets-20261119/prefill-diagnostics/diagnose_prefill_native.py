import argparse,json,time,os
from pathlib import Path
from diagnose_prefill_v15 import first_difference

def main(*,guard=None):
 p=argparse.ArgumentParser()
 for name in ['model','prompts','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--tokens',type=int,default=64);args,_=p.parse_known_args()
 from pypto.runtime import configure_log
 from serving.distributed import ServingConfig
 from serving.engine import InferenceEngine
 from blockspec.pypto_qwen38.oneserving_tp2 import KVCache
 from blockspec.pypto_qwen38.oneserving_resources import bounded_ring_policy
 from orthrus_bench_dense_v14 import run_pair
 configure_log('warn');pair=json.loads(args.prompts.read_text())[:2]
 c=ServingConfig(args.model,(0,1),max_model_len=512,max_num_batched_tokens=128,max_num_seqs=2,speculative_tokens=0,prefix_cache=False,long_prefill_token_threshold=0)
 e=InferenceEngine(c);r=e.runner
 assert type(r).__module__.startswith('serving.') and not hasattr(r,'suffix')
 r.pool_blocks=32;r.cache=KVCache(r.cache.namespace,32,512,128,on_event=e._on_cache)
 policy=r._resource_policy;r._resource_policy=lambda:bounded_ring_policy(policy(),512)
 if guard:
  admission=r._check_memory_admission
  def admit():
   for i in [0,1]:guard.release(i)
   guard.close();admission()
  r._check_memory_admission=admit
 out=dict(completed=False,diagnostic_only=True,pure_native_runner=type(r).__module__+'.'+type(r).__name__,no_orthrus_model_or_head=True,visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),cases={},stage='starting')
 def save():
  t=args.output.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(args.output)
 save()
 try:
  e.start();out['stage']='measuring';save()
  for cap in [0,4]:
   r.chunk_threshold=cap;e.scheduler.long_prefill_token_threshold=cap
   first=run_pair(e,pair,args.tokens);first['invocations']=list(r.invocations)
   second=run_pair(e,pair,args.tokens);second['invocations']=list(r.invocations)
   out['cases'][str(cap)]=dict(first=first,second=second,difference=first_difference(second,first));save();print('pure AR',cap,out['cases'][str(cap)]['difference'],flush=True)
  out['completed']=True;out['stage']='completed'
 except BaseException as x:out['error']=repr(x);raise
 finally:
  try:e.stop();out['clean_shutdown']=True
  finally:save()
