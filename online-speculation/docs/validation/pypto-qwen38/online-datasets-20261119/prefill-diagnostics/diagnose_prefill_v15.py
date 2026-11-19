"""只定位 prefill / verifier 一致性；oracle 分支不作吞吐评测。"""
import argparse,json,time,hashlib,os
from pathlib import Path
from types import MethodType

def first_difference(actual,reference):
    result=[]
    for i,(a,b) in enumerate(zip(actual['tokens'],reference['tokens'],strict=True)):
        first=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        if first is None and len(a)!=len(b):first=min(len(a),len(b))
        if first is not None:result.append(dict(request=i,offset=first,actual=a[first:first+4],reference=b[first:first+4]))
    return result

def main(*,guard=None):
    p=argparse.ArgumentParser()
    for name in ['model','head','prompts','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokens',type=int,default=64)
    args,_=p.parse_known_args()
    from pypto.runtime import configure_log
    from serving.distributed import ServingConfig
    from serving.engine import InferenceEngine
    from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner
    from orthrus_bench_dense_v14 import run_pair
    configure_log('warn')
    pair=json.loads(args.prompts.read_text())[:2]
    config=ServingConfig(args.model,(0,1),max_model_len=512,max_num_batched_tokens=128,max_num_seqs=2,speculative_tokens=0,prefix_cache=False,long_prefill_token_threshold=0)
    engine=InferenceEngine(config)
    runner=engine.runner=OrthrusTP2Runner(config,args.head,draft_batch_size=2,kv_pages=32,ring_heap_min_mib=512,on_cache_event=engine._on_cache)
    engine.scheduler.runner=runner
    if guard:
        admit=runner._check_memory_admission
        def admission():
            for rank in config.devices:guard.release(rank)
            guard.close();admit()
        runner._check_memory_admission=admission
    out=dict(completed=False,diagnostic_only=True,oracle_is_not_performance_result=True,visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),cases={},stage='starting')
    def save():
        t=args.output.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(args.output)
    def run():
        start=runner.calls
        r=run_pair(engine,pair,args.tokens)
        r['invocations']=[x for x in runner.invocations if x['id']>start]
        return r
    def encoded(text):
        x=engine.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        return tuple(x['input_ids'] if hasattr(x,'keys') else x)
    original=runner.propose_many
    save()
    try:
        engine.start();out['stage']='measuring';save()
        for cap in [0,4,16,32]:
            runner.chunk_threshold=cap;engine.scheduler.long_prefill_token_threshold=cap
            runner.output_candidates=0
            reference=run();repeat=run()
            runner.output_candidates=7
            actual=run()
            refs={encoded(q):tokens for q,tokens in zip(pair,reference['tokens'],strict=True)}
            def oracle(self,requests,feedback_index=None,proposals=None,**kwargs):
                self.spec_stats['rounds']+=len(requests)
                return [refs[tuple(s.request.prompt_ids)][len(s.request.generated):len(s.request.generated)+budget] for s,budget in requests]
            runner.propose_many=MethodType(oracle,runner)
            try:truth=run()
            finally:runner.propose_many=original
            out['cases'][str(cap)]=dict(reference=reference,repeat=repeat,actual_draft=actual,oracle_verifier=truth,repeat_difference=first_difference(repeat,reference),actual_difference=first_difference(actual,reference),oracle_difference=first_difference(truth,reference))
            save();print(cap,{k:v for k,v in out['cases'][str(cap)].items() if k.endswith('difference')},flush=True)
        out['completed']=True;out['stage']='completed'
    except BaseException as e:
        out['error']=repr(e);raise
    finally:
        try:engine.stop();out['clean_shutdown']=True
        finally:save()
