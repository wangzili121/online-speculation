"""单卡验证真实反馈传输：临时缓冲与 fork 前登记缓冲；不代表 TP2/整流吞吐。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,statistics,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    import torch
    from collections import Counter
    from types import SimpleNamespace
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker,ExecutionMode,RunConfig
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
    from blockspec.pypto_qwen38.online_feedback_programs import make_loss_sum
    torch.set_num_threads(1);torch.manual_seed(141)
    spec=importlib.util.spec_from_file_location('blockspec.pypto_qwen38.feedback_reference141',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    config=RunConfig(backend_type=BackendType.Ascend910B,execution_mode=ExecutionMode.ONBOARD,distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0))
    report=dict(passed=False,clean_shutdown=False,scope='Single physical card with two vocabulary-shaped buffers; real DMA, no TP collective/model or fullstream speed claim',reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),cases=[])
    worker=None;handles=[];V=124160
    parts=torch.empty((2,4,V),dtype=torch.bfloat16).share_memory_();teacher=torch.empty((4,2*V),dtype=torch.bfloat16).share_memory_();hidden=torch.empty((32,5120),dtype=torch.bfloat16).share_memory_()
    try:
        p=make_loss_sum().compile(torch.empty((32,8),dtype=torch.float32,device='meta'),torch.empty((1,8),dtype=torch.float32,device='meta'),config=config)
        worker=DistributedWorker([p],config=config,inherited_host_tensors=[parts,teacher,hidden])
        def alloc(x):
            d=worker.alloc_tensor(tuple(x.shape),x.dtype,init=x);handles.append(d);return d
        native=[torch.randn(8,V,dtype=torch.bfloat16)for _ in range(2)];device=[alloc(x)for x in native];output=alloc(torch.zeros(32,2*V,dtype=torch.bfloat16));check=torch.empty((32,2*V),dtype=torch.bfloat16)
        def trainer(cls,temp):
            t=cls(None,device=0,last_layers=1,teacher_temperature=temp);t.feedback_hidden,t.feedback_teacher_parts,t.feedback_teacher=hidden,parts,teacher;t.worker=worker;t.inputs={'teacher':output};t.runner=SimpleNamespace(outputs={'target':(None,device)},_last_layout={'output_key':'target'},spec_stats=Counter())
            t.read=lambda rank,source,dest,offset=0:worker.copy_from(dest.data_ptr(),source[rank].data_ptr,dest.nbytes,src_offset=offset)
            return t
        for temp in [1.,.0625]:
            old,new=trainer(ref.TP2Online,temp),trainer(TP2Online,temp)
            for count,slot,row in [(1,0,0),(3,0,1),(4,1,0)]:
                times={'temporary':[],'registered':[]};expected=torch.cat([x[4*slot+row:4*slot+row+count]for x in native],dim=-1);expected=(expected.float()/temp).to(expected.dtype);copies=[]
                for name,t in [('temporary',old),('registered',new)]:
                    t.feedback=dict(candidates=31,valid=2,complete=False);t.capture_teacher(slot,2,count,final=True,source_row=row)
                    worker.copy_from(check.data_ptr(),output.data_ptr,check.nbytes);copies.append(check[2:2+count].clone());assert torch.equal(copies[-1],expected)
                assert torch.equal(*copies)
                for i in range(24):
                    order=[('temporary',old),('registered',new)]if i%2==0 else [('registered',new),('temporary',old)]
                    for name,t in order:
                        t.feedback=dict(candidates=31,valid=2,complete=False);start=time.monotonic_ns();t.capture_teacher(slot,2,count,final=True,source_row=row);elapsed=time.monotonic_ns()-start
                        if i>=4:times[name].append(elapsed)
                report['cases'].append(dict(teacher_temperature=temp,count=count,batch_slot=slot,source_row=row,exact=True,raw_ns=times,median_ns={n:statistics.median(v)for n,v in times.items()}))
        report['passed']=True
    finally:
        if worker is not None:
            for d in reversed(handles):worker.free_tensor(d)
            worker.close();report['clean_shutdown']=True
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],clean_shutdown=report['clean_shutdown'],medians=[c['median_ns']for c in report['cases']])))
if __name__=='__main__':main()
