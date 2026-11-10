#!/usr/bin/env python3
"""真实 27B 反馈的数值快照与固定批次训练；参考对照可在 NPU 释放后运行。"""

import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import random
import time


@dataclass
class ReadableSpan:
    owner: object
    shape: tuple
    dtype: object
    offset: int


def read_tensor(worker, value):
    import torch
    from blockspec.pypto_qwen38.loading import tensor_layout
    if isinstance(value,ReadableSpan):
        shape,dtype,owner,offset=value.shape,value.dtype,value.owner,value.offset
    else:
        shape,dtype=tensor_layout(value);owner,offset=value,0
    host = torch.empty(shape,dtype=dtype)
    worker.copy_from(host.data_ptr(),owner.data_ptr,host.numel()*host.element_size(),src_offset=offset)
    return host


def flat_views(optimizer, owner):
    import torch
    return {e.parameter.name:ReadableSpan(owner,e.parameter.shape,torch.float32,e.offset*4)
            for e in optimizer.layout.entries}


def state_views(optimizer, column, last_layers):
    if last_layers==1:
        return {name:state[column] for name,state in optimizer.state.items()}
    return flat_views(optimizer,optimizer.state[column])


def execution_views(optimizer, last_layers):
    import torch
    if last_layers==1:return optimizer.tensors()
    ordinary,special=optimizer.execution[optimizer.index]
    return {e.parameter.name:ReadableSpan(special if e.special is not None else ordinary,e.parameter.shape,
        torch.float32 if e.special is not None else torch.bfloat16,(e.special*4 if e.special is not None else e.offset*2))
        for e in optimizer.layout.entries}


def capture(a):
    import torch
    from safetensors.torch import save_file
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.cold_replay import ReplayRunner,sample_jobs
    from benchmark_qwen38_pypto_online import prepare_questions,publish_initial_suffix,generate
    from benchmark_qwen38_pypto_cold import response_records,replay_loss
    from benchmark_qwen38_pypto import idle
    torch.set_num_threads(4)
    directory = a.output.with_suffix(".tensors")
    directory.mkdir(parents=True,exist_ok=False)
    report = dict(completed=False,phase="capture",last_layers=a.last_layers,learning_rate=a.learning_rate,
        fixed_batch=True,updates=[],blocks=[],directory=str(directory),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    engine = learner = None
    original_run = original_step = original_propose = None

    def persist():
        a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")

    def save(name, values):
        tensors = {key:read_tensor(learner.trainer,value) for key,value in values.items()}
        save_file(tensors,str(directory/name))

    try:
        source = json.loads(a.records_report.read_text())
        head = HeadSource(a.model)
        options = OnlineConfig(objective="kl",last_layers=a.last_layers,stride=8,replay_blocks=8,
                               learning_rate=a.learning_rate,learner_device=a.learner_device)
        engine = make_engine(a.source,a.model,a.device,head,head_transfer="packed",attention_backend="paged",
                             group_size=4,gdn_backend="vector",online_config=options)
        dataset = prepare_questions(a.data,engine.tokenizer,seed=source["dataset"]["seed"],
            max_prompt=source["arguments"]["max_prompt"],train_count=source["arguments"]["train_count"],
            dev_count=source["arguments"]["dev_count"],test_count=source["arguments"]["test_count"])
        records,_ = response_records(source,dataset)
        engine.start()
        runner,learner = engine.runner,engine.runner.draft.learner
        if learner.head_identity!=source["head_identity"]:raise ValueError("base/head provenance changed")
        publish_initial_suffix(learner)
        layers,owned = borrow_prefix(runner.native,head,64-a.last_layers)
        runner.draft.handles.extend(owned)
        runner.draft.weights.publish_many(layers,expected_version=0);runner.draft.weights.version=0
        runner.draft.block_size=a.width
        report.update(head_identity=learner.head_identity,shape=asdict(learner.graph.shape),
            gdn_shape=asdict(learner.graph.gdn_shape) if a.last_layers>1 else None,
            records_source_sha256=hashlib.sha256(a.records_report.read_bytes()).hexdigest())
        save("shared.safetensors",learner.shared)
        save("initial.master.safetensors",state_views(learner.optimizer,0,a.last_layers))
        original_run,original_step,original_propose = learner.workspace.run,learner.optimizer.step,runner.draft.propose
        proposals = []

        def propose(request,anchor,width):
            result=original_propose(request,anchor,width)
            if learner.updates==0:proposals.append(runner.draft.read_logits(width-1).clone())
            return result
        runner.draft.propose=propose

        def workspace(inputs,**kwargs):
            result=original_run(inputs,**kwargs)
            if learner.updates==0:
                index=len(report["blocks"])
                dynamic={name:value for name,value in inputs.items()
                         if name not in learner.shared and name not in learner.graph.gradients}
                tensors={name:read_tensor(learner.trainer,value) for name,value in dynamic.items()}
                logits=read_tensor(learner.trainer,learner.workspace.values[learner.graph.logits.name])
                tensors["observed_logits"]=logits[:kwargs["valid"]].clone()
                tensors["proposal_logits"]=proposals[index]
                save_file(tensors,str(directory/f"block-{index:02d}.safetensors"))
                if index==0:
                    save("gradient-first.safetensors",result if a.last_layers==1 else flat_views(learner.optimizer,learner.optimizer.scratch))
                report["blocks"].append({k:v for k,v in kwargs.items() if k!="gradients"})
                persist()
            return result
        learner.workspace.run=workspace

        def step(publisher,rate):
            first=learner.optimizer.steps==0
            if first:
                gradients=(learner.optimizer.accum if a.last_layers==1 else
                           flat_views(learner.optimizer,learner.optimizer.candidate[0]))
                save("gradient-sum.safetensors",gradients)
            result=original_step(publisher,rate)
            if first:
                for column,label in enumerate(("master","first","second")):
                    save(f"step1.{label}.safetensors",state_views(learner.optimizer,column,a.last_layers))
                save("step1.execution.safetensors",execution_views(learner.optimizer,a.last_layers))
                report["first_step"]=result
            return result
        learner.optimizer.step=step

        replay=ReplayRunner(runner,learner)
        jobs=sample_jobs(records,random.Random(857),width=a.width)
        engine.scheduler.runner=replay
        for job in jobs:
            idle(engine);replay.job=job
            generate(engine,dict(id=job.record_id,prompt_ids=list(job.tokens[:job.anchors[0]])),1,1200)
            replay.job=None
        idle(engine)
        if learner.updates!=1 or len(report["blocks"])!=8:raise AssertionError("one complete 8-block update required")
        engine.scheduler.runner=runner
        report["updates"].append(dict(step=1,**learner.last_result))
        report["samples"]=replay.samples
        report["capture_ready"]=True
        # 参考进程只读这份不再修改的快照，可与后续 NPU 固定批次训练并行。
        (directory/"manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        # 前缀、AR 历史和教师固定，只重放同一批后段，排除数据变化与采集开销。
        for _ in range(1,a.steps):
            start=time.monotonic();result=learner.update()
            report["updates"].append(dict(step=learner.updates,seconds=time.monotonic()-start,**result))
            print(json.dumps(dict(event="fixed_batch_update",**report["updates"][-1])),flush=True)
            persist()
        report["final_loss"]=replay_loss(learner)
        save("final.master.safetensors",state_views(learner.optimizer,0,a.last_layers))
        if a.last_layers==1:
            from blockspec.pypto_qwen38.online_checkpoint import save as checkpoint
        else:
            from blockspec.pypto_qwen38.online_suffix_checkpoint import save as checkpoint
        report["checkpoint"]=str(a.output.with_suffix(".checkpoint"))
        checkpoint(learner.optimizer,report["checkpoint"],learner.head_identity)
        report.update(completed=True,snapshot=engine.snapshot())
    except BaseException as error:
        report["error"]=repr(error)
        raise
    finally:
        if learner is not None:
            if original_run is not None:learner.workspace.run=original_run
            if original_step is not None:learner.optimizer.step=original_step
            if original_propose is not None:engine.runner.draft.propose=original_propose
            engine.scheduler.runner=engine.runner
        try:
            if engine is not None:engine.stop()
            report["clean_shutdown"]=True
        except BaseException as error:
            report.update(completed=False,cleanup_error=repr(error));raise
        finally:persist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("capture","reference"),default="capture")
    p.add_argument("--source",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--data",type=Path)
    p.add_argument("--records-report",type=Path)
    p.add_argument("--device",type=int)
    p.add_argument("--learner-device",type=int)
    p.add_argument("--last-layers",type=int,choices=(1,4),default=1)
    p.add_argument("--steps",type=int,default=128)
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--threads",type=int,default=16)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--capture-report",type=Path)
    a=p.parse_args()
    if a.output.exists() or a.steps<1:p.error("new output and positive step budget required")
    if a.phase=="reference":
        if a.capture_report is None:p.error("reference requires --capture-report")
        from diagnose_qwen38_pypto_reference import run
        return run(a)
    if any(getattr(a,k) is None for k in ("source","model","data","records_report","device","learner_device")):
        p.error("capture requires model, source, data, records and two devices")
    if a.device==a.learner_device:p.error("separate model and learner devices required")
    capture(a)


if __name__=="__main__":main()
