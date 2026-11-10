#!/usr/bin/env python3
"""完整 AR 回答重放、随机 anchor、全块 KL 预热；评估仍走真实投机验证。"""

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time

from benchmark_qwen38_pypto import DraftMeter, idle
from benchmark_qwen38_pypto_online import generate, prepare_questions, publish_initial_suffix, summarize


def response_records(source, dataset):
    if not source.get("completed"):
        raise ValueError("response collection must have completed")
    prompts = {row["id"]:row for row in dataset["splits"]["train"]}
    heldout = {row["id"] for name in ("dev","test","calibration") for row in dataset["splits"][name]}
    rows = source["online_net"]["target_rows"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate response record")
    records = []
    old_prompts = {row["id"]:row for row in source["dataset"]["splits"]["train"]}
    if dataset["files_sha256"] != source["dataset"]["files_sha256"]:
        raise ValueError("raw data changed")
    for row in rows:
        if row["id"] in heldout or row["id"] not in prompts or prompts[row["id"]] != old_prompts[row["id"]]:
            raise ValueError("response must belong to the same training partition")
        if not row["tokens"] or len(row["tokens"]) != row["output_tokens"]:
            raise ValueError("incomplete delivered response")
        tokens=[*prompts[row["id"]]["prompt_ids"],*row["tokens"]]
        if any(type(t) is not int or not 0<=t<248320 for t in tokens):
            raise ValueError("invalid clean token ID")
        records.append(dict(id=row["id"], tokens=tokens))
    return records, sum(row["seconds"] for row in rows)


def replay_loss(learner):
    """用新参数重算同一批 loss；不调用 Adam、不发布参数，仅用于诊断步幅。"""
    positions = sum(item.valid for item in learner.replay)
    optimizer = learner.optimizer
    gradients = optimizer.accum if learner.options.last_layers == 1 else optimizer.gradients
    before = (optimizer.steps, learner.updates, learner.draft.weights.version)
    loss = 0.
    optimizer.zero_grad()
    try:
        for item in learner.replay:
            for name,value in item.inputs.items():
                learner.bridge.copy(learner.native.worker,value,learner.trainer,item.training_inputs[name])
            inputs = {**learner.shared,**item.training_inputs,**optimizer.tensors()}
            learner.workspace.run(inputs,past=item.past,width=item.width,valid=item.valid,
                                  normalizer=positions,gradients=gradients)
            handle = learner.workspace.values[learner.graph.losses.name]
            learner.trainer.run(learner.loss_program,handle,learner.loss_device,config=learner.training_config)
            learner.trainer.copy_from(learner.loss_host.data_ptr(),learner.loss_device.data_ptr,32)
            loss += learner.loss_host[0,0].item()
    finally:
        optimizer.zero_grad()
    if before != (optimizer.steps,learner.updates,learner.draft.weights.version):
        raise AssertionError("loss audit changed the published optimizer state")
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source","model","data","records-report","output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device",type=int,required=True)
    p.add_argument("--learner-device",type=int,required=True)
    p.add_argument("--last-layers",type=int,choices=(1,4),default=1)
    p.add_argument("--objective",choices=("tv","kl"),default="kl")
    p.add_argument("--steps",type=int,default=128)
    p.add_argument("--eval-steps",default="0,16,32,64,128")
    p.add_argument("--width",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--warmup-steps",type=int,default=8)
    p.add_argument("--records-per-update",type=int,default=2)
    p.add_argument("--anchors-per-record",type=int,default=4)
    p.add_argument("--seed",type=int,default=857)
    p.add_argument("--tokens",type=int,default=48)
    p.add_argument("--test-count",type=int,default=8)
    p.add_argument("--check-kernel",action="store_true",help="加载模型前核对 KL 及完整末层反向的 NPU 数值")
    p.add_argument("--audit-steps",default="",help="指定步骤在更新后重算同一批 loss，仅诊断，不额外更新")
    p.add_argument("--request-timeout",type=float,default=1200)
    a = p.parse_args()
    marks = sorted({0,a.steps,*map(int,a.eval_steps.split(","))})
    audit_steps = set(map(int,filter(None,a.audit_steps.split(","))))
    if (a.output.exists() or a.device==a.learner_device or not 2<=a.width<=32 or a.steps<1
            or marks[0]<0 or marks[-1]>a.steps or not 1<=a.tokens<=384
            or a.records_per_update<2 or a.anchors_per_record<1 or a.warmup_steps<1 or a.test_count<1
            or any(not 1<=step<=a.steps for step in audit_steps)):
        p.error("new output, two devices, positive training budgets and valid evaluation steps required")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig
    from blockspec.pypto_qwen38.ar_prefix import borrow_prefix
    from blockspec.pypto_qwen38.cold_replay import ReplayRunner, sample_jobs
    source = json.loads(a.records_report.read_text())
    batch = a.records_per_update*a.anchors_per_record
    options = OnlineConfig(objective=a.objective,last_layers=a.last_layers,stride=batch,replay_blocks=batch,
                           learning_rate=a.learning_rate,learner_device=a.learner_device)
    report = dict(completed=False, objective=f"full-block {a.objective.upper()}(teacher||draft)",
        initialization="AR-copy; no pretrained diffusion head",
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        updates=[], replay_jobs=[], samples=[], evaluations=[], audits=[], physical_block_size=32,
        records_source_sha256=hashlib.sha256(a.records_report.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    engine = runner = learner = replay = meter = None
    original_update = None
    def persist():
        temporary = a.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        temporary.replace(a.output)
    def emit(event,**fields):
        print(json.dumps(dict(event=event,**fields)),flush=True)
        persist()
    try:
        if a.check_kernel:
            report["kernel_checks"]=[]
            for suffix,flags in (("loss",["--vocabulary","248320"]),("suffix",["--suffix","--combined"])):
                output=a.output.with_suffix(f".{suffix}.json")
                command=[sys.executable,str(Path(__file__).with_name("validate_qwen38_pypto_training.py")),
                         "--device",str(a.device),"--output",str(output),"--objective",a.objective,*flags]
                subprocess.run(command,check=True,timeout=900)
                check=json.loads(output.read_text())
                if not check["passed"]:raise AssertionError("NPU kernel preflight failed")
                report["kernel_checks"].append(dict(path=str(output),passed=True))
        head = HeadSource(a.model)
        engine = make_engine(a.source,a.model,a.device,head,head_transfer="packed",attention_backend="paged",
                             group_size=4,gdn_backend="vector",online_config=options)
        dataset = report["dataset"] = prepare_questions(a.data,engine.tokenizer,seed=source["dataset"]["seed"],
            max_prompt=source["arguments"]["max_prompt"],train_count=source["arguments"]["train_count"],
            dev_count=source["arguments"]["dev_count"],test_count=a.test_count)
        records,collection_seconds = response_records(source,dataset)
        report["records"] = records
        report["collection_seconds"] = collection_seconds
        if any(len(row["tokens"])>512 for row in records):
            raise ValueError("clean response exceeds pinned total context")
        persist()
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic()-start
        runner,learner = engine.runner,engine.runner.draft.learner
        if learner.head_identity != source["head_identity"]:
            raise ValueError("collected responses were generated by a different base/head contract")
        publish_initial_suffix(learner)
        layers,owned = borrow_prefix(runner.native,head,64-a.last_layers)
        runner.draft.handles.extend(owned)
        runner.draft.weights.publish_many(layers,expected_version=0)
        runner.draft.weights.version=0
        runner.draft.block_size = a.width
        report["initial_head_sha256"] = learner.head_identity["payload_sha256"]
        report["head_identity"] = learner.head_identity
        report["shared_ar_prefix_layers"] = sorted(layers)
        meter = DraftMeter(runner.draft)
        runner.draft.propose = meter
        replay = ReplayRunner(runner,learner)
        original_update = learner.update

        def update():
            rate = a.learning_rate*min(1.,(learner.updates+1)/a.warmup_steps)
            learner.options = replace(learner.options,learning_rate=rate)
            start = time.monotonic()
            result = original_update()
            entry = dict(step=learner.updates,seconds=time.monotonic()-start,learning_rate=rate,**result)
            report["updates"].append(entry)
            emit("update",**entry)
            if learner.updates in audit_steps:
                start = time.monotonic()
                after = replay_loss(learner)
                audit = dict(step=learner.updates,before=result["loss"],after=after,seconds=time.monotonic()-start)
                report["audits"].append(audit)
                emit("same_batch_loss",**audit)
            return result
        learner.update = update

        def measure(row,target=False):
            idle(engine)
            engine.scheduler.runner = runner.native if target else runner
            runner.draft.learner = None
            meter.mode="real_head"
            meter.reset()
            before = (runner.rounds,runner.accepted,learner.updates)
            result=generate(engine,row,a.tokens,a.request_timeout)
            if learner.updates!=before[2]:
                raise AssertionError("held-out inference updated the head")
            result.update(rounds=runner.rounds-before[0],accepted=runner.accepted-before[1],
                          proposed=meter.candidates,draft_phases=dict(meter.phases))
            return result

        def evaluate(step,final=False):
            entry=dict(step=step,dev={},test={})
            for name in (("dev","test") if step==0 or final else ("dev",)):
                refs,rows=[],[]
                for row in dataset["splits"][name]:
                    refs.append(measure(row,True));rows.append(measure(row))
                entry[name]=dict(rows=rows,target_rows=refs,summary=summarize(rows,target=refs))
            report["evaluations"].append(entry)
            emit("heldout",step=step,**{k:v["summary"] for k,v in entry.items() if k!="step" and v})

        measure(dataset["splits"]["calibration"][0],True)
        measure(dataset["splits"]["calibration"][0])
        evaluate(0)
        rng=random.Random(a.seed)
        replay_seconds=0.
        for goal in marks[1:]:
            while learner.updates<goal:
                expected=learner.updates+1
                jobs=sample_jobs(records,rng,width=a.width,records_per_update=a.records_per_update,
                                 anchors_per_record=a.anchors_per_record)
                for job in jobs:
                    idle(engine)
                    replay.job=job
                    engine.scheduler.runner=replay
                    runner.draft.learner=learner
                    start=time.monotonic()
                    generate(engine,dict(id=job.record_id,prompt_ids=list(job.tokens[:job.anchors[0]])),1,a.request_timeout)
                    replay_seconds+=time.monotonic()-start
                    report["replay_jobs"].append(dict(record_id=job.record_id,anchors=job.anchors,
                        seconds=time.monotonic()-start))
                    replay.job=None
                if learner.updates!=expected:
                    raise AssertionError("one sampled replay batch must produce exactly one update")
                report["samples"] = list(replay.samples)
                report["replay_seconds"] = replay_seconds
                persist()
            evaluate(goal,final=goal==a.steps)
        if a.last_layers==1:
            from blockspec.pypto_qwen38.online_checkpoint import save
        else:
            from blockspec.pypto_qwen38.online_suffix_checkpoint import save
        start=time.monotonic()
        path=a.output.with_suffix(".checkpoint")
        save(learner.optimizer,path,learner.head_identity)
        report["checkpoint"]=dict(path=str(path),seconds=time.monotonic()-start)
        report["cost"]=dict(collection_seconds=collection_seconds,replay_seconds=replay_seconds,
            total_prewarm_seconds=collection_seconds+replay_seconds,
            update_seconds=sum(row["seconds"] for row in report["updates"]),
            supervised_positions=sum(row["reached_positions"] for row in report["updates"]),
            loss_audit_seconds=sum(row["seconds"] for row in report["audits"]),
            note="replay prefill/draft/clean teacher/Adam/publication included; no forced-acceptance speedup reported")
        final=report["evaluations"][-1]["test"]["summary"]
        margin=(1/final["target_tokens_per_second"]-1/final["tokens_per_second"]
                if final["greedy_equal_all"] else 0.)
        report["cost"]["estimated_future_tokens_to_amortize_replay"]=(replay_seconds/margin if margin>0 else None)
        report["cost"]["collection_time_origin"]="historical measured AR-only collection; records reused this run"
        report.update(completed=True,snapshot=engine.snapshot())
        emit("finished",cost=report["cost"])
    except BaseException as error:
        report["error"]=repr(error)
        raise
    finally:
        if learner is not None:
            learner.update=original_update or learner.update
        if runner is not None:
            runner.draft.learner=learner
            if meter is not None:runner.draft.propose=meter.original
            engine.scheduler.runner=runner
        try:
            if engine is not None:engine.stop()
            report["clean_shutdown"]=True
        except BaseException as error:
            report.update(completed=False,cleanup_error=repr(error))
            raise
        finally:
            persist()


if __name__=="__main__":
    main()
