"""读取真实 NPU 快照，用 CPU Torch autograd 独立核对梯度、Adam 和固定批次学习。"""

import hashlib
import json
import math
from pathlib import Path
import time


def metrics(actual, expected):
    import torch
    a,b=actual.detach().double().reshape(-1),expected.detach().double().reshape(-1)
    an,bn=float(a.norm()),float(b.norm())
    error=float((a-b).norm())/max(bn,1e-30)
    cosine=float(torch.dot(a,b))/(an*bn) if min(an,bn)>1e-30 else float(an==bn)
    return dict(relative_l2=error,cosine=cosine,absolute_max=float((a-b).abs().max()),
                actual_norm=an,reference_norm=bn)


def gradient_checks(actual, masters):
    result={}
    for name,p in masters.items():
        entry=metrics(actual[name],p.grad)
        entry["passed"]=(entry["relative_l2"]<.06 and entry["cosine"]>.99
                         if entry["reference_norm"]>1e-10 else entry["absolute_max"]<1e-7)
        result[name]=entry
    return result


def run(a):
    import torch
    from safetensors.torch import load_file
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph,SuffixShape
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    from blockspec.pypto_qwen38.training_autograd import AutogradReference
    torch.set_num_threads(a.threads)
    captured=json.loads(a.capture_report.read_text())
    if not captured.get("capture_ready") or len(captured["blocks"])!=8:
        raise ValueError("immutable complete 8-block snapshot required")
    directory=Path(captured["directory"])
    n=captured["last_layers"]
    shape=SuffixShape(**captured["shape"])
    graph=(LastLayerGraph(shape,objective="kl") if n==1 else
           SuffixGraph(n,shape,GDNShape(**captured["gdn_shape"]),objective="kl"))
    report=dict(completed=False,passed=False,phase="torch_reference",last_layers=n,updates=[],
        capture_manifest_sha256=hashlib.sha256(a.capture_report.read_bytes()).hexdigest(),
        capture=str(a.capture_report),gradient_relative_tolerance=.06,gradient_minimum_cosine=.99,
        computation="CPU Torch autograd; forward-only operators; FP32 masters with BF16 execution rounding")
    def persist():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    try:
        shared=load_file(str(directory/"shared.safetensors"))
        initial=load_file(str(directory/"initial.master.safetensors"))
        masters={name:torch.nn.Parameter(value.clone()) for name,value in initial.items()}
        reference=AutogradReference(graph,shared,masters)
        blocks=[load_file(str(directory/f"block-{i:02d}.safetensors")) for i in range(8)]
        optimizer=torch.optim.Adam(list(masters.values()),lr=captured["learning_rate"],foreach=False)
        total=0.;forward=[]
        start=time.monotonic()
        for i,(block,meta) in enumerate(zip(blocks,captured["blocks"],strict=True)):
            loss,logits=reference.loss(block,**meta)
            if not math.isfinite(float(loss.detach())):raise FloatingPointError("nonfinite reference loss")
            item=dict(block=i,loss=float(loss.detach()),
                replay_logits=metrics(block["observed_logits"],logits[:meta["valid"]]),
                serving_logits=metrics(block["proposal_logits"],logits[:meta["valid"]]))
            forward.append(item);total+=item["loss"]
            loss.backward()
            del loss,logits
            if i==0:
                report["gradient_first"]=gradient_checks(load_file(str(directory/"gradient-first.safetensors")),masters)
            print(json.dumps(dict(event="reference_block",**item)),flush=True)
            persist()
        actual_gradient=load_file(str(directory/"gradient-sum.safetensors"))
        report.update(forward=forward,initial_loss=total,initial_reference_seconds=time.monotonic()-start,
            gradient_accumulated=gradient_checks(actual_gradient,masters))
        # 使用同一份 NPU 梯度单独对照 Adam，避免把反向误差算成优化器误差。
        isolated={name:torch.nn.Parameter(value.clone()) for name,value in initial.items()}
        adam=torch.optim.Adam(list(isolated.values()),lr=captured["learning_rate"],foreach=False)
        norm=math.sqrt(sum(float(value.double().square().sum()) for value in actual_gradient.values()))
        clip=min(1.,1./(norm+1e-6))
        for name,p in isolated.items():p.grad=actual_gradient[name]*clip
        adam.step()
        checks={}
        for label in ("master","first","second"):
            observed=load_file(str(directory/f"step1.{label}.safetensors"))
            checks[label]={}
            for name,p in isolated.items():
                expected={"master":p,"first":adam.state[p]["exp_avg"],"second":adam.state[p]["exp_avg_sq"]}[label]
                item=metrics(observed[name],expected)
                item["passed"]=bool(torch.allclose(observed[name],expected.detach(),atol=2e-6,rtol=.003))
                checks[label][name]=item
        report["adam_same_gradient"]=dict(norm=norm,clip=clip,states=checks)
        report["publication_checked"]=(directory/"step1.execution.safetensors").exists()
        report["publication_exact"]={}
        if report["publication_checked"]:
            execution=load_file(str(directory/"step1.execution.safetensors"))
            device_master=load_file(str(directory/"step1.master.safetensors"))
            rendered=AutogradReference(graph,{},device_master).execution()
            report["publication_exact"]={name:bool(torch.equal(value.float(),rendered[name])) for name,value in execution.items()}
            del execution,device_master,rendered
        del isolated,adam,actual_gradient

        for step in range(a.steps):
            if step:
                optimizer.zero_grad(set_to_none=True);total=0.
                for block,meta in zip(blocks,captured["blocks"],strict=True):
                    loss,logits=reference.loss(block,**meta)
                    total+=float(loss.detach());loss.backward();del loss,logits
            norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in masters.values()))
            scale=min(1.,1./(norm+1e-6))
            for p in masters.values():p.grad.mul_(scale)
            optimizer.step()
            report["updates"].append(dict(step=step+1,loss=total,gradient_norm=norm,clip_scale=scale))
            if step==0 or (step+1)%8==0:
                print(json.dumps(dict(event="torch_fixed_batch",**report["updates"][-1])),flush=True);persist()
        with torch.no_grad():
            report["final_loss"]=sum(float(reference.loss(block,**meta)[0]) for block,meta in zip(blocks,captured["blocks"],strict=True))
        report["passed"]=(all(item["replay_logits"]["relative_l2"]<.02 for item in forward)
            and all(v["passed"] for key in ("gradient_first","gradient_accumulated") for v in report[key].values())
            and all(v["passed"] for entries in checks.values() for v in entries.values())
            and all(report["publication_exact"].values()))
        report["completed"]=True
    except BaseException as error:
        report["error"]=repr(error);raise
    finally:persist()
