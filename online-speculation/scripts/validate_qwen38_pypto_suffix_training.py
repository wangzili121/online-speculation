#!/usr/bin/env python3
"""last-2/4 完整 PyPTO 后段梯度对照 Torch autograd；不加载真实主模型。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--last-layers",type=int,choices=(2,3,4),default=2)
    parser.add_argument("--compile-only",action="store_true")
    parser.add_argument("--optimizer",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_graph import SuffixShape
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms,TrainingWorkspace
    from blockspec.pypto_qwen38.training_reference import evaluate
    torch.set_num_threads(4)
    torch.manual_seed(71)
    graph = SuffixGraph(args.last_layers,SuffixShape(hidden=128,intermediate=256,vocabulary=128,
        q_heads=4,kv_heads=2,dimension=64,context=64,rotary=32),GDNShape(heads=8,query_heads=4,dimension=64))
    config = RunConfig(backend_type=BackendType.Ascend910B,execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device],num_sub_workers=0))
    programs = TrainingPrograms(graph,config,combined=True,chunk_size=24)
    optimization = None
    if args.optimizer:
        from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizerPrograms
        optimization = SuffixOptimizerPrograms(graph,config,clip_norm=.00001)
    report = dict(passed=False,last_layers=args.last_layers,chunks=len(programs.chunks),
                  npu_execution=not args.compile_only,cases=[])
    worker = workspace = None
    try:
        if args.compile_only:
            report["passed"] = True
            return
        worker = DistributedWorker([*programs.programs.values(),*(optimization.programs() if optimization else ())],config=config)
        workspace = TrainingWorkspace(programs,worker)
        buffers = {name:worker.alloc_tensor(value.shape,torch.bfloat16 if value.dtype == "bf16" else torch.float32)
                   for name,value in graph.inputs.items()}
        parameter_map = {p.name:p for p in graph.parameters}
        for width,past in ((17,7),(32,0)):
            inputs,masters = {},{}
            for name,value in graph.inputs.items():
                x = torch.randn(value.shape)*(.03 if "weight" in name or "mlp" in name or name == "lm_head" else .15)
                if name in parameter_map:
                    param = parameter_map[name]
                    raw = torch.nn.Parameter(x)
                    masters[name] = raw
                    execution = raw if param.raw_dtype == "fp32" else raw.bfloat16()
                    inputs[name] = execution.float()+1 if param.offset_norm else execution
                elif "gamma" in name:
                    inputs[name] = x+1
                else:
                    inputs[name] = x if value.dtype == "fp32" else x.bfloat16()
                tensor = inputs[name].detach().contiguous()
                worker.copy_to(buffers[name].data_ptr,tensor.data_ptr(),tensor.numel()*tensor.element_size())
            valid = min(4,width-1)
            values = evaluate(graph,inputs,past=past,width=width,valid=valid,normalizer=7)
            expected = torch.autograd.grad(values[graph.losses.name][:,0].sum(),tuple(masters.values()),retain_graph=True)
            gradients = workspace.run(buffers,past=past,width=width,valid=valid,normalizer=7)
            errors = {}
            failed = []
            for (name,_),reference in zip(masters.items(),expected,strict=True):
                actual = torch.empty(graph.gradients[name].shape,dtype=torch.float32)
                worker.copy_from(actual.data_ptr(),gradients[name].data_ptr,actual.numel()*4)
                assert torch.isfinite(actual).all(),name
                error = float(((actual-reference).norm()/reference.norm().clamp_min(1e-30)).detach())
                errors[name] = error
                if error >= .06:
                    manual = values[graph.gradients[name].name].detach()
                    failed.append(dict(name=name,autograd_error=error,
                        manual_error=float((actual-manual).norm()/manual.norm().clamp_min(1e-30)),
                        reference_norm=float(reference.norm()),absolute_max=float((actual-reference).abs().max())))
            for name,expected_input in inputs.items():
                actual = torch.empty_like(expected_input)
                worker.copy_from(actual.data_ptr(),buffers[name].data_ptr,actual.numel()*actual.element_size())
                assert torch.equal(actual.view(torch.uint8),expected_input.detach().view(torch.uint8)),name
            report["cases"].append(dict(width=width,past=past,errors=errors,inputs_byte_exact=True))
            print(json.dumps(report["cases"][-1]),flush=True)
            if failed:
                diagnostics = []
                observed = {}
                for node in graph.nodes:
                    for value in node.outputs:
                        reference = values[value.name].detach().float()
                        actual = torch.empty(value.shape,dtype=torch.bfloat16 if value.dtype == "bf16" else torch.float32)
                        handle = workspace.values[value.name]
                        worker.copy_from(actual.data_ptr(),handle.data_ptr,actual.numel()*actual.element_size())
                        observed[value.name] = actual
                        error = float((actual.float()-reference).norm()/reference.norm().clamp_min(1e-30))
                        diagnostics.append(dict(name=value.name,kind=node.kind,error=error,
                            absolute_max=float((actual.float()-reference).abs().max())))
                local = evaluate(graph,inputs,past=past,width=width,valid=valid,normalizer=7,observed=observed)
                for item in diagnostics:
                    actual,reference = observed[item["name"]].float(),local[item["name"]].detach().float()
                    item["local_error"] = float((actual-reference).norm()/reference.norm().clamp_min(1e-30))
                report["diagnostics"] = dict(failed=failed,nodes=diagnostics)
                # TV 的绝对值在 p=q 处不可微；BF16 前向微差可能改变 sign。
                # 先逐阶段检查同一输入，再用设备实际 TV 梯度驱动独立 autograd，不能放宽梯度阈值。
                assert max(x["local_error"] for x in diagnostics) < .02, "local stage mismatch"
                tv = next(node for node in graph.nodes if node.kind == "tv")
                upstream = observed[tv.outputs[0].name]
                matched = torch.autograd.grad(values[graph.logits.name],tuple(masters.values()),grad_outputs=upstream)
                matched_errors = {}
                for (name,_),reference in zip(masters.items(),matched,strict=True):
                    actual = observed[graph.gradients[name].name].float()
                    error = float((actual-reference).norm()/reference.norm().clamp_min(1e-30))
                    assert error < .06,("matched TV subgradient",name,error)
                    matched_errors[name] = error
                q = inputs["teacher"][:valid].float().softmax(-1)
                p = values[graph.logits.name][:valid].detach().float().softmax(-1)
                device_p = observed[graph.logits.name][:valid].float().softmax(-1)
                sign_changes = int(((p-q).sign() != (device_p-q).sign()).sum())
                assert sign_changes > 0, "gradient mismatch has no TV sign change explanation"
                tv_local = max(x["local_error"] for x in diagnostics if x["kind"] == "tv")
                assert tv_local < 1e-5, "TV derivative differs at identical inputs"
                report["diagnostics"].update(matched_tv_errors=matched_errors,tv_sign_changes=sign_changes,
                    probability_delta_max=float((p-device_p).abs().max()),passed=True)
        if optimization is not None:
            report["optimizer"] = validate_optimizer(optimization,worker,graph,inputs,masters,gradients,args.output)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            worker.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")


def validate_optimizer(programs,worker,graph,inputs,masters,gradients,output):
    import torch
    from types import SimpleNamespace
    from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer
    from blockspec.pypto_qwen38.online_suffix_checkpoint import save,restore
    from blockspec.pypto_qwen38.online_checkpoint import STATES
    optimizer = SuffixOptimizer(programs,worker)
    restored = None
    class Publisher:
        version = 0
        def publish_flat(self,layout,values,*,expected_version):
            assert expected_version == self.version
            assert len(layout.banks(*values)) == graph.last_layers
            self.version += 1
    publisher = Publisher()
    by_layer = {}
    for p in graph.parameters:
        by_layer.setdefault(p.layer,[]).append(inputs[p.name].detach().contiguous())
    raw_norms = {p.name:masters[p.name].detach().bfloat16().reshape(p.shape)
                 for p in graph.parameters if p.offset_norm}
    def read(handle):
        value = torch.empty(handle.shape,dtype=handle.dtype)
        worker.copy_from(value.data_ptr(),handle.data_ptr,value.numel()*value.element_size())
        return value
    try:
        optimizer.initialize(SimpleNamespace(banks=lambda layer:by_layer[layer]),raw_norms)
        reference = torch.nn.Parameter(read(optimizer.state[0]))
        adam = torch.optim.Adam([reference],lr=1e-4)
        flat_gradient = torch.zeros(programs.layout.shape)
        for e in programs.layout.entries:
            value = torch.empty(e.parameter.shape,dtype=torch.float32)
            worker.copy_from(value.data_ptr(),gradients[e.parameter.name].data_ptr,value.numel()*4)
            flat_gradient[0,e.offset:e.offset+value.numel()] = value.flatten()
        worker.copy_to(optimizer.scratch.data_ptr,flat_gradient.data_ptr(),flat_gradient.numel()*4)
        result = None
        for _ in range(2):
            optimizer.zero_grad(); optimizer.accumulate()
            result = optimizer.step(publisher,1e-4)
            reference.grad = flat_gradient.clone()*min(1.,.00001/(flat_gradient.norm().item()+1e-6))
            adam.step(); adam.zero_grad()
            torch.testing.assert_close(read(optimizer.state[0]),reference.detach(),rtol=2e-5,atol=1e-7)
        for e in programs.layout.entries:
            p = e.parameter
            ordinary,special = optimizer.execution[optimizer.index]
            if e.special is None:
                actual = read(ordinary)[0,e.offset:e.offset+torch.tensor(p.shape).prod()].reshape(p.shape)
                expected = reference.detach()[0,e.offset:e.offset+actual.numel()].reshape(p.shape).bfloat16()
            else:
                actual = read(special)[0,e.special:e.special+torch.tensor(p.shape).prod()].reshape(p.shape)
                raw = reference.detach()[0,e.offset:e.offset+actual.numel()].reshape(p.shape)
                expected = raw.bfloat16().float()+1 if p.offset_norm else raw
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        directory = output.with_suffix(".checkpoint")
        save(optimizer,directory,{"reference":"small-suffix"})
        restored = SuffixOptimizer(programs,worker)
        cold_publisher = Publisher()
        restore(restored,cold_publisher,directory,{"reference":"small-suffix"})
        assert restored.steps == cold_publisher.version == optimizer.steps == 2
        for name,a,b in zip(STATES,optimizer.state,restored.state,strict=True):
            torch.testing.assert_close(read(a),read(b),atol=0,rtol=0,msg=name)
        return dict(passed=True,steps=2,parameter_tensors=len(graph.parameters),
                    checkpoint_byte_exact=True,gradient_norm=result["gradient_norm"],clip_scale=result["clip_scale"])
    finally:
        if restored is not None:
            restored.close()
        optimizer.close()


if __name__ == "__main__":
    main()
