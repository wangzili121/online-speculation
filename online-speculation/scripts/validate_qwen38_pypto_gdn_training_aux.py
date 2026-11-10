#!/usr/bin/env python3
"""GDN 卷积、门控、gated RMSNorm 的 NPU 前向/反向独立对照。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.training_gdn_reference import (
        GDNShape, convolution_forward, gates_forward, tail_forward,
    )
    from blockspec.pypto_qwen38.training_gdn_aux_programs import (
        make_convolution, make_gates, make_gate_reduce, make_tail, make_tail_weight,
    )
    torch.set_num_threads(4)
    torch.manual_seed(731)
    s = GDNShape() if args.real_shape else GDNShape(heads=8, query_heads=4, dimension=64)
    K,C,H,D = s.rows,s.channels,s.heads,s.dimension
    bf,fp = torch.bfloat16,torch.float32
    spec = dict(x=((K,C),bf), weight=((4,C),bf), history=((4,C),bf), y=((K,C),bf), pre=((K,C),bf),
                dy=((K,C),bf), dx=((K,C),bf), dw=((4,C),fp), a=((K,128),bf), b=((K,128),bf),
                al=((1,64),fp), bias=((1,64),bf), g=((K,64),fp), beta=((K,64),fp),
                dg=((K,64),fp), dbeta=((K,64),fp), da=((K,128),bf), db=((K,128),bf),
                pa=((K,64),fp), pb=((K,64),fp), dal=((1,64),fp), dbias=((1,64),fp),
                core=((K,H*D),bf), z=((K,H*D),bf), gamma=((1,D),bf), gated=((K,H*D),bf),
                normed=((K*H,D),bf), weighted=((K*H,D),bf), inverse=((K*H,8),fp),
                upstream=((K,H*D),bf), dcore=((K,H*D),bf), dz=((K,H*D),bf),
                dweighted=((K*H,D),bf), dgamma=((1,D),fp))
    definitions = [
        (make_convolution(s), ("x","weight","history","y","pre"), True),
        (make_convolution(s,backward=True), ("x","weight","history","pre","dy","dx","dw"), True),
        (make_gates(s), ("a","b","al","bias","g","beta"), True),
        (make_gates(s,backward=True), ("a","b","al","bias","dg","dbeta","da","db","pa","pb"), True),
        (make_gate_reduce(s), ("pa","pb","dal","dbias"), False),
        (make_tail(s), ("core","z","gamma","gated","normed","weighted","inverse"), False),
        (make_tail(s,backward=True), ("core","z","gamma","weighted","inverse","upstream","dcore","dz","dweighted"), False),
        (make_tail_weight(s), ("normed","dweighted","dgamma"), False),
    ]
    config = RunConfig(backend_type=BackendType.Ascend910B,execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device],num_sub_workers=0))
    programs = []
    for entry,names,width in definitions:
        meta = [torch.empty(spec[n][0],dtype=spec[n][1],device="meta") for n in names]
        programs.append((entry.compile(*meta,*((32,) if width else ()),config=config),names,width))
    report = dict(passed=False,real_shape=args.real_shape,npu_execution=not args.compile_only,cases=[])
    worker = None
    try:
        if args.compile_only:
            report["passed"] = True
            return
        worker = DistributedWorker([p for p,_,_ in programs],config=config)
        buffers = {n:worker.alloc_tensor(shape,dtype) for n,(shape,dtype) in spec.items()}
        def upload(n,x):
            worker.copy_to(buffers[n].data_ptr,x.data_ptr(),x.numel()*x.element_size())
        def read(n):
            shape,dtype = spec[n]
            x = torch.empty(shape,dtype=dtype)
            worker.copy_from(x.data_ptr(),buffers[n].data_ptr,x.numel()*x.element_size())
            return x
        for width in (1,17,32):
            values = {n:(torch.randn(shape)*.2).to(dtype) for n,(shape,dtype) in spec.items()}
            for name in ("x","a","b","dy","dg","dbeta"):
                values[name][width:] = float("nan")
            for name in ("core","z","upstream"):
                values[name][width:] = 0
            values["gamma"] += 1
            for n,x in values.items():
                upload(n,x)
            x,w,a,b,al,bias,core,z,gamma = [values[n].clone().requires_grad_() for n in
                ("x","weight","a","b","al","bias","core","z","gamma")]
            y,_ = convolution_forward(x,w,values["history"],width)
            conv_grad = torch.autograd.grad((y[:width]*values["dy"][:width]).sum(),(x,w))
            g,beta = gates_forward(a[:width],b[:width],al,bias,H)
            gate_grad = torch.autograd.grad((g*values["dg"][:width,:H]+beta*values["dbeta"][:width,:H]).sum(),(a,b,al,bias))
            gated,_,_,_ = tail_forward(core.reshape(K,H,D),z.reshape(K,H,D),gamma)
            tail_grad = torch.autograd.grad((gated*values["upstream"].reshape(K,H,D)).sum(),(core,z,gamma))
            for program,names,needs_width in programs:
                worker.run(program,*(buffers[n] for n in names),*((width,) if needs_width else ()),config=config)
            expected = dict(y=y,g=g,beta=beta,gated=gated.reshape(K,H*D),
                **dict(zip(("dx","dw"),conv_grad,strict=True)),
                **dict(zip(("da","db","dal","dbias"),gate_grad,strict=True)),
                **dict(zip(("dcore","dz","dgamma"),tail_grad,strict=True)))
            errors = {}
            for name,reference in expected.items():
                actual = read(name)
                if name in ("g","beta"):
                    actual = actual[:width,:H]
                reference = reference.detach().float()
                actual = actual.float()
                assert torch.isfinite(actual).all(),name
                error = float((actual-reference).norm()/reference.norm().clamp_min(1e-20))
                assert error < .025,(name,error)
                errors[name] = error
            for name in ("x","weight","history","a","b","al","bias","core","z","gamma"):
                assert torch.equal(read(name).view(torch.uint8),values[name].view(torch.uint8)),name
            report["cases"].append(dict(width=width,errors=errors,inputs_byte_exact=True))
            print(json.dumps(report["cases"][-1]),flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            worker.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
