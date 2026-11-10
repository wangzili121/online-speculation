#!/usr/bin/env python3
"""真实 NPU：GDN 递推前向/反向对照独立 Torch autograd，历史保持只读。"""

import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--real-shape", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("new output required")
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape, recurrence_forward
    from blockspec.pypto_qwen38.training_gdn_programs import make_recurrence, make_recurrence_vjp, make_recurrence_reduce
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    s = GDNShape() if a.real_shape else GDNShape(heads=8, query_heads=4, dimension=64)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    spec = {"x": ((s.rows, s.channels), torch.bfloat16), "g": ((s.rows, 64), torch.float32),
            "b": ((s.rows, 64), torch.float32), "history": ((s.heads, s.dimension, s.dimension), torch.float32),
            "y": ((s.rows, s.heads*s.dimension), torch.bfloat16),
            "states": (((s.rows+1)*s.heads*s.dimension, s.dimension), torch.float32),
            "dy": ((s.rows, s.heads*s.dimension), torch.bfloat16)}
    tiles = s.dimension//16
    for name in ("dq", "dk"):
        spec[name] = ((s.rows*s.heads*tiles, s.dimension), torch.float32)
    for name in ("pg", "pb"):
        spec[name] = ((s.rows*s.heads*tiles, 8), torch.float32)
    spec.update(dv=spec["y"], dx=spec["x"], dg=spec["g"], db=spec["b"])
    meta = {name: torch.empty(shape, dtype=dtype, device="meta") for name, (shape, dtype) in spec.items()}
    calls = [(make_recurrence(s), ("x", "g", "b", "history", "y", "states")),
             (make_recurrence_vjp(s), ("x", "g", "b", "states", "dy", "dq", "dk", "dv", "pg", "pb")),
             (make_recurrence_reduce(s), ("x", "dq", "dk", "dv", "pg", "pb", "dx", "dg", "db"))]
    programs = [(entry.compile(*(meta[name] for name in names), 32, config=config), names) for entry,names in calls]
    worker, handles = None, {}
    report = dict(passed=False, scope="gdn_recurrence_vjp", real_shape=a.real_shape,
                  runtime=runtime_identity(pypto), cases=[])
    if a.compile_only:
        report.update(passed=True,scope="gdn_compile_only",npu_execution=False,
                      artifacts=[str(entry.output_dir) for entry,_ in programs])
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        return
    try:
        worker = DistributedWorker([entry for entry,_ in programs], config=config)
        handles = {name: worker.alloc_tensor(shape, dtype) for name,(shape,dtype) in spec.items()}
        def read(name):
            shape, dtype = spec[name]
            value = torch.empty(shape, dtype=dtype)
            worker.copy_from(value.data_ptr(), handles[name].data_ptr, value.numel()*value.element_size())
            return value
        for width in (1, 17, 32):
            for nonzero in (False, True):
                x = (torch.randn(spec["x"][0])*.2).bfloat16()
                g = -torch.rand(s.rows,64)*.3
                b = torch.rand(s.rows,64).bfloat16().float()
                history = torch.randn(spec["history"][0])*.03 if nonzero else torch.zeros(spec["history"][0])
                dy = (torch.randn(s.rows,s.heads,s.dimension)*.1).bfloat16()
                x[width:] = g[width:] = b[width:] = float("nan")
                dy[width:] = float("nan")
                x.requires_grad_(); g.requires_grad_(); b.requires_grad_()
                y, states, _ = recurrence_forward(x,g[:,:s.heads],b[:,:s.heads],history,width,s)
                gradients = torch.autograd.grad((y[:width]*dy[:width]).sum(),(x,g,b))
                values = dict(x=x.detach(),g=g.detach(),b=b.detach(),history=history,dy=dy.reshape(spec["dy"][0]))
                for name,value in values.items():
                    worker.copy_to(handles[name].data_ptr,value.data_ptr(),value.numel()*value.element_size())
                start = time.monotonic()
                for entry,names in programs:
                    worker.run(entry,*(handles[name] for name in names),width,config=config)
                seconds = time.monotonic()-start
                torch.testing.assert_close(read("y"),y.detach().reshape(spec["y"][0]),rtol=.02,atol=.002)
                torch.testing.assert_close(read("states"),states.detach().reshape(spec["states"][0]),rtol=.002,atol=.00002)
                errors = {}
                for name,expected in zip(("dx","dg","db"),gradients,strict=True):
                    actual = read(name)
                    assert torch.isfinite(actual).all(), name
                    error = float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-20))
                    assert error < .02,(name,error)
                    errors[name] = error
                for name,value in values.items():
                    assert torch.equal(read(name).view(torch.uint8),value.view(torch.uint8)),name
                report["cases"].append(dict(width=width,nonzero_history=nonzero,errors=errors,
                    execution_seconds=seconds,inputs_byte_exact=True,passed=True))
                print(json.dumps(report["cases"][-1]),flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            worker.close()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
