#!/usr/bin/env python3
"""真实 last-4 尺寸：先以主机字节填满缓冲，再检查 PyPTO 清零覆盖全部位置。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--elements",type=int,default=454946816)
    p.add_argument("--legacy",action="store_true",help="reproduce the former matrix-zero path")
    a = p.parse_args()
    if a.output.exists():
        p.error("new output required")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker,ExecutionMode,RunConfig
    from blockspec.pypto_qwen38.training_flat import make_flat_zero
    from blockspec.pypto_qwen38.loading_programs import make_zero
    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910B,execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.device],num_sub_workers=0))
    factory = (lambda n,dtype:make_zero((1,n),dtype)) if a.legacy else make_flat_zero
    programs = {dtype:factory(a.elements,"fp32" if dtype == torch.float32 else "bf16").compile(
        torch.empty((1,a.elements),dtype=dtype,device="meta"),config=config) for dtype in (torch.float32,torch.bfloat16)}
    report = dict(passed=False,elements=a.elements,legacy=a.legacy,cases=[])
    worker = None
    try:
        worker = DistributedWorker(list(programs.values()),config=config)
        pattern = torch.full((16<<20,),0x3f,dtype=torch.uint8)
        observed = torch.empty_like(pattern)
        for dtype,program in programs.items():
            handle = worker.alloc_tensor((1,a.elements),dtype)
            size = a.elements*dtype.itemsize
            try:
                for iteration in range(2):
                    # 初始化不复用被测 kernel，避免两个实现同时漏写尾部而误判通过。
                    for offset in range(0,size,pattern.numel()):
                        count = min(pattern.numel(),size-offset)
                        worker.copy_to(handle.data_ptr,pattern.data_ptr(),count,dst_offset=offset)
                    worker.run(program,handle,config=config)
                    for offset in range(0,size,observed.numel()):
                        count = min(observed.numel(),size-offset)
                        worker.copy_from(observed.data_ptr(),handle.data_ptr,count,src_offset=offset)
                        assert torch.count_nonzero(observed[:count]).item() == 0,(str(dtype),iteration,offset)
                    report["cases"].append(dict(dtype=str(dtype),iteration=iteration,checked_bytes=size,passed=True))
            finally:
                worker.free_tensor(handle)
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
