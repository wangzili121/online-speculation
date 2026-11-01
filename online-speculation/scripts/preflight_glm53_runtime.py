#!/usr/bin/env python3
"""Check model/runner/KDA interfaces before arming a full-model smoke run."""

import argparse
from datetime import timedelta
import importlib.metadata
from pathlib import Path
import tempfile

from watch_glm53_validation import runtime_network_environment, runtime_sources, write_json


def check_cann_runtime(acl, num_devices, report):
    # 先检查驱动访问；芯片信息为空时导入原生扩展可能直接 C++ abort。
    cann = report["cann"] = {"init_code": acl.init()}
    if cann["init_code"] != 0:
        raise RuntimeError(f"CANN initialization failed before native worker import: {cann}; "
                           f"{acl.get_recent_err_msg()}")
    count, code = acl.rt.get_device_count()
    cann.update(device_count=count, device_count_code=code)
    if code != 0 or count != num_devices:
        raise RuntimeError(f"CANN device enumeration failed before native worker import: {cann}; "
                           f"{acl.get_recent_err_msg()}")
    cann["soc_name"] = acl.get_soc_name()
    if not cann["soc_name"]:
        raise RuntimeError(f"CANN returned no chip type before native worker import: {cann}")


def check_hccl_runtime(torch):
    import torch.distributed as dist

    # MoE 即使 TP=1 也会索取真实 HCCL communicator，不能只测张量分配。
    with tempfile.TemporaryDirectory(prefix="glm53-hccl-") as directory:
        try:
            dist.init_process_group("hccl", init_method=Path(directory, "store").as_uri(),
                                    rank=0, world_size=1, timeout=timedelta(seconds=60))
            backend = dist.distributed_c10d._get_default_group()._get_backend(torch.device("npu:0"))
            backend.get_hccl_comm_name(0)
            value = torch.ones(16, device="npu:0")
            dist.all_reduce(value)
            torch.npu.synchronize()
            torch.testing.assert_close(value.cpu(), torch.ones(16))
            return dict(status="passed", world_size=1, communicator_created=True)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-devices", type=int, choices=(1, 8), default=1)
    parser.add_argument("--privileged-runtime", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = dict(status="failed", image=args.image, checked_devices=[],
                  privileged_runtime=args.privileged_runtime, network_environment=runtime_network_environment())
    source_hash = runtime_sources(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 原生扩展可能直接 abort；先撤销旧 ready，不能依赖 Python finally。
    write_json(args.output, report)
    try:
        import acl
        check_cann_runtime(acl, args.num_devices, report)
        write_json(args.output, report)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM
        from blockspec.vllm_ascend.qwen38_runtime import install_runner_hooks
        from probe_glm53_ops import initialize_npu_runtime, main as probe

        importlib.import_module("vllm_ascend.worker.worker")
        report["native_worker_imported"] = True
        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")}
        initialize_npu_runtime()
        import torch
        for device in range(args.num_devices):
            report["checking_device"] = device
            write_json(args.output, report)
            torch.npu.set_device(device)
            value = torch.ones(16, device=f"npu:{device}") + 1
            torch.npu.synchronize()
            torch.testing.assert_close(value.cpu(), torch.full((16,), 2.))
            report["checked_devices"].append(device)
        report.pop("checking_device", None)
        torch.npu.set_device(0)
        report["hccl"] = check_hccl_runtime(torch)
        write_json(args.output, report)
        importlib.import_module("acl.rt")
        if not callable(getattr(Glm5NextForCausalLM, "compute_logits", None)):
            raise RuntimeError("native GLM causal LM interface is unavailable")
        install_runner_hooks()
        # 全部原生模块测试必须在同一运行镜像内通过，才会发布 ready 文件。
        import sys
        original = sys.argv
        try:
            sys.argv = ["probe_glm53_ops", "--output", str(args.output.with_name("operators.json"))]
            probe()
        finally:
            sys.argv = original
        if runtime_sources(root) != source_hash:
            raise RuntimeError("source changed during native runtime preflight")
        report["sha256"] = source_hash
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
