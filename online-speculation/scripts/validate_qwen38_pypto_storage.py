#!/usr/bin/env python3
"""Test real NPU head storage/copies with small tensors, not the 27B model."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or args.output.exists():
        parser.error("require an assigned device and a new output")
    import torch
    import pypto
    import pypto.language as pl
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ReadOnlyHostTensor, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.weights import DraftWeights

    class SyntheticHead:
        def __init__(self):
            self.values = torch.empty(
                (64, 2, 16, 128), dtype=torch.bfloat16
            ).share_memory_()
            for layer in range(64):
                self.values[layer, 0].fill_(layer + 1)
                self.values[layer, 1].fill_(layer + 2)
            self.shared_weights = self.values.view(torch.uint8).reshape(-1)

        def banks(self, layer):
            return list(self.values[layer])

    @pl.jit
    def add(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            x = pl.load(a, [0, 0], [16, 128])
            y = pl.load(b, [0, 0], [16, 128])
            summed = pl.add(pl.cast(x, pl.FP32), pl.cast(y, pl.FP32))
            output = pl.store(pl.cast(summed, pl.BF16, mode="rint"), [0, 0], output)
        return output

    @pl.jit.host
    def check(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        add(a, b, output, device=0)

    report = dict(
        passed=False, scope="npu_head_storage_component", full_27b_model=False, cases=[]
    )
    worker = weights = None
    try:
        report["runtime"] = runtime_identity(pypto)
        head = SyntheticHead()
        host = torch.empty((16, 128), dtype=torch.bfloat16).share_memory_()
        config = RunConfig(
            platform="a2a3",
            distributed_config=DistributedConfig(
                device_ids=[args.device], num_sub_workers=0
            ),
        )
        compiled = check.compile(*head.banks(0), host, config=config)
        worker = DistributedWorker(
            [compiled],
            config=config,
            inherited_host_tensors=[ReadOnlyHostTensor(head.shared_weights), host],
        )
        output = worker.alloc_tensor((16, 128), torch.bfloat16)
        original, copies = worker.copy_to, [0]

        def counted(*a, **kw):
            copies[0] += 1
            return original(*a, **kw)

        worker.copy_to = counted
        for mode in ("resident", "stream"):
            weights = DraftWeights(head, mode)
            weights.start(worker)
            before = copies[0]
            start = time.monotonic()
            for _ in range(2):
                for layer in range(64):
                    a, b = weights.for_layer(layer)
                    worker.run(compiled, a, b, output, config=config)
                    worker.copy_from(
                        host.data_ptr(),
                        output.data_ptr,
                        host.numel() * host.element_size(),
                    )
                    if not torch.equal(host, torch.full_like(host, 2 * layer + 3)):
                        raise AssertionError(
                            f"{mode}: stale or incorrect head at layer {layer}"
                        )
            uploads = copies[0] - before
            if uploads != (0 if mode == "resident" else 256):
                raise AssertionError("unexpected per-round head upload count")
            report["cases"].append(
                dict(
                    mode=mode,
                    checks=128,
                    per_round_upload_calls=uploads // 2,
                    seconds=time.monotonic() - start,
                    passed=True,
                )
            )
            weights.close()
            weights = None
        # 共享 head 注册为只读，误把它当输出缓冲时须在主机侧拒绝。
        try:
            worker.copy_from(head.shared_weights.data_ptr(), output.data_ptr, 4096)
        except ValueError as error:
            if "ReadOnlyHostTensor" not in str(error):
                raise
        else:
            raise AssertionError("read-only head accepted a D2H write")
        report.update(readonly_head_guard=True, passed=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if weights is not None:
                weights.close()
            if worker is not None:
                worker.close()
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
