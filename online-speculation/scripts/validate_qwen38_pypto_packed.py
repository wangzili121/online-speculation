#!/usr/bin/env python3
"""Audit real head uploads and an existing Full Attention prepare program."""

import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.device < 0:
        parser.error("require a new output and assigned device")
    import torch
    import pypto
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ReadOnlyHostTensor, RunConfig
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.draft import metadata
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.programs import make_programs
    from blockspec.pypto_qwen38.weights import DraftWeights

    report = dict(passed=False, scope="real_head_transfer_and_full_prepare_component",
                  full_model=False, cases=[], runtime=runtime_identity(pypto))
    worker, strategies, allocated = None, {}, []
    try:
        torch.manual_seed(1234)
        head = HeadSource(args.model)
        head.prepare()
        runtime = native_module(args.source, "runtime")
        inputs = metadata(head, runtime.single_meta_samples())["full_prepare"]
        config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0))
        program = make_programs(args.source)["full_prepare"].compile(*inputs, config=config)
        maximum = max(values[-1].data_ptr() + values[-1].numel() * values[-1].element_size()
                      - values[0].data_ptr() for values in (head.banks(i) for i in range(64)))
        readback = torch.empty(maximum, dtype=torch.uint8).share_memory_()
        outputs = [torch.empty(value.shape, dtype=value.dtype).share_memory_() for value in inputs[9:13]]
        worker = DistributedWorker([program], config=config,
            inherited_host_tensors=[ReadOnlyHostTensor(head.shared_weights), readback, *outputs])
        constants = {}
        for index in (0, 1, 7, 8):
            sample = inputs[index]
            value = (torch.randn(sample.shape, dtype=sample.dtype) * .01 if index == 0
                     else torch.ones(sample.shape, dtype=sample.dtype))
            constants[index] = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)
            allocated.append(constants[index])
        device_outputs = [worker.alloc_tensor(tuple(value.shape), value.dtype) for value in outputs]
        allocated.extend(device_outputs)
        for transfer in ("individual", "packed"):
            weights = DraftWeights(head, "stream", transfer)
            strategies[transfer] = weights
            weights.start(worker)
            for layer in range(64):
                weights.for_layer(layer)

        # 交错计时相同的 64 层真实参数上传，不在计时区间内做回读或校验。
        copies = [0]
        original = worker.copy_to

        def counted(*a, **kw):
            copies[0] += 1
            return original(*a, **kw)

        worker.copy_to = counted
        for repeat in range(3):
            order = ("individual", "packed") if repeat % 2 == 0 else ("packed", "individual")
            for transfer in order:
                copies[0] = 0
                start = time.monotonic()
                for layer in range(64):
                    strategies[transfer].for_layer(layer)
                report["cases"].append(dict(transfer=transfer, repeat=repeat,
                    upload_loop_seconds=time.monotonic() - start, copy_calls=copies[0]))
        worker.copy_to = original

        # 每层整段回读，核对全部参数及布局间隙，覆盖复用父缓冲后的生命周期。
        packed = strategies["packed"]
        for layer in range(64):
            packed.for_layer(layer)
            span, parent, _ = packed.packed[layer]
            destination = readback[:span.numel()]
            worker.copy_from(destination.data_ptr(), parent.data_ptr, destination.numel())
            if not torch.equal(destination, span):
                raise AssertionError(f"packed upload changed layer {layer} bytes")
        report["byte_exact_layers"] = 64

        # 复用原有 full_prepare，验证真实 Q/K/V 投影与混合精度 norm 的设备视图。
        expected = None
        for transfer, weights in strategies.items():
            values = weights.for_layer(3)
            worker.run(program, constants[0], constants[1], *values[:5], constants[7], constants[8],
                       *device_outputs, 3, config=config)
            for host, device in zip(outputs, device_outputs, strict=True):
                worker.copy_from(host.data_ptr(), device.data_ptr, host.numel() * host.element_size())
                if not host.isfinite().all():
                    raise AssertionError("Full Attention preparation returned nonfinite values")
            if expected is None:
                expected = [value.clone() for value in outputs]
            elif not all(torch.equal(old, new) for old, new in zip(expected, outputs, strict=True)):
                raise AssertionError("packed views changed Full Attention preparation")
        report["full_prepare_bitwise_equal"] = True
        report["median_upload_seconds"] = {name: statistics.median(
            row["upload_loop_seconds"] for row in report["cases"] if row["transfer"] == name)
            for name in strategies}
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            try:
                for weights in strategies.values():
                    weights.close()
                if worker is not None:
                    for value in reversed(allocated):
                        worker.free_tensor(value)
            finally:
                if worker is not None:
                    worker.close()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
