#!/usr/bin/env python3
"""Validate real rank-local heads and native row projections on two NPUs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if args.output.exists() or len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0 or args.repeats < 2:
        parser.error("require two assigned devices, 2+ repeats and a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import pypto
    from safetensors import safe_open
    from pypto import ir
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ReadOnlyHostTensor, RunConfig, StackedDeviceTensor
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.weights import DraftWeights
    from blockspec.pypto_qwen38.tp_programs import make_output_sum, make_residual, make_row_projection

    report = dict(passed=False, scope="two_device_head_residency_and_row_projection_components",
                  full_model_loaded=False, full_tp_inference=False, npu_execution=False, cases=[])
    worker, heads, weights, handles = None, [], [], []
    try:
        torch.set_num_threads(4)
        report["runtime"] = runtime_identity(pypto)
        native_module(args.source, "weight").verify_model_identity(args.model)
        config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
            device_ids=devices, num_sub_workers=0), ring_heap=(0, 0, 0, 4 << 30))
        meta = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        rows = meta((2, 32, 5120))
        programs = {name: make_row_projection(args.source, 2, width).compile(
            meta((2, 32, width)), meta((2, 5120, width)), rows, config=config)
            for name, width in (("attention_o", 3072), ("mlp_down", 8704))}
        programs["sum"] = ir.compile(make_output_sum(2, iterations=3), platform="a2a3",
                                     distributed_config=config.distributed_config)
        programs["residual"] = make_residual(args.source, 2).compile(rows, rows, rows, config=config)
        report["compiled_programs"] = list(programs)
        report["sum_shape"] = [32, 5120]
        report["sum_iterations_per_dispatch"] = 3
        if args.compile_only:
            report.update(passed=True, scope="tp_components_compile_only")
            return
        for rank in range(2):
            head = HeadSource(args.model, args.head_dir, runtime_tp=2, runtime_rank=rank)
            head.prepare()
            heads.append(head)
            print(json.dumps(dict(head_prepared_rank=rank, bytes=head.shared_weights.numel())), flush=True)
        report["rank_head_bytes"] = [head.shared_weights.numel() for head in heads]
        inputs = {name: torch.zeros((2, 32, width), dtype=torch.bfloat16).share_memory_()
                  for name, width in (("attention_o", 3072), ("mlp_down", 8704))}
        residual = torch.stack([(torch.arange(32 * 5120).reshape(32, 5120) % 17 / 16).bfloat16()] * 2).share_memory_()
        partial, summed, result = [torch.empty((2, 32, 5120), dtype=torch.bfloat16).share_memory_() for _ in range(3)]
        scratch = torch.empty(32 << 20, dtype=torch.uint8).share_memory_()
        key = "model.language_model.layers.3.mlp.down_proj.weight"
        filename = Path(heads[0].index[key])
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError("checkpoint shard names must be local basenames")
        with safe_open(args.model / filename, framework="pt", device="cpu") as file:
            mlp = [file.get_slice(key)[:, rank * 8704:(rank + 1) * 8704].clone().share_memory_() for rank in range(2)]
        inherited = [ReadOnlyHostTensor(head.shared_weights) for head in heads]
        inherited += [ReadOnlyHostTensor(value) for value in mlp]
        inherited += [*inputs.values(), residual, partial, summed, result, scratch]
        worker = DistributedWorker(list(programs.values()), config=config, inherited_host_tensors=inherited)
        report["npu_execution"] = True

        def allocate(value, rank, *, initialize=False):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype,
                                         init=value if initialize else None, worker_id=rank)
            handles.append((rank, handle))
            return handle

        def stack(values):
            return StackedDeviceTensor(tuple(values), (2, *values[0].shape), (0, 1))

        original_copy, uploads = worker.copy_to, [0, 0]

        def counted(*a, **kw):
            uploads[kw.get("worker_id", 0)] += 1
            return original_copy(*a, **kw)

        worker.copy_to = counted
        for rank, head in enumerate(heads):
            value = DraftWeights(head, "resident", worker_id=rank)
            weights.append(value)
            value.start(worker)
        report["cold_head_upload_calls"] = list(uploads)
        report["committed_bytes_after_head"] = [worker.committed_device_memory(rank) for rank in range(2)]
        try:
            worker.free_tensor(weights[1].for_layer(0)[0], worker_id=0)
        except ValueError:
            report["wrong_worker_free_rejected"] = True
        else:
            raise AssertionError("freeing a head through the wrong worker was accepted")

        # 回读两卡的全部 64 层参数，核对 owner 和数据；不用源 head 当回读目的地。
        for rank, (head, owned) in enumerate(zip(heads, weights, strict=True)):
            for layer in range(64):
                for handle, expected in zip(owned.for_layer(layer), head.banks(layer), strict=True):
                    raw = expected.view(torch.uint8).reshape(-1)
                    for offset in range(0, raw.numel(), scratch.numel()):
                        count = min(scratch.numel(), raw.numel() - offset)
                        worker.copy_from(scratch.data_ptr(), handle.data_ptr, count, src_offset=offset, worker_id=rank)
                        if not torch.equal(scratch[:count], raw[offset:offset + count]):
                            raise AssertionError(f"rank {rank} layer {layer}: resident head differs")
        report["byte_exact_head_layers_per_rank"] = 64
        source_weights = {"attention_o": [head.banks(3)[-1] for head in heads], "mlp_down": mlp}
        device_weights = {"attention_o": stack([value.for_layer(3)[-1] for value in weights]),
                          "mlp_down": stack([allocate(value, rank, initialize=True) for rank, value in enumerate(mlp)])}
        device_inputs = {name: stack([allocate(value[rank], rank) for rank in range(2)]) for name, value in inputs.items()}
        device_partial, device_sum, device_result, device_residual = [stack([
            allocate(value[rank], rank, initialize=value is residual) for rank in range(2)])
            for value in (partial, summed, result, residual)]
        row_ids = torch.arange(32)
        for name, host in inputs.items():
            width = host.shape[-1]
            for repeat in range(args.repeats):
                before = list(uploads)
                for owned in weights:
                    for layer in range(64):
                        owned.for_layer(layer)
                if uploads != before:
                    raise AssertionError("resident head was uploaded again during a round")
                host.zero_()
                expected_partial = []
                for rank in range(2):
                    host[rank, row_ids, row_ids] = repeat + 1
                    host[rank, row_ids, width - 1 - row_ids] = -(rank + 1) / 2
                    expected_partial.append((source_weights[name][rank][:, row_ids].T.float() * (repeat + 1)
                        - source_weights[name][rank][:, width - 1 - row_ids].T.float() * ((rank + 1) / 2)).bfloat16())
                    handle = device_inputs[name].shards[rank]
                    worker.copy_to(handle.data_ptr, host[rank].data_ptr(), host[rank].numel() * 2, worker_id=rank)
                worker.run(programs[name], device_inputs[name], device_weights[name], device_partial, config=config)
                worker.run(programs["sum"], device_partial, device_sum, config=config)
                worker.run(programs["residual"], device_residual, device_sum, device_result, config=config)
                for host_value, device_value in ((partial, device_partial), (summed, device_sum), (result, device_result)):
                    for rank, handle in enumerate(device_value.shards):
                        worker.copy_from(host_value[rank].data_ptr(), handle.data_ptr, host_value[rank].numel() * 2, worker_id=rank)
                if not torch.equal(partial, torch.stack(expected_partial)):
                    raise AssertionError(f"{name}: native shard projection differs from sparse exact reference")
                expected_sum = partial.float().sum(0).bfloat16()
                expected_result = (residual[0].float() + expected_sum.float()).bfloat16()
                if not all(torch.equal(summed[rank], expected_sum) and torch.equal(result[rank], expected_result) for rank in range(2)):
                    raise AssertionError(f"{name}: Sum or reduce-before-residual differs")
                row = dict(projection=name, repeat=repeat, native_projection_exact=True,
                           sum_exact=True, residual_once_exact=True, hot_head_upload_calls=0)
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        failures = []
        for owned in reversed(weights):
            try:
                owned.close()
            except BaseException as error:
                failures.append(error)
        if worker is not None:
            for rank, handle in reversed(handles):
                try:
                    worker.free_tensor(handle, worker_id=rank)
                except BaseException as error:
                    failures.append(error)
            try:
                worker.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            report.update(passed=False, cleanup_errors=[repr(error) for error in failures])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if failures:
            raise RuntimeError("TP component cleanup failed") from failures[0]


if __name__ == "__main__":
    main()
