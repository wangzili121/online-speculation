#!/usr/bin/env python3
"""独立 Torch autograd/Adam 对照，验证 PyPTO 的训练基础计算。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vocabulary", type=int, default=1024)
    p.add_argument("--suffix", action="store_true", help="validate the complete last-layer forward/backward")
    p.add_argument("--optimizer", action="store_true", help="validate clipping, Adam states and atomic publication")
    p.add_argument("--combined", action="store_true", help="compile the suffix as one CHIP entry")
    p.add_argument("--feedback", action="store_true", help="validate private paged history and teacher capture")
    p.add_argument("--accumulate-gradients", action="store_true", help="write VJP directly into external replay gradients")
    p.add_argument("--device-init", action="store_true", help="initialize masters from existing device head banks")
    p.add_argument("--tensor-views", action="store_true", help="exercise shared frozen-weight Tensor views")
    p.add_argument("--bridge", action="store_true", help="validate isolated workers and cross-device byte copies")
    p.add_argument("--learner-device", type=int)
    p.add_argument("--real-shape", action="store_true", help="exercise all six real 27B parameter dimensions")
    a = p.parse_args()
    if a.output.exists():
        p.error("new output required")
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.training_programs import make_tv, make_adam, make_linear
    torch.set_num_threads(4)
    torch.manual_seed(41)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    if a.bridge:
        if a.learner_device is None or a.learner_device == a.device:
            p.error("bridge validation needs two different visible devices")
        return bridge(a, config)
    if a.feedback:
        return feedback(a, config)
    if a.optimizer:
        return validate_optimizer(a, config)
    if a.suffix:
        return suffix(a, config)
    V = a.vocabulary
    tv = make_tv(32, V).compile(empty((32, V)), empty((32, V)), empty((32, V)),
                                 empty((32, 8), torch.float32), 31, 31, config=config)
    adam = make_adam((64, 128)).compile(*[empty((64, 128), torch.float32) for _ in range(7)],
                                      .0001, .1, .001, 1., config=config)
    linear = make_linear((32, 128), (64, 128)).compile(empty((32, 128)), empty((64, 128)), empty((32, 64)), config=config)
    report, worker, handles = dict(passed=False, runtime=runtime_identity(pypto), cases=[]), None, []
    try:
        worker = DistributedWorker([tv, adam, linear], config=config)

        def alloc(x):
            h = worker.alloc_tensor(tuple(x.shape), x.dtype, init=x)
            handles.append(h)
            return h

        def read(h):
            x = torch.empty(tuple(h.shape), dtype=h.dtype)
            worker.copy_from(x.data_ptr(), h.data_ptr, x.numel() * x.element_size())
            return x

        student = torch.randn(32, V, dtype=torch.bfloat16).requires_grad_()
        teacher = torch.randn_like(student)
        ds, dt = alloc(student.detach()), alloc(teacher)
        dg, losses = alloc(torch.zeros_like(student)), alloc(torch.zeros(32, 8))
        for valid in (1, 17, 31):
            worker.run(tv, ds, dt, dg, losses, valid, valid, config=config)
            expected = .5 * (student[:valid].float().softmax(-1) - teacher[:valid].float().softmax(-1)).abs().sum(-1).mean()
            grad, = torch.autograd.grad(expected, student)
            actual, loss = read(dg), read(losses)[:, 0].sum()
            torch.testing.assert_close(actual, grad, atol=2e-5, rtol=.03)
            assert float((actual.float()-grad.float()).norm()/grad.float().norm()) < .02
            torch.testing.assert_close(loss, expected.detach(), atol=2e-5, rtol=.002)
            assert torch.count_nonzero(actual[valid:]) == 0
            report["cases"].append(dict(name="tv", valid=valid, loss=float(loss), passed=True))
        x, w = torch.randn(32, 128).bfloat16(), torch.randn(64, 128).bfloat16()
        dx, dw, out = alloc(x), alloc(w), alloc(torch.zeros(32, 64).bfloat16())
        worker.run(linear, dx, dw, out, config=config)
        torch.testing.assert_close(read(out), x @ w.T, atol=.125, rtol=.015)
        report["cases"].append(dict(name="linear", passed=True))
        master = torch.nn.Parameter(torch.randn(64, 128))
        optimizer = torch.optim.AdamW([master], lr=.0001, weight_decay=0, foreach=False)
        state = [alloc(master.detach()), alloc(torch.zeros_like(master)), alloc(torch.zeros_like(master))]
        next_state = [alloc(torch.zeros_like(master)) for _ in range(3)]
        grad = alloc(torch.zeros_like(master))
        for step in range(1, 5):
            g = torch.randn_like(master) * .02
            master.grad = g
            worker.copy_to(grad.data_ptr, g.data_ptr(), g.numel() * 4)
            worker.run(adam, state[0], grad, state[1], state[2], *next_state,
                       .0001, 1-.9**step, 1-.999**step, 1., config=config)
            optimizer.step()
            for expected, actual in zip((master.detach(), optimizer.state[master]["exp_avg"],
                                         optimizer.state[master]["exp_avg_sq"]), next_state, strict=True):
                torch.testing.assert_close(read(actual), expected, atol=2e-6, rtol=.003)
            state, next_state = next_state, state
        report["cases"].append(dict(name="adam_four_steps", passed=True))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            for h in reversed(handles):
                worker.free_tensor(h)
            worker.close()
        a.output.write_text(json.dumps(report, indent=2) + "\n")


def suffix(args, config):
    import time
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.training_reference import evaluate
    graph = LastLayerGraph() if args.real_shape else LastLayerGraph(SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32))
    programs = TrainingPrograms(graph, config, combined=args.combined, accumulate_gradients=args.accumulate_gradients)
    worker = workspace = None
    handles, report = [], dict(passed=False, cases=[])
    try:
        worker = DistributedWorker(list(programs.programs.values()), config=config)
        workspace = TrainingWorkspace(programs, worker)
        inputs, device = {}, {}
        for name, value in graph.inputs.items():
            x = torch.randn(value.shape) * (.03 if "weight" in name or "mlp" in name or name == "lm_head" else .2)
            x = x.float() + 1 if value.dtype == "fp32" else x.bfloat16()
            inputs[name] = x
            device[name] = worker.alloc_tensor(value.shape, x.dtype, init=x)
            handles.append(device[name])
            if args.tensor_views:
                from pypto.runtime.task_interface import DataType
                dtype = DataType.FLOAT32 if x.dtype == torch.float32 else DataType.BFLOAT16
                device[name] = device[name].buffer.tensor(value.shape, dtype)
        accumulated, destination = {}, None
        if args.accumulate_gradients:
            destination = {}
            for name, value in graph.gradients.items():
                zero = torch.zeros(value.shape)
                destination[name] = worker.alloc_tensor(value.shape, torch.float32, init=zero)
                handles.append(destination[name])
                accumulated[value.name] = zero
        cases = ((32, 32, 17),) if args.real_shape else ((0, 32, 31), (7, 17, 4), (32, 32, 1))
        report["scope"] = "real_shape_finiteness" if args.real_shape else "torch_numerical_comparison"
        for past, width, valid in cases:
            expected = None if args.real_shape else evaluate(graph, inputs, past=past, width=width, valid=valid)
            if destination is not None and expected is not None:
                for value in graph.gradients.values():
                    accumulated[value.name] = accumulated[value.name] + expected[value.name]
                    expected[value.name] = accumulated[value.name]
            start = time.monotonic()
            actual = workspace.run(device, past=past, width=width, valid=valid, gradients=destination)
            seconds = time.monotonic()-start
            # 每个阶段都核对；不能只看最终 loss 而漏掉错误梯度。
            errors = []
            for node in graph.nodes:
                for value in node.outputs:
                    tensor = None if expected is None else expected[value.name].detach()
                    returned = torch.empty(value.shape, dtype=torch.bfloat16 if value.dtype == "bf16" else torch.float32)
                    handle = workspace.values[value.name]
                    worker.copy_from(returned.data_ptr(), handle.data_ptr, returned.numel() * returned.element_size())
                    error = 0. if tensor is None else float((returned.float()-tensor.float()).norm() / tensor.float().norm().clamp_min(1e-20))
                    if not torch.isfinite(returned).all() or error > .03:
                        raise AssertionError((node.kind, value.name, past, width, valid, error))
                    if tensor is None and value.name in {v.name for v in graph.gradients.values()}:
                        assert torch.count_nonzero(returned), value.name
                    errors.append(error)
            assert len(actual) == 6
            report["cases"].append(dict(past=past, width=width, valid=valid,
                checked_tensors=len(errors), execution_seconds=seconds,
                max_relative_l2=None if expected is None else max(errors), passed=True))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if workspace is not None:
            workspace.close()
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def bridge(args, config):
    from dataclasses import replace
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.online_feedback_programs import make_loss_sum
    from blockspec.pypto_qwen38.online_bridge import TensorBridge
    workers, report = [], dict(passed=False, cases=[])
    try:
        for device in (args.device, args.learner_device):
            cfg = replace(config, distributed_config=replace(config.distributed_config, device_ids=[device]))
            program = make_loss_sum().compile(torch.empty((32, 8), device="meta"),
                                              torch.empty((1, 8), device="meta"), config=cfg)
            worker = DistributedWorker([program], config=cfg)
            workers.append((worker, program, cfg, []))
        for i, (worker, program, cfg, owned) in enumerate(workers):
            x = worker.alloc_tensor((32, 8), torch.float32, init=torch.full((32, 8), float(i+1)))
            owned.append(x)
            y = worker.alloc_tensor((1, 8), torch.float32)
            owned.append(y)
            worker.run(program, x, y, config=cfg)
            actual = torch.empty(1, 8)
            worker.copy_from(actual.data_ptr(), y.data_ptr, 32)
            torch.testing.assert_close(actual, torch.full((1, 8), 32.*(i+1)), atol=0, rtol=0)
        transfer = TensorBridge()
        a, b = workers
        transfer.copy(a[0], a[3][0], b[0], b[3][0])
        b[0].run(b[1], b[3][0], b[3][1], config=b[2])
        b[0].copy_from(actual.data_ptr(), b[3][1].data_ptr, 32)
        torch.testing.assert_close(actual, torch.full((1, 8), 32.), atol=0, rtol=0)
        tail = a[0].alloc_tensor((1, 8), torch.float32)
        a[3].append(tail)
        transfer.copy(b[0], b[3][0], a[0], tail, source_offset=31*8*4)
        a[0].copy_from(actual.data_ptr(), tail.data_ptr, 32)
        torch.testing.assert_close(actual, torch.ones(1, 8), atol=0, rtol=0)
        report.update(passed=True, devices=[args.device, args.learner_device], copied_bytes=transfer.bytes,
            cases=[dict(name=name, passed=True) for name in ("isolated_compute", "forward_copy", "reverse_offset_copy")])
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        for worker, _, _, owned in reversed(workers):
            for handle in reversed(owned):
                worker.free_tensor(handle)
            worker.close()
        args.output.write_text(json.dumps(report, indent=2)+"\n")


def feedback(args, config):
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.online_feedback_programs import make_history, make_teacher, make_loss_sum
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    shape = (16, 4, 37, 16, 256)
    history = make_history(37).compile(empty(shape), empty(shape), empty((32,), torch.int32),
        empty((2048, 256)), empty((2048, 256)), 0, config=config)
    teacher = make_teacher().compile(empty((8, 16, 248320)), empty((32, 248320)), 0, 0, 16, config=config)
    loss = make_loss_sum().compile(empty((32, 8), torch.float32), empty((1, 8), torch.float32), config=config)
    worker, handles, report = None, [], dict(passed=False, cases=[])
    try:
        worker = DistributedWorker([history, teacher, loss], config=config)
        def alloc(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)
            handles.append(handle)
            return handle
        def read(handle):
            value = torch.empty(tuple(handle.shape), dtype=handle.dtype)
            worker.copy_from(value.data_ptr(), handle.data_ptr, value.numel()*value.element_size())
            return value
        k, v = torch.randn(shape).bfloat16(), torch.randn(shape).bfloat16()
        table = torch.randperm(37)[:32].to(torch.int32)
        dk, dv, dt = alloc(k), alloc(v), alloc(table)
        out = [alloc(torch.empty(2048, 256).bfloat16()) for _ in range(2)]
        for past in (0, 1, 17, 255, 480):
            worker.run(history, dk, dv, dt, *out, past, config=config)
            for source, handle in zip((k, v), out, strict=True):
                expected = torch.zeros(4, 512, 256).bfloat16()
                for row in range(past):
                    expected[:, row] = source[15, :, int(table[row//16]), row%16]
                torch.testing.assert_close(read(handle).reshape(4, 512, 256), expected, atol=0, rtol=0)
            report["cases"].append(dict(name="private_history", past=past, passed=True))
        torch.testing.assert_close(read(dk), k, atol=0, rtol=0)
        torch.testing.assert_close(read(dv), v, atol=0, rtol=0)
        source = torch.randn(8, 16, 248320).bfloat16()
        output = torch.full((32, 248320), float("nan"), dtype=torch.bfloat16)
        ds, do = alloc(source), alloc(output)
        worker.run(teacher, ds, do, 48, 0, 16, config=config)
        worker.run(teacher, ds, do, 112, 16, 1, config=config)
        actual = read(do)
        torch.testing.assert_close(actual[:16], source[3], atol=0, rtol=0)
        torch.testing.assert_close(actual[16], source[7, 0], atol=0, rtol=0)
        assert torch.isnan(actual[17:]).all()
        values = torch.randn(32, 1).expand(32, 8).contiguous()
        summed = alloc(torch.empty(1, 8))
        worker.run(loss, alloc(values), summed, config=config)
        torch.testing.assert_close(read(summed), values.sum(0, keepdim=True), atol=1e-5, rtol=1e-5)
        report["cases"].append(dict(name="teacher_slot_offset_and_loss_summary", passed=True))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
        args.output.write_text(json.dumps(report, indent=2)+"\n")


def validate_optimizer(args, config):
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.online_optimizer import DeviceOptimizer, OptimizerPrograms
    from blockspec.pypto_qwen38.weights import DraftWeights
    graph = LastLayerGraph() if args.real_shape else LastLayerGraph(SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32))
    programs = OptimizerPrograms(graph, config, clip_norm=.1, combined=args.combined)
    order = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
    raw, banks = {}, {}
    for name in order:
        raw[name] = (torch.randn(programs.shapes[name]) * .02).bfloat16()
        banks[name] = raw[name].float() + 1 if "gamma" in name else raw[name]

    class Head:
        @staticmethod
        def banks(layer):
            assert layer == 63
            return tuple(banks[name] for name in order)

    worker = state = weights = restored = restored_weights = None
    handles, report = [], dict(passed=False, cases=[])
    try:
        worker = DistributedWorker(programs.programs(), config=config)
        state = DeviceOptimizer(programs, worker)
        device_banks = None
        if args.device_init:
            device_banks = [worker.alloc_tensor(tuple(banks[name].shape), banks[name].dtype, init=banks[name]) for name in order]
            handles += device_banks
        state.initialize(Head(), {name: raw[name] for name in order if "gamma" in name}, device_banks=device_banks)
        weights = DraftWeights(Head())
        weights.worker = worker
        reference = {name: torch.nn.Parameter(value.float()) for name, value in raw.items()}
        opt = torch.optim.AdamW(list(reference.values()), lr=1e-3, weight_decay=0, foreach=False)

        def read(handle):
            value = torch.empty(tuple(handle.shape), dtype=handle.dtype)
            worker.copy_from(value.data_ptr(), handle.data_ptr, value.numel()*value.element_size())
            return value

        gradients = {name: worker.alloc_tensor(shape, torch.float32) for name, shape in programs.shapes.items()}
        handles += list(gradients.values())
        for step in range(1, 5):
            state.zero_grad()
            for name, parameter in reference.items():
                parameter.grad = torch.randn_like(parameter) * .03
                g = parameter.grad
                worker.copy_to(gradients[name].data_ptr, g.data_ptr(), g.numel()*4)
            state.accumulate(gradients)
            # 大矩阵的顺序 FP32 norm 会积累误差；使用独立 FP64 范数核对设备归约。
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(parameter.grad, dtype=torch.float64) for parameter in reference.values()]))
            clip = min(1., .1/(float(norm)+1e-6))
            for parameter in reference.values():
                parameter.grad.mul_(clip)
            result = state.step(weights, 1e-3)
            opt.step()
            torch.testing.assert_close(torch.tensor(result["gradient_norm"]), norm.float(), rtol=1e-4, atol=1e-6)
            for name, parameter in reference.items():
                for actual, expected in zip(state.state[name], (parameter, opt.state[parameter]["exp_avg"],
                                                                opt.state[parameter]["exp_avg_sq"]), strict=True):
                    torch.testing.assert_close(read(actual), expected.detach(), atol=2e-6, rtol=.003)
                # Adam 的 FP32 近似误差已独立核对；发布必须精确转换实际设备 master。
                expected = read(state.state[name][0]).bfloat16()
                if "gamma" in name:
                    expected = expected.float() + 1
                torch.testing.assert_close(read(state.tensors()[name]), expected, atol=0, rtol=0)
            assert weights.version == step and state.steps == step
            assert tuple(weights.overrides[63]) == tuple(state.tensors()[name] for name in order)
            report["cases"].append(dict(name="clipped_adam_publish", step=step,
                reference_fp64_norm=float(norm), **result, passed=True))
        before = {name: read(tensors[0]).clone() for name, tensors in state.state.items()}
        old_banks = weights.overrides[63]
        state.zero_grad()
        broken = torch.full(programs.shapes["q_gamma"], float("nan"))
        worker.copy_to(state.accum["q_gamma"].data_ptr, broken.data_ptr(), broken.numel()*4)
        try:
            state.step(weights, 1e-3)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("nonfinite update published")
        assert weights.version == 4 and state.steps == 4 and weights.overrides[63] is old_banks
        for name, tensors in state.state.items():
            torch.testing.assert_close(read(tensors[0]), before[name], atol=0, rtol=0)
        report["cases"].append(dict(name="nonfinite_transaction_aborted", passed=True))
        from blockspec.pypto_qwen38.online_checkpoint import save, restore
        checkpoint = args.output.with_suffix(".checkpoint")
        identity = dict(test="independent_optimizer_fixture")
        save(state, checkpoint, identity)
        restored = DeviceOptimizer(programs, worker)
        restored_weights = DraftWeights(Head())
        restored_weights.worker = worker
        restore(restored, restored_weights, checkpoint, identity)
        assert restored.steps == 4 and restored_weights.version == 4
        for name in programs.names:
            for actual, expected in zip(restored.state[name], state.state[name], strict=True):
                torch.testing.assert_close(read(actual), read(expected), atol=0, rtol=0)
            torch.testing.assert_close(read(restored.tensors()[name]), read(state.tensors()[name]), atol=0, rtol=0)
        for item in (state, restored):
            item.zero_grad()
            item.accumulate(gradients)
        state.step(weights, 1e-3)
        restored.step(restored_weights, 1e-3)
        assert weights.version == restored_weights.version == 5
        for name in programs.names:
            for actual, expected in zip(restored.state[name], state.state[name], strict=True):
                torch.testing.assert_close(read(actual), read(expected), atol=0, rtol=0)
        report["cases"].append(dict(name="checkpoint_resume_next_step_exact", passed=True))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if restored_weights is not None:
            restored_weights.close()
        if restored is not None:
            restored.close()
        if weights is not None:
            weights.close()
        if state is not None:
            state.close()
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
