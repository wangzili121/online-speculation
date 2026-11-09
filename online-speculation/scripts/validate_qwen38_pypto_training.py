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
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.training_reference import evaluate
    graph = LastLayerGraph(SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32))
    programs = TrainingPrograms(graph, config)
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
        for past, width, valid in ((0, 32, 31), (7, 17, 4), (32, 32, 1)):
            expected = evaluate(graph, inputs, past=past, width=width, valid=valid)
            actual = workspace.run(device, past=past, width=width, valid=valid)
            # 每个阶段都核对；不能只看最终 loss 而漏掉错误梯度。
            errors = []
            for node in graph.nodes:
                for value in node.outputs:
                    tensor = expected[value.name].detach()
                    returned = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
                    handle = workspace.values[value.name]
                    worker.copy_from(returned.data_ptr(), handle.data_ptr, returned.numel() * returned.element_size())
                    error = float((returned.float()-tensor.float()).norm() / tensor.float().norm().clamp_min(1e-20))
                    if not torch.isfinite(returned).all() or error > .03:
                        raise AssertionError((node.kind, value.name, past, width, valid, error))
                    errors.append(error)
            assert len(actual) == 6
            report["cases"].append(dict(past=past, width=width, valid=valid,
                                        checked_tensors=len(errors), max_relative_l2=max(errors), passed=True))
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


def validate_optimizer(args, config):
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.online_optimizer import DeviceOptimizer, OptimizerPrograms
    from blockspec.pypto_qwen38.weights import DraftWeights
    graph = LastLayerGraph(SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32))
    programs = OptimizerPrograms(graph, config, clip_norm=.1)
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
        state.initialize(Head(), {name: raw[name] for name in order if "gamma" in name})
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
            norm = torch.nn.utils.clip_grad_norm_(list(reference.values()), .1)
            result = state.step(weights, 1e-3)
            opt.step()
            torch.testing.assert_close(torch.tensor(result["gradient_norm"]), norm, rtol=1e-4, atol=1e-6)
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
            report["cases"].append(dict(name="clipped_adam_publish", step=step, **result, passed=True))
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
