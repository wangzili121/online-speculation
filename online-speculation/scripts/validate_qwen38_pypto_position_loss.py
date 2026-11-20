#!/usr/bin/env python3
"""完整词表的动态行权重损失：PyPTO 前向/手写梯度对照 Torch autograd。"""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocabulary", type=int, default=248320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.training_programs import make_tv, make_kl
    from blockspec.pypto_qwen38.verification_weights import rejection_weights

    torch.set_num_threads(4)
    torch.manual_seed(95)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    report = dict(passed=False, clean_shutdown=False, vocabulary=args.vocabulary, cases=[], timings=[])
    worker, handles = None, []
    try:
        student = torch.randn(32, args.vocabulary, dtype=torch.bfloat16).requires_grad_()
        teacher = torch.randn_like(student)
        weights = torch.ones(32, 8)
        gradient, losses = torch.zeros_like(student), torch.zeros(32, 8)
        programs = {}
        for objective, factory in [("tv", make_tv), ("kl", make_kl)]:
            for weighted in [False, True]:
                tensors = [student, teacher] + ([weights] if weighted else []) + [gradient, losses]
                programs[objective, weighted] = factory(32, args.vocabulary, position_weighted=weighted).compile(
                    *[torch.empty_like(t, device="meta") for t in tensors], 7, 63, config=config)
        worker = DistributedWorker(list(programs.values()), config=config)
        def alloc(tensor):
            handle = worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor.detach())
            handles.append(handle)
            return handle
        ds, dt, dw, dg, dl = map(alloc, [student, teacher, weights, gradient, losses])
        def read(handle, target):
            worker.copy_from(target.data_ptr(), handle.data_ptr, target.nbytes)
            return target
        for (objective, weighted), program in programs.items():
            for valid, rejection, tied in [(1, 0, False), (7, 2, False), (17, None, False), (31, 0, False), (7, 2, True)]:
                row_weights, total = rejection_weights(valid, rejection)
                if not weighted:
                    row_weights, total = [1] * valid + [0] * (32 - valid), valid
                weights.copy_(torch.tensor(row_weights)[:, None].expand(-1, 8))
                worker.copy_to(dw.data_ptr, weights.data_ptr(), weights.nbytes)
                teacher_value = student.detach() if tied else teacher
                worker.copy_to(dt.data_ptr, teacher_value.data_ptr(), teacher_value.nbytes)
                normalizer = total * 3  # 同一批多个 feedback block 共用归一化。
                call = [ds, dt] + ([dw] if weighted else []) + [dg, dl]
                worker.run(program, *call, valid, normalizer, config=config)
                log_q, log_p = student.float().log_softmax(-1), teacher_value.float().log_softmax(-1)
                per_row = (.5 * (log_q.exp() - log_p.exp()).abs().sum(-1) if objective == "tv" else
                           (log_p.exp() * (log_p - log_q)).sum(-1))
                expected = (per_row * weights[:, 0]).sum() / normalizer
                expected_gradient, = torch.autograd.grad(expected, student)
                actual, actual_loss = read(dg, gradient), read(dl, losses)
                torch.testing.assert_close(actual, expected_gradient, atol=2e-7, rtol=.03)
                error_norm = (actual.float() - expected_gradient.float()).norm()
                reference_norm = expected_gradient.float().norm()
                # p=q 时理论梯度为零；autograd 的舍入残差不适合作相对误差分母。
                relative = None if tied else float(error_norm / reference_norm.clamp_min(1e-20))
                if tied:
                    assert float(actual.float().norm()) < 1e-10
                else:
                    assert relative < .02
                torch.testing.assert_close(actual_loss[:, 0].sum(), expected.detach(), atol=2e-5, rtol=.002)
                assert torch.count_nonzero(actual[valid:]) == 0
                assert torch.equal(actual_loss, actual_loss[:, :1].expand(-1, 8))
                report["cases"].append(dict(objective=objective, weighted=weighted, valid=valid,
                                            rejection=rejection, tied=tied, relative_gradient_error=relative,
                                            gradient_error_norm=float(error_norm), reference_gradient_norm=float(reference_norm),
                                            loss=float(actual_loss[:, 0].sum())))
            # 仅为独立核的同步调用耗时，不代表模型全流吞吐。
            samples = []
            for _ in range(5):
                started = time.monotonic_ns()
                worker.run(program, *call, valid, normalizer, config=config)
                samples.append(time.monotonic_ns() - started)
            report["timings"].append(dict(objective=objective, weighted=weighted, synchronous_call_ns=samples))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
            report["clean_shutdown"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
