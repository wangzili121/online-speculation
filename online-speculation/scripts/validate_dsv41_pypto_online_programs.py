#!/usr/bin/env python3
"""Validate reusable DSV4.1 online loss, linear-gradient and Adam PyPTO kernels."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocabulary", type=int, default=129280)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.vocabulary < 64 or args.vocabulary % 64:
        parser.error("fresh output and a 64-aligned vocabulary required")

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_online_programs import make_adam, make_kl, make_linear, make_tv

    torch.set_num_threads(4)
    torch.manual_seed(41)
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    vocabulary = args.vocabulary
    programs = {
        "tv": make_tv(32, vocabulary).compile(
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "kl": make_kl(32, vocabulary).compile(
            empty((32, vocabulary)), empty((32, vocabulary)), empty((32, vocabulary)),
            empty((32, 8), torch.float32), 7, 7, config=config,
        ),
        "adam": make_adam((64, 128)).compile(
            *[empty((64, 128), torch.float32) for _ in range(7)],
            1e-4, .1, .001, 1., config=config,
        ),
        "linear": make_linear((32, 128), (64, 128)).compile(
            empty((32, 128)), empty((64, 128)), empty((32, 64)), config=config,
        ),
    }
    report = {
        "format": "orthrus-dsv41-pypto-online-programs-v1",
        "passed": False,
        "compiled": sorted(programs),
        "executed": False,
        "vocabulary": vocabulary,
        "cases": [],
    }
    if args.compile_only:
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    worker, handles = None, []
    try:
        worker = DistributedWorker(list(programs.values()), config=config)

        def allocate(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.detach())
            handles.append(handle)
            return handle

        def read(handle, target):
            worker.copy_from(target.data_ptr(), handle.data_ptr, target.nbytes)
            return target

        student = torch.randn(32, vocabulary, dtype=torch.bfloat16).requires_grad_()
        teacher = torch.randn_like(student)
        student_handle, teacher_handle = allocate(student), allocate(teacher)
        gradient_handle = allocate(torch.zeros_like(student))
        losses_handle = allocate(torch.zeros(32, 8, dtype=torch.float32))
        for objective in ("tv", "kl"):
            for valid in (1, 7, 31):
                worker.run(
                    programs[objective], student_handle, teacher_handle,
                    gradient_handle, losses_handle, valid, valid, config=config,
                )
                student_log = student[:valid].float().log_softmax(-1)
                teacher_log = teacher[:valid].float().log_softmax(-1)
                expected = (
                    .5 * (student_log.exp() - teacher_log.exp()).abs().sum(-1).mean()
                    if objective == "tv"
                    else (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()
                )
                expected_gradient, = torch.autograd.grad(expected, student, retain_graph=True)
                actual = read(gradient_handle, torch.empty_like(student))
                losses = read(losses_handle, torch.empty(32, 8, dtype=torch.float32))
                torch.testing.assert_close(actual, expected_gradient, atol=2e-5, rtol=.03)
                torch.testing.assert_close(losses[:, 0].sum(), expected.detach(), atol=2e-5, rtol=.002)
                assert torch.count_nonzero(actual[valid:]) == 0
                report["cases"].append({"operator": objective, "valid": valid, "loss": float(expected)})

        left, weight = torch.randn(32, 128).bfloat16(), torch.randn(64, 128).bfloat16()
        left_handle, weight_handle = allocate(left), allocate(weight)
        output_handle = allocate(torch.zeros(32, 64, dtype=torch.bfloat16))
        worker.run(programs["linear"], left_handle, weight_handle, output_handle, config=config)
        torch.testing.assert_close(
            read(output_handle, torch.empty(32, 64, dtype=torch.bfloat16)),
            left @ weight.T, atol=.125, rtol=.015,
        )
        report["cases"].append({"operator": "linear", "shape": [32, 128, 64]})

        master = torch.nn.Parameter(torch.randn(64, 128))
        optimizer = torch.optim.AdamW([master], lr=1e-4, weight_decay=0, foreach=False)
        state = [allocate(master.detach()), allocate(torch.zeros_like(master)), allocate(torch.zeros_like(master))]
        next_state = [allocate(torch.zeros_like(master)) for _ in range(3)]
        gradient = allocate(torch.zeros_like(master))
        for step in range(1, 5):
            value = torch.randn_like(master) * .02
            master.grad = value
            worker.copy_to(gradient.data_ptr, value.data_ptr(), value.nbytes)
            worker.run(
                programs["adam"], state[0], gradient, state[1], state[2], *next_state,
                1e-4, 1 - .9**step, 1 - .999**step, 1., config=config,
            )
            optimizer.step()
            expected = (master.detach(), optimizer.state[master]["exp_avg"], optimizer.state[master]["exp_avg_sq"])
            for reference, actual in zip(expected, next_state, strict=True):
                torch.testing.assert_close(read(actual, torch.empty_like(reference)), reference, atol=2e-6, rtol=.003)
            state, next_state = next_state, state
        report["cases"].append({"operator": "adam", "steps": 4})
        report["executed"] = report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
