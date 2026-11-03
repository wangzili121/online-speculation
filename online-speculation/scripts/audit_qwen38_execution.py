#!/usr/bin/env python3
"""Compare existing GDN, gradient reduction and CPU AdamW execution paths."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def measure(operation, repeats, synchronize=lambda: None):
    for _ in range(3):
        operation()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def adamw_audit(gradients, repeats, synchronize=lambda: None):
    # 同一初值核对连续五步的参数和两个 moment，再比较既有后端。
    expected, report = None, []
    for backend, options in (("single", dict(foreach=False)), ("foreach", dict(foreach=True)),
                             ("fused", dict(fused=True))):
        parameters = [torch.nn.Parameter(torch.zeros_like(g)) for g in gradients]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient
        values, optimizer = None, None
        try:
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0, **options)
            for _ in range(5):
                optimizer.step()
            values = [(p.detach().clone(), optimizer.state[p]["exp_avg"].clone(),
                       optimizer.state[p]["exp_avg_sq"].clone()) for p in parameters]
            if expected is None:
                expected = values
            numerical_error = None
            try:
                for actual_row, expected_row in zip(values, expected, strict=True):
                    for a, b in zip(actual_row, expected_row, strict=True):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)
            except AssertionError as error:
                numerical_error = str(error)
            report.append(dict(backend=backend, numerical_pass=numerical_error is None,
                               numerical_error=numerical_error,
                               timing=measure(optimizer.step, repeats, synchronize)))
        except (RuntimeError, NotImplementedError) as error:
            report.append(dict(backend=backend, error=str(error)))
        del optimizer, parameters, values
    return report


def cpu_audit(repeats):
    # 27B TP2 的投影矩阵，并混入小型 norm；不代表整步训练耗时。
    gradients = [torch.randn(shape) for shape in ((5120, 5120), (5120, 5120), (128,), (24,))]

    def original_norm():
        total = torch.zeros(())
        for gradient in gradients:
            total += gradient.square().sum()
        return total.sqrt()

    def reduced_norm():
        return torch.stack([torch.linalg.vector_norm(g, dim=-1).square().sum() for g in gradients]).sum().sqrt()

    reference = original_norm()
    actual = reduced_norm()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    return dict(norm=dict(original=measure(original_norm, repeats), reduced=measure(reduced_norm, repeats),
                          reference=float(reference), actual=float(actual)), adamw=adamw_audit(gradients, repeats))


def original_slices(operation, **kwargs):
    state, indices = kwargs["state"], kwargs["ssm_state_indices"]
    initial = state.index_select(0, indices[kwargs["num_accepted_tokens"].long() - 1].long()).clone()
    outputs = []
    for start in range(0, kwargs["query"].shape[0], 16):
        end = min(start + 16, kwargs["query"].shape[0])
        slots = indices[start:end].contiguous()
        state.index_copy_(0, slots[:1].long(), initial)
        arguments = {name: kwargs[name][start:end].contiguous() for name in ("query", "key", "value", "g", "beta")}
        outputs.append(operation(**arguments, state=state, scale=kwargs["scale"],
            actual_seq_lengths=kwargs["actual_seq_lengths"].new_tensor([0, end - start]),
            ssm_state_indices=slots, num_accepted_tokens=None))
        initial = state.index_select(0, slots[-1:].long()).clone()
    return torch.cat(outputs, dim=0)


def npu_audit(model, repeats):
    import torch_npu
    from vllm_ascend.utils import enable_custom_op
    from transformers import AutoConfig
    from blockspec.vllm_ascend.qwen38_gdn_backend import recurrent_slices

    if not enable_custom_op():
        raise RuntimeError("installed recurrent GDN operator required")
    cfg = AutoConfig.from_pretrained(model, local_files_only=True).text_config
    nk, nv = cfg.linear_num_key_heads // 2, cfg.linear_num_value_heads // 2
    dim = cfg.linear_key_head_dim
    operation = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    operation = getattr(operation, "__wrapped__", operation)
    report = dict(torch_npu=str(torch_npu.__version__), heads=[nk, nv], dim=dim, cases=[])
    for length, accepted in ((1, 32), (2, 27), (8, 1), (17, 16), (32, 31), (32, 32)):
        q, k = [torch.nn.functional.normalize(torch.randn(length, nk, dim, device="npu",
                                                        dtype=torch.bfloat16), dim=-1) for _ in range(2)]
        original = torch.randn(40, nv, dim, dim, device="npu") * .01
        state = original.clone()
        arguments = dict(query=q, key=k, value=torch.randn(length, nv, dim, device="npu", dtype=torch.bfloat16),
            g=-torch.rand(length, nv, device="npu"), beta=torch.rand(length, nv, device="npu", dtype=torch.bfloat16),
            state=state, ssm_state_indices=torch.randperm(40, device="npu")[:32].int(),
            actual_seq_lengths=torch.tensor([0, length], device="npu", dtype=torch.int32),
            num_accepted_tokens=torch.tensor([accepted], device="npu", dtype=torch.int32), scale=dim ** -.5)

        def run(function, state=state, original=original, arguments=arguments):
            state.copy_(original)
            return function(operation, **arguments)

        expected = run(original_slices).clone()
        expected_state = state.clone()
        actual = run(recurrent_slices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        row = dict(length=length, accepted=accepted, output_exact=True, all_states_exact=True,
                   original=measure(lambda: run(original_slices), repeats, torch.npu.synchronize),
                   reduced=measure(lambda: run(recurrent_slices), repeats, torch.npu.synchronize))
        report["cases"].append(row)
    # 只给首行初态，其余 checkpoint 填 NaN，验证算子确实先写后读。
    report["scratch"] = []
    for length in (1, 8, 17, 32):
        short = arguments | dict(query=q[:length], key=k[:length], value=arguments["value"][:length],
            g=arguments["g"][:length], beta=arguments["beta"][:length],
            ssm_state_indices=torch.arange(length, device="npu", dtype=torch.int32),
            actual_seq_lengths=torch.tensor([0, length], device="npu", dtype=torch.int32),
            num_accepted_tokens=torch.ones(1, device="npu", dtype=torch.int32))
        clean = torch.zeros(length, nv, dim, dim, device="npu")
        poisoned = torch.full_like(clean, float("nan"))
        clean[:1].copy_(original[:1])
        poisoned[:1].copy_(original[:1])
        expected = recurrent_slices(operation, **(short | dict(state=clean)))
        actual = recurrent_slices(operation, **(short | dict(state=poisoned)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(poisoned, clean, rtol=0, atol=0)
        report["scratch"].append(dict(length=length, poisoned_scratch_exact=True))
    from blockspec.vllm_ascend.qwen38_ops import delta_rule
    report["draft"] = []
    for length in (1, 8, 17, 32):
        query = q[None, :length].repeat_interleave(nv // nk, dim=2)
        key = k[None, :length].repeat_interleave(nv // nk, dim=2)
        inputs = dict(g=arguments["g"][None, :length], beta=arguments["beta"][None, :length],
                      initial_state=original[:1].transpose(-1, -2).contiguous(), output_final_state=True)

        def old_draft(query=query, key=key, length=length, inputs=inputs):
            state = torch.zeros((length, nv, dim, dim), device="npu")
            state[:1] = inputs["initial_state"].transpose(-1, -2).float()
            output = original_slices(operation, state=state,
                query=query[0, :, ::nv // nk].contiguous(), key=key[0, :, ::nv // nk].contiguous(),
                value=arguments["value"][:length].contiguous(), g=inputs["g"][0].float().contiguous(),
                beta=inputs["beta"][0].contiguous(), scale=dim ** -.5,
                actual_seq_lengths=torch.tensor([0, length], device="npu", dtype=torch.int32),
                ssm_state_indices=torch.arange(length, device="npu", dtype=torch.int32),
                num_accepted_tokens=torch.ones(1, device="npu", dtype=torch.int32))
            return output[None], state[-1:].transpose(-1, -2).contiguous()

        def new_draft(query=query, key=key, length=length, inputs=inputs):
            return delta_rule(query, key, arguments["value"][None, :length], groups=nv // nk,
                              fallback=None, **inputs)

        for actual, expected in zip(new_draft(), old_draft(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report["draft"].append(dict(length=length, output_and_final_exact=True,
            original=measure(old_draft, repeats, torch.npu.synchronize),
            reduced=measure(new_draft, repeats, torch.npu.synchronize)))
    with torch.inference_mode(False):
        gradients = [torch.randn(shape, device="npu") for shape in ((5120, 5120), (5120, 5120), (128,), (24,))]
        report["adamw"] = adamw_audit(gradients, repeats, torch.npu.synchronize)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="local 27B model config; enables the NPU audit")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    torch.manual_seed(731)
    report = dict(torch=str(torch.__version__), threads=torch.get_num_threads(),
                  cpu=cpu_audit(args.repeats))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.model:
        with torch.inference_mode():
            report["npu"] = npu_audit(args.model, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
