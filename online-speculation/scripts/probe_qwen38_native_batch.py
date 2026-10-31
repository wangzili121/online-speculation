#!/usr/bin/env python3
"""Isolate native GEMM row-count effects with single-request attention and caches."""

import hashlib
import json
import os
from pathlib import Path

import torch

from blockspec.vllm_ascend.qwen38_proposer import OrthrusWorkerExtension
from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment


def logit_summary(row):
    row = row.detach().float().cpu().contiguous()
    if not torch.isfinite(row).all():
        raise FloatingPointError("native control produced nonfinite logits")
    values, indices = row.topk(4)
    return dict(argmax=int(row.argmax()), tokens=indices.tolist(), values=values.tolist(),
                sha256=hashlib.sha256(row.numpy().tobytes()).hexdigest())


def linear_width_call(operation, value, width, *args, **kwargs):
    if not width or value.ndim != 2 or value.shape[0] != 1:
        return operation(value, *args, **kwargs)
    result = operation(value.expand(width, -1).contiguous(), *args, **kwargs)
    if isinstance(result, tuple):
        return (result[0][:1], *result[1:])
    return result[:1]


class NativeBatchWorker(OrthrusWorkerExtension):
    def native_linear_width(self, width):
        """Only GEMMs see duplicate rows; attention and all cache metadata stay batch-one."""
        from vllm.model_executor.layers.linear import LinearBase

        if width not in (0, 2, 8, 16, 32):
            raise ValueError("native linear control width must be 0, 2, 8, 16 or 32")
        target = self.model_runner.get_model()
        text = getattr(target, "language_model", target)
        if not hasattr(self, "_native_linears"):
            self._native_linears = [(module, module.forward) for module in text.model.modules()
                                    if isinstance(module, LinearBase)]
            self._native_logits = text.compute_logits
        for module, original in self._native_linears:
            def forward(value, *args, _original=original, **kwargs):
                return linear_width_call(_original, value, width, *args, **kwargs)
            module.forward = forward if width else original

        def logits(hidden_states):
            batch = self.model_runner.input_batch
            decode = batch.num_computed_tokens_cpu[0] >= batch.num_prompt_tokens[0]
            return linear_width_call(self._native_logits, hidden_states, width if decode else 0)

        text.compute_logits = logits if width else self._native_logits
        return dict(width=width, linear_modules=len(self._native_linears))

    def native_batch_audit(self):
        runner = self.model_runner
        status = dict(speculative_config=runner.speculative_config is not None,
                      drafter=type(runner.drafter).__name__ if runner.drafter is not None else None,
                      orthrus_runner_hooks=bool(getattr(type(runner), "_qwen38_orthrus_hooks", False)))
        from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
        status["orthrus_attention_hooks"] = bool(getattr(
            AscendAttentionMetadataBuilder, "_qwen38_query_limit_hook", False))
        status["orthrus_recurrent_hook"] = bool(getattr(
            torch.ops._C_ascend.npu_recurrent_gated_delta_rule, "_qwen38_recurrent_hook", False) is True)
        if status["drafter"] is not None or any(value for key, value in status.items() if key != "drafter"):
            raise AssertionError(f"native control unexpectedly enabled speculation: {status}")
        self._native_batch_status = status
        self._native_batch_rows = {}
        sample = runner._sample

        def checked(logits, metadata):
            if metadata is not None or logits.shape[0] != runner.input_batch.num_reqs:
                raise AssertionError("native batch control requires one logit row per request")
            captured = []
            for index, request in enumerate(runner.input_batch.req_ids[:logits.shape[0]]):
                records = self._native_batch_rows.setdefault(request, [])
                if len(records) < 2:
                    row = logit_summary(logits[index])
                    row.update(batch_rows=logits.shape[0],
                               computed=int(runner.input_batch.num_computed_tokens_cpu[index]))
                    records.append(row)
                    captured.append((index, row))
            result = sample(logits, metadata)
            for index, row in captured:
                row["sampled"] = result.sampled_token_ids[index].cpu().tolist()
            return result

        runner._sample = checked
        return status

    def native_batch_take(self):
        rows, self._native_batch_rows = self._native_batch_rows, {}
        return dict(status=self._native_batch_status, rows=rows)


def main():
    prepare_runtime_environment()
    from vllm import LLM, SamplingParams

    args = arguments()
    if args.online or args.head_dir:
        raise ValueError("native control must not load a draft head or online learner")
    widths = [int(width) for width in os.environ.get("ORTHRUS_NATIVE_LINEAR_WIDTHS", "0,2,8,16,32,0").split(",")]
    if not widths or widths[0] != 0 or any(width not in (0, 2, 8, 16, 32) for width in widths):
        raise ValueError("linear widths must begin with unchanged execution (0) and fit 0,2,8,16,32")
    options = native_options(args)
    options["worker_extension_cls"] = "probe_qwen38_native_batch.NativeBatchWorker"
    llm = LLM(**options)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    status = llm.collective_rpc("native_batch_audit")
    destination = Path(os.environ["ORTHRUS_BATCH_OUTPUT"])
    report = dict(diagnostic_only=True, completed=False, model=args.model, tp=args.tp,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  max_num_seqs=1, max_model_len=args.max_model_len,
                  gdn_prefill_backend=args.gdn_prefill_backend, native_status=status, outputs=[])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    prompt = 'Return JSON with keys "name" and "age":'
    parameters = SamplingParams(temperature=0, seed=7, max_tokens=16, ignore_eos=True)
    for width in widths:
        control = llm.collective_rpc("native_linear_width", args=(width,))
        results = llm.generate([prompt], parameters, use_tqdm=False)
        traces = llm.collective_rpc("native_batch_take")
        tokens = [list(result.outputs[0].token_ids) for result in results]
        if len(tokens) != 1 or len(tokens[0]) != 16:
            raise AssertionError("native batch did not finish all requests")
        baseline = report["outputs"][0]["tokens"][0] if report["outputs"] else tokens[0]
        row = dict(batch_size=1, linear_width=width, linear_control=control, tokens=tokens, traces=traces,
                   matches_batch_one=[row == baseline for row in tokens])
        report["outputs"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(batch_size=1, linear_width=width, first_tokens=tokens[0][:4],
                              matching=sum(row["matches_batch_one"]))), flush=True)
    report["completed"] = True
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
