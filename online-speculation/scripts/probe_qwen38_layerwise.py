#!/usr/bin/env python3
"""Compare native decode, native K32 verification and the mapped AR view."""

import json
import os
import hashlib
from functools import partial
from pathlib import Path

import torch

from qwen38_validation_worker import ValidationWorker, ar_verifier_inputs
from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def differences(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"boundary shapes differ: {actual.shape} != {expected.shape}")
    actual, expected = actual.float(), expected.float()
    error = actual - expected
    return dict(equal=torch.equal(actual, expected), max_abs=float(error.abs().max()),
                relative_l2=float(error.norm() / expected.norm().clamp_min(1e-12)))


def first_row(value):
    return value.detach().reshape(-1, value.shape[-1])[0].cpu().clone()


class LayerwiseWorker(ValidationWorker):
    def orthrus_set_norm_diagnostic(self, mode):
        from blockspec.vllm_ascend.qwen38_weights import SharedRMSNorm

        if mode not in {"original", "gamma_bf16", "both_bf16", "native", "native_residual", "native_residual_fused"}:
            raise ValueError("unknown diagnostic norm mode")
        model = self.model_runner.drafter.model
        if not hasattr(self, "_norm_diagnostic_originals"):
            self._norm_diagnostic_originals = {module: module.forward for module in model.modules()
                                               if isinstance(module, SharedRMSNorm)}
        fused = ({model.norm, *(layer.post_norm for layer in model.layers),
                  *(layer.input_norm for layer in model.layers[1:])} if mode.startswith("native_residual") else set())
        for module, original in self._norm_diagnostic_originals.items():
            if mode == "original" or mode.startswith("native_residual") and module not in fused:
                module.forward = original
                continue

            def normalized(value, norm=module):
                gamma = 1 + norm.weight
                if mode == "native":
                    import torch_npu
                    return torch_npu.npu_rms_norm(value, gamma, norm.eps)[0]
                if mode == "native_residual_fused":
                    import torch_npu
                    return torch_npu.npu_rms_norm(value, torch.ones_like(norm.weight), norm.eps)[0] * gamma
                out = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.eps)
                if mode in {"both_bf16", "native_residual"}:
                    out = out.to(value.dtype).float()
                return (out * gamma.float()).to(value.dtype)

            module.forward = normalized
        return mode

    def orthrus_diagnostic_candidates(self, mode):
        if mode not in {"head", "fixed", "shifted"}:
            raise ValueError("unknown candidate diagnostic mode")
        draft = self.model_runner.drafter
        if not hasattr(self, "_unshifted_propose"):
            self._unshifted_propose = draft.propose_batch
        if mode == "head":
            draft.propose_batch = self._unshifted_propose
            return mode

        def shifted(*args, **kwargs):
            result = self._unshifted_propose(*args, **kwargs)
            if draft.pending is not None and result[0]:
                # 固定 verifier 输入，避免把 head 改变候选造成的差异算到 norm 上。
                values = self._layerwise_fixed_candidates.clone()
                if mode == "shifted":
                    values = (values + 37) % draft.model.config.vocab_size
                draft.pending["candidates"], draft.pending["q"] = values, None
                return [values[:len(result[0])].cpu().tolist()]
            return result

        draft.propose_batch = shifted
        return mode

    def orthrus_enable_layerwise(self, history):
        runner, draft = self.model_runner, self.model_runner.drafter
        native = getattr(runner.get_model(), "language_model", runner.get_model()).model
        self._layerwise = {}
        self._layerwise_history = history
        self._layerwise_label = None
        self._layerwise_active = False
        self._layerwise_handles = []

        def native_input(index, module, args, kwargs):
            if index == 0:
                boundary = self._logit_boundary
                self._layerwise_active = (self._layerwise_label is not None and boundary is not None
                    and boundary[0][0] - boundary[1][0] == history
                    and "native" not in self._layerwise[self._layerwise_label])
                if self._layerwise_active:
                    self._layerwise[self._layerwise_label]["native"] = {}
                    self._layerwise[self._layerwise_label]["query_rows"] = boundary[1][0]
            if self._layerwise_active:
                hidden, residual = kwargs["hidden_states"], kwargs.get("residual")
                save_native(f"{index}.input", hidden if residual is None else hidden + residual)

        def save_native(name, value):
            if self._layerwise_active:
                self._layerwise[self._layerwise_label]["native"][name] = (
                    value.detach().reshape(-1, value.shape[-1]).cpu().clone())

        def native_output(index, module, args, output):
            if self._layerwise_active:
                hidden, residual = output
                save_native(f"{index}.output", hidden if residual is None else hidden + residual)
            if index == len(native.layers) - 1:
                self._layerwise_active = False

        def norm_output(name, module, args, output):
            save_native(name, output[0] if isinstance(output, tuple) else output)
            if self._layerwise_active and name.endswith("_norm"):
                # 相同原生输入，分别控制 gamma 与归一化结果的 BF16 舍入。
                value = args[0] if len(args) == 1 or args[1] is None else args[0] + args[1]
                normalized = value.float() * torch.rsqrt(
                    value.float().square().mean(-1, keepdim=True) + module.variance_epsilon)
                gamma = 1 + module.weight.float()
                rounded_gamma = (1 + module.weight).float()
                variants = dict(hf=normalized * gamma, gamma_bf16=normalized * rounded_gamma,
                    normalized_bf16=normalized.to(value.dtype).float() * gamma,
                    both_bf16=normalized.to(value.dtype).float() * rounded_gamma)
                expected = output[0] if isinstance(output, tuple) else output
                self._layerwise[self._layerwise_label].setdefault("norm_replay", {})[name] = {
                    key: differences(actual.to(value.dtype), expected) for key, actual in variants.items()}

        for index, layer in enumerate(native.layers):
            self._layerwise_handles.append(layer.register_forward_pre_hook(
                partial(native_input, index), with_kwargs=True))
            self._layerwise_handles.append(layer.register_forward_hook(partial(native_output, index)))
            for name, module in (("input_norm", layer.input_layernorm),
                                 ("post_norm", layer.post_attention_layernorm), ("mlp", layer.mlp)):
                self._layerwise_handles.append(module.register_forward_hook(
                    partial(norm_output, f"{index}.{name}")))

        observe = draft.observe_verified

        def teacher(logits, metadata):
            if not draft._verification_count(metadata):
                return None
            with torch.inference_mode(False), torch.no_grad():
                return logits[metadata.target_logits_indices].detach().clone()

        def checked(metadata, target, sampled):
            label, pending = self._layerwise_label, draft.pending
            if label is not None and pending is not None and "mapped" not in self._layerwise[label]:
                row = self._layerwise[label]
                row["mapped"] = {}
                if label == "k32":
                    self._layerwise_fixed_candidates = pending["candidates"].detach().clone()
                row["target_logits"] = target.detach().float().cpu().clone()

                def mapped_output(name, module, args, output):
                    row["mapped"][name] = output.detach().reshape(-1, output.shape[-1]).cpu().clone()

                def mapped_input(name, module, args):
                    value = args[0]
                    row["mapped"][name] = value.detach().reshape(-1, value.shape[-1]).cpu().clone()

                handles = []
                for index, layer in enumerate(draft.model.layers):
                    handles.append(layer.register_forward_pre_hook(partial(mapped_input, f"{index}.input")))
                    handles.append(layer.register_forward_hook(partial(mapped_output, f"{index}.output")))
                    for name in ("input_norm", "post_norm", "mlp"):
                        handles.append(getattr(layer, name).register_forward_hook(
                            partial(mapped_output, f"{index}.{name}")))
                try:
                    with torch.inference_mode(False), torch.no_grad():
                        count = metadata.num_draft_tokens[0]
                        tokens = ar_verifier_inputs(pending, count)
                        actual = draft.model(tokens, view="ar", cache=pending["cache"]).logits[0, :count].float()
                        expected = target[:count].float()
                        tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
                        row["logits"] = dict(max_tv=float(tv.max()), first_tv=float(tv[0]),
                                             row_tv=tv.cpu().tolist(), rows=count, input_rows=tokens.shape[1],
                                             **differences(actual, expected))
                finally:
                    for handle in handles:
                        handle.remove()
            return observe(metadata, target, sampled)

        draft.teacher_rows, draft.observe_verified = teacher, checked
        return "enabled"

    def orthrus_arm_layerwise(self, label):
        self._layerwise_label = label
        self._layerwise[label] = {}
        return label

    def orthrus_layerwise_report(self):
        baseline = self._layerwise["decode"]["native"]
        result = {}
        for label, row in self._layerwise.items():
            native = row["native"]
            expected_rows = {"decode": 1, "k2": 2, "k16": 16}.get(label, 32)
            if row["query_rows"] != expected_rows:
                raise AssertionError(f"{label} scheduled {row['query_rows']} rows, expected {expected_rows}")
            result[label] = dict(query_rows=row["query_rows"],
                native_vs_decode={name: differences(value[0], baseline[name][0]) for name, value in native.items()},
                norm_replay=row.get("norm_replay", {}))
            if "k32" in self._layerwise:
                result[label]["native_vs_k32"] = {name: differences(value[0],
                    self._layerwise["k32"]["native"][name][0]) for name, value in native.items()}
                if "target_logits" in row and row["query_rows"] == 32:
                    reference = self._layerwise["k32"]["target_logits"]
                    result[label]["target_logits_vs_k32"] = differences(row["target_logits"], reference)
            if "mapped" in row:
                result[label]["mapped_vs_native"] = {
                    name: dict(**differences(value, native[name]), rows={str(index):
                        differences(value[index], native[name][index]) for index in (0, 15, 16, 26)
                        if index < value.shape[0]}) for name, value in row["mapped"].items()}
                result[label]["logits"] = row["logits"]
        return result


def main():
    prepare_runtime_environment()
    from vllm import LLM, SamplingParams

    args = arguments()
    if args.online or args.block_size != 32:
        raise ValueError("layerwise diagnostics require a fixed K32 head")
    options = native_options(args) | speculative_options(args)
    options["worker_extension_cls"] = "probe_qwen38_layerwise.LayerwiseWorker"
    llm = LLM(**options)
    prompt = 'Return JSON with keys "name" and "age":'
    history = len(llm.get_tokenizer().encode(prompt))
    llm.collective_rpc("orthrus_enable_logits_audit", args=(0,))
    llm.collective_rpc("orthrus_enable_layerwise", args=(history,))
    llm.collective_rpc("orthrus_enable_cache_audit")
    from blockspec.vllm_ascend import qwen38_weights
    report = dict(diagnostic_only=True, model=args.model, history=history, tp=args.tp, outputs=[],
        source_sha256={str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("qwen38_validation_worker.py"), Path(qwen38_weights.__file__))})
    matrix = [("decode", 0), ("k2", 1), ("k16", 15), ("k32", 31)]
    if os.environ.get("ORTHRUS_LAYERWISE_VARIANTS") == "1":
        matrix.extend([(mode, 31) for mode in ("gamma_bf16", "both_bf16", "native",
                                             "native_residual", "native_residual_fused", "future_shifted")])
    for label, limit in matrix:
        llm.collective_rpc("orthrus_set_proposal_limit", args=(limit,))
        llm.collective_rpc("orthrus_set_norm_diagnostic", args=(label if label in {
            "gamma_bf16", "both_bf16", "native", "native_residual", "native_residual_fused"} else "original",))
        candidate_mode = ("shifted" if label == "future_shifted" else "fixed" if label in {
            "gamma_bf16", "both_bf16", "native", "native_residual", "native_residual_fused"} else "head")
        llm.collective_rpc("orthrus_diagnostic_candidates", args=(candidate_mode,))
        llm.collective_rpc("orthrus_arm_layerwise", args=(label,))
        output = llm.generate([prompt], SamplingParams(temperature=0, seed=7,
            max_tokens=2, ignore_eos=True), use_tqdm=False)[0].outputs[0]
        report["outputs"].append(dict(label=label, candidate_mode=candidate_mode,
                                      tokens=list(output.token_ids), text=output.text))
        print(json.dumps(report["outputs"][-1]), flush=True)
    report["workers"] = llm.collective_rpc("orthrus_layerwise_report")
    report["cache_audit"] = llm.collective_rpc("orthrus_stats")
    Path(os.environ["ORTHRUS_LAYERWISE_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"layerwise_output": os.environ["ORTHRUS_LAYERWISE_OUTPUT"]}), flush=True)


if __name__ == "__main__":
    main()
