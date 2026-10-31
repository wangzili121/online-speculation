#!/usr/bin/env python3
"""Replay mapped modules on the native verifier's identical operands and history."""

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from probe_qwen38_layerwise import LayerwiseWorker, differences
from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


def compare(actual, expected):
    actual = actual.detach().reshape(-1, actual.shape[-1])
    expected = expected.detach().reshape(-1, expected.shape[-1])
    return dict(**differences(actual, expected), flattened_rows={str(i): differences(actual[i], expected[i])
        for i in (0, 15, 16, 26, 31) if i < actual.shape[0]})


def expand_query_heads(query, value):
    if value.shape[-2] % query.shape[-2]:
        raise ValueError("value heads must be a multiple of query heads")
    return query.repeat_interleave(value.shape[-2] // query.shape[-2], dim=-2)


def native_cached_conv(x, weight, bias, activation, seq_idx=None):
    # 仅用于 K32 缓存续接的无梯度诊断；状态完全自有，不引用主模型 cache。
    if bias is not None or activation != "silu" or seq_idx is not None or weight.shape[-1] != 4 or x.shape[-1] != 36:
        raise ValueError("native convolution diagnostic requires width=4 and cached K32")
    inputs = x[0, :, 4:].T.contiguous()
    output = torch.empty_like(inputs)
    state = x.new_zeros((1, 34, x.shape[1]))
    state[0, :3] = x[0, :, 1:4].T
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output, inputs, weight.T, conv_state=state, bias_opt=None,
        query_start_loc_opt=torch.tensor([0, 32], dtype=torch.int32, device=x.device),
        cache_indices_opt=torch.zeros((1, 32), dtype=torch.int32, device=x.device),
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=torch.ones(1, dtype=torch.int32, device=x.device),
        activation_mode=1, pad_slot_id=-1, run_mode=1)
    return torch.cat((x[..., :4], output.T[None]), dim=-1)


class ModuleReplayWorker(LayerwiseWorker):
    def orthrus_projection_diagnostic(self, mode):
        if mode not in {"original", "packed_full", "packed_all", "packed_all_gates"}:
            raise ValueError("unknown packed projection diagnostic")
        model = self.model_runner.drafter.model
        native = getattr(self.model_runner.get_model(), "language_model", self.model_runner.get_model()).model
        if not hasattr(self, "_projection_originals"):
            self._projection_originals = []
            for layer in model.layers:
                mixer = layer.attention.ar
                names = (("q_proj", "k_proj", "v_proj") if layer.kind == "full_attention"
                         else ("in_proj_qkv", "in_proj_z"))
                self._projection_originals.extend((getattr(mixer, name), "forward", getattr(mixer, name).forward)
                                                  for name in names)
                if layer.kind == "linear_attention":
                    self._projection_originals.append((mixer, "chunk_gated_delta_rule", mixer.chunk_gated_delta_rule))
            self._projection_handles = []
        for module, name, original in self._projection_originals:
            setattr(module, name, original)
        for handle in self._projection_handles:
            handle.remove()
        self._projection_handles.clear()
        if mode == "original":
            return mode
        for index, layer in enumerate(model.layers):
            mixer = layer.attention.ar
            source = native.layers[index]
            if layer.kind == "full_attention":
                names, weight = ("q_proj", "k_proj", "v_proj"), source.self_attn.qkv_proj.weight
            elif mode != "packed_full":
                names, weight = ("in_proj_qkv", "in_proj_z"), source.linear_attn.in_proj_qkvz.weight
            else:
                continue
            sizes = [getattr(mixer, name).weight.shape[0] for name in names]
            for part, name in enumerate(names):
                def projected(value, packed_weight=weight, splits=sizes, position=part):
                    # 仅用于 AR 对照：读取同一份冻结权重，保持原生 GEMM 的整块形状。
                    return F.linear(value[0], packed_weight).split(splits, dim=-1)[position][None]

                getattr(mixer, name).forward = projected
            if layer.kind == "linear_attention" and mode == "packed_all_gates":
                captured = {}

                def inputs(module, args, saved=captured):
                    saved["input"] = args[0]

                self._projection_handles.append(mixer.register_forward_pre_hook(inputs))
                original = mixer.chunk_gated_delta_rule

                def gated(q, k, v, operation=original, owner=mixer, saved=captured, **kwargs):
                    from vllm_ascend.device.device_op import DeviceOperator
                    value = saved.pop("input")
                    a, b = owner.in_proj_a(value)[0], owner.in_proj_b(value)[0]
                    kwargs["g"], kwargs["beta"] = DeviceOperator.fused_gdn_gating(owner.A_log, a, b, owner.dt_bias)
                    return operation(q, k, v, **kwargs)

                mixer.chunk_gated_delta_rule = gated
        return mode

    def orthrus_set_fixed_module_candidates(self, values):
        draft = self.model_runner.drafter
        if (len(values) != 31 or any(type(x) is not int or not 0 <= x < draft.model.config.vocab_size
                                    for x in values)):
            raise ValueError("31 valid fixed candidates required")
        self._layerwise_fixed_candidates = torch.tensor(values, device=draft.model.embedding.weight.device)
        self.orthrus_diagnostic_candidates("fixed")
        return values

    def orthrus_module_precision(self, mode):
        modes = {"original", "fp32_l2", "fp32_gate_norm", "fused_mlp", "fp32_conv", "combined",
                 "combined_no_conv", "combined_native_projection", "combined_native_conv", "combined_native_ops",
                 "combined_native_residual", "residual_only"}
        if mode not in modes:
            raise ValueError("unknown module precision diagnostic")
        model = self.model_runner.drafter.model
        native = getattr(self.model_runner.get_model(), "language_model", self.model_runner.get_model()).model
        if not hasattr(self, "_module_originals"):
            self._module_originals = [(model, "forward", model.forward)]
            for index, layer in enumerate(model.layers):
                self._module_originals.append((layer.mlp, "forward", layer.mlp.forward))
                if layer.kind == "linear_attention":
                    mixer = layer.attention.ar
                    self._module_originals.extend(((mixer, "chunk_gated_delta_rule", mixer.chunk_gated_delta_rule),
                        (mixer, "causal_conv1d_fn", mixer.causal_conv1d_fn),
                        (mixer.norm, "forward", mixer.norm.forward)))
                else:
                    self._module_originals.append((layer.attention.ar, "forward", layer.attention.ar.forward))
        for module, name, original in self._module_originals:
            setattr(module, name, original)
        for index, layer in enumerate(model.layers):
            combined = mode.startswith("combined")
            if mode == "fused_mlp" or combined:
                def mlp(value, weights=native.layers[index].mlp.gate_up_proj.weight, down=layer.mlp.down_proj):
                    import torch_npu
                    return down(torch_npu.npu_swiglu(F.linear(value, weights)))

                layer.mlp.forward = mlp
            if layer.kind != "linear_attention":
                if mode in {"combined_native_projection", "combined_native_ops", "combined_native_residual"}:
                    def full(value, position_embeddings, attention_mask, past_key_values,
                             source=native.layers[index].self_attn, mixer=layer.attention.ar):
                        length = value.shape[1]
                        past = self.model_runner.drafter.pending["cache"].length
                        positions = torch.arange(past, past + length, device=value.device)
                        packed = F.linear(value[0], source.qkv_proj.weight)
                        q, k, v, gate = source._project_qkv_gate(packed, positions)
                        q = q.reshape(1, length, source.num_heads, source.head_dim).transpose(1, 2)
                        k = k.reshape(1, length, source.num_kv_heads, source.head_dim).transpose(1, 2)
                        v = v.reshape(1, length, source.num_kv_heads, source.head_dim).transpose(1, 2)
                        k, v = past_key_values.update(k, v, mixer.layer_idx)
                        copies = source.num_heads // source.num_kv_heads
                        out = F.scaled_dot_product_attention(q, k.repeat_interleave(copies, dim=1),
                            v.repeat_interleave(copies, dim=1), attn_mask=attention_mask, scale=source.scaling)
                        out = out.transpose(1, 2).reshape(1, length, -1) * gate.sigmoid()[None]
                        return mixer.o_proj(out), None

                    layer.attention.ar.forward = full
                continue
            mixer = layer.attention.ar
            if mode == "fp32_l2" or combined:
                def chunk(q, k, v, operation=mixer.chunk_gated_delta_rule, **kwargs):
                    def normalize(value):
                        if mode in {"combined_native_ops", "combined_native_residual"}:
                            from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
                            return l2norm_fwd(value)
                        full = value.float()
                        return (full * torch.rsqrt(full.square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype)
                    if not kwargs.pop("use_qk_l2norm_in_kernel", False):
                        raise ValueError("expected Q/K normalization in diagnostic GDN call")
                    return operation(normalize(q), normalize(k), v, use_qk_l2norm_in_kernel=False, **kwargs)

                mixer.chunk_gated_delta_rule = chunk
            if mode == "fp32_gate_norm" or combined:
                def norm(value, gate, module=mixer.norm):
                    full = value.float()
                    normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + module.variance_epsilon)
                    return (normalized * module.weight.float() * F.silu(gate.float())).to(value.dtype)

                mixer.norm.forward = norm
            if mode == "fp32_conv" or combined and mode != "combined_no_conv":
                def conv(x, weight, bias, activation, seq_idx=None):
                    if bias is not None or activation != "silu" or seq_idx is not None:
                        raise ValueError("unsupported convolution diagnostic")
                    output = F.conv1d(x.float(), weight[:, None].float(),
                        padding=weight.shape[-1] - 1, groups=weight.shape[0])[..., :x.shape[-1]]
                    return F.silu(output).to(x.dtype)

                mixer.causal_conv1d_fn = conv
            if mode in {"combined_native_conv", "combined_native_ops", "combined_native_residual"}:
                mixer.causal_conv1d_fn = native_cached_conv
            if mode in {"combined_native_ops", "combined_native_residual"}:
                mixer.norm.forward = native.layers[index].linear_attn.norm.forward
        if mode in {"combined_native_residual", "residual_only"}:
            original = model.forward

            def fused_ar(tokens, **kwargs):
                if kwargs.get("view", "ar") != "ar":
                    return original(tokens, **kwargs)
                from blockspec.parallel.backbone import BackboneOutput
                cache = kwargs["cache"]
                if tokens.shape != (1, 32) or torch.is_grad_enabled():
                    raise ValueError("fused AR diagnostic requires no-grad cached K32")
                scratch = model._cache(cache)
                hidden, residual = model.embedding(tokens)[0], None
                positions = torch.arange(cache.length, cache.length + 32, device=tokens.device)[None]
                rotary = model.rotary(hidden, positions[None].expand(3, -1, -1))
                allowed = hidden.new_zeros((1, 1, 32, cache.length + 32))
                allowed.masked_fill_(torch.arange(cache.length + 32, device=tokens.device)[None]
                                     > positions[0, :, None], float("-inf"))
                for layer, source in zip(model.layers, native.layers, strict=True):
                    # 保留两个残差分量直到融合 norm，不预先生成已舍入的相加结果。
                    if residual is None:
                        residual = hidden
                        normalized = source.input_layernorm(hidden)
                    else:
                        normalized, residual = source.input_layernorm(hidden, residual)
                    mixer = layer.attention.ar
                    if layer.kind == "full_attention":
                        mixed, _ = mixer(normalized[None], position_embeddings=rotary,
                            attention_mask=allowed, past_key_values=scratch)
                    else:
                        mixed = mixer(normalized[None], cache_params=scratch, attention_mask=None)
                    normalized, residual = source.post_attention_layernorm(mixed[0], residual)
                    hidden = layer.mlp(normalized[None])[0]
                normalized, residual = native.norm(hidden, residual)
                return BackboneOutput(residual[None], None, model.head(normalized[None]), None)

            model.forward = fused_ar
        return mode

    def orthrus_enable_module_replay(self, history):
        self.orthrus_enable_layerwise(history)
        runner, draft = self.model_runner, self.model_runner.drafter
        native = getattr(runner.get_model(), "language_model", runner.get_model()).model
        self._module_inputs, self._module_results = {}, {}
        self._module_layer = None

        def capture(index, name, module, args, output):
            if self._layerwise_active and self._layerwise_label == "k32":
                value = output[0] if isinstance(output, tuple) else output
                self._module_inputs.setdefault(index, {})[name] = value.detach().clone()

        def active_layer(index, module, args, kwargs):
            self._module_layer = index

        def capture_core(index, module, args):
            if self._layerwise_active and self._layerwise_label == "k32":
                self._module_inputs.setdefault(index, {})["core"] = args[0].detach().clone()
                self._module_inputs[index]["z"] = args[1].detach().clone()

        def full_inputs(index, module, args):
            if self._layerwise_active and self._layerwise_label == "k32":
                self._module_inputs.setdefault(index, {}).update({name: value.detach().clone()
                    for name, value in zip(("full_query", "full_key", "full_value"), args, strict=True)})

        def norm_input(index, name, module, args):
            if self._layerwise_active and self._layerwise_label == "k32":
                self._module_inputs.setdefault(index, {})[name] = args[0].detach().clone()

        for index, layer in enumerate(native.layers):
            layer.register_forward_pre_hook(partial(active_layer, index), with_kwargs=True)
            for name, module in (("input_norm", layer.input_layernorm),
                                 ("post_norm", layer.post_attention_layernorm), ("mlp", layer.mlp)):
                module.register_forward_hook(partial(capture, index, name))
            if draft.model.layer_types[index] == "linear_attention":
                layer.linear_attn.norm.register_forward_pre_hook(partial(capture_core, index))
            else:
                layer.self_attn.attn.register_forward_pre_hook(partial(full_inputs, index))
                layer.self_attn.attn.register_forward_hook(partial(capture, index, "full_output"))
                layer.self_attn.qkv_proj.register_forward_hook(partial(capture, index, "full_packed"))
                for name in ("q_norm", "k_norm"):
                    module = getattr(layer.self_attn, name)
                    module.register_forward_pre_hook(partial(norm_input, index, name + "_input"))
                    module.register_forward_hook(partial(capture, index, name + "_output"))

        recurrent = torch.ops._C_ascend.npu_recurrent_gated_delta_rule

        def observed_recurrent(**kwargs):
            recording = self._layerwise_active and self._layerwise_label == "k32"
            if recording:
                row = self._module_inputs.setdefault(self._module_layer, {})
                row["recurrent"] = {name: kwargs[name].detach().clone()
                                    for name in ("query", "key", "value", "g", "beta")}
                accepted = kwargs.get("num_accepted_tokens")
                indices = kwargs["ssm_state_indices"].flatten()
                selected = indices[:1] if accepted is None else indices[accepted.long() - 1]
                row["recurrent"]["initial_state"] = kwargs["state"].index_select(
                    0, selected.long()).transpose(-1, -2).contiguous().clone()
            return recurrent(**kwargs)

        torch.ops._C_ascend.npu_recurrent_gated_delta_rule = observed_recurrent
        observe = draft.observe_verified

        def verified(metadata, target, sampled):
            if self._layerwise_label == "k32" and not self._module_results:
                with torch.inference_mode(False), torch.no_grad():
                    self._replay_modules(native, draft)
            return observe(metadata, target, sampled)

        draft.observe_verified = verified
        return "enabled"

    def _replay_modules(self, native, draft):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            l2norm, torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
        )
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        model, pending = draft.model, draft.pending
        scratch = model._cache(pending["cache"])
        length, past = 32, pending["cache"].length
        device = model.embedding.weight.device
        positions = torch.arange(past, past + length, device=device)[None]
        rotary = model.rotary(torch.empty((), device=device, dtype=torch.bfloat16),
                             positions[None].expand(3, -1, -1))
        allowed = torch.zeros((1, 1, length, past + length), device=device, dtype=torch.bfloat16)
        allowed.masked_fill_(torch.arange(past + length, device=device)[None] > positions[0, :, None],
                             float("-inf"))
        self._module_candidates = pending["candidates"].cpu().tolist()
        for index, layer in enumerate(model.layers):
            captured, source = self._module_inputs[index], native.layers[index]
            inputs = captured["input_norm"][None]
            row = dict(kind=layer.kind)
            # 每层都用原生输入重放，避免把前面层累积的误差误归因到当前模块。
            actual_mlp = layer.mlp(captured["post_norm"][None])[0]
            row["mlp_same_input"] = compare(actual_mlp, captured["mlp"])
            packed = source.mlp.gate_up_proj(captured["post_norm"])[0]
            gate, up = packed.chunk(2, dim=-1)
            actual_projection = (layer.mlp.gate_up_proj(captured["post_norm"])
                if hasattr(layer.mlp, "gate_up_proj") else torch.cat((
                    layer.mlp.gate_proj(captured["post_norm"]), layer.mlp.up_proj(captured["post_norm"])), dim=-1))
            row["mlp_packed_projection"] = compare(actual_projection, packed)
            row["mlp_activation"] = compare(F.silu(gate) * up, source.mlp.act_fn(packed))
            mixer = layer.attention.ar
            if layer.kind == "linear_attention":
                local = {}
                row["gdn_initial_state"] = compare(scratch.layers[index].recurrent_states[0],
                                                    captured["recurrent"]["initial_state"])

                def core_input(module, args, captured_local=local):
                    captured_local["core"], captured_local["z"] = args[0].detach().clone(), args[1].detach().clone()

                hook = mixer.norm.register_forward_pre_hook(core_input)
                chunk = mixer.chunk_gated_delta_rule

                def chunk_input(q, k, v, captured_local=local, operation=chunk, **kwargs):
                    captured_local.update(query=q.detach().clone(), key=k.detach().clone(), value=v.detach().clone(),
                                          g=kwargs["g"].detach().clone(), beta=kwargs["beta"].detach().clone())
                    return operation(q, k, v, **kwargs)

                mixer.chunk_gated_delta_rule = chunk_input
                try:
                    mixed = mixer(inputs, cache_params=scratch, attention_mask=None)
                finally:
                    hook.remove()
                    mixer.chunk_gated_delta_rule = chunk
                row["gdn_core_same_layer_input"] = compare(local["core"], captured["core"])
                row["gdn_z_projection"] = compare(local["z"], captured["z"])
                norm_actual = mixer.norm(captured["core"], captured["z"])
                norm_expected = source.linear_attn.norm(captured["core"], captured["z"])
                row["gdn_output_norm_same_input"] = compare(norm_actual, norm_expected)
                packed = source.linear_attn.in_proj_qkvz(captured["input_norm"])[0]
                row["gdn_packed_qkvz"] = compare(torch.cat((mixer.in_proj_qkv(inputs),
                    mixer.in_proj_z(inputs)), dim=-1)[0], packed)
                packed_ba = source.linear_attn.in_proj_ba(captured["input_norm"])[0]
                row["gdn_packed_ba"] = compare(torch.cat((mixer.in_proj_b(inputs),
                    mixer.in_proj_a(inputs)), dim=-1)[0], packed_ba)
                rec = captured["recurrent"].copy()
                # 原生 kernel 接收 GQA 的少量 Q/K 头；PyTorch 参考需要显式展开。
                for name in ("query", "key"):
                    rec[name] = expand_query_heads(rec[name], rec["value"])
                for name in ("query", "key"):
                    normalized = l2norm_fwd(local[name])[0]
                    row[f"gdn_{name}_conv_and_projection"] = compare(normalized, rec[name])
                    row[f"gdn_{name}_l2norm_same_input"] = compare(
                        l2norm(local[name], dim=-1, eps=1e-6), l2norm_fwd(local[name]))
                for name in ("value", "g", "beta"):
                    row[f"gdn_{name}_same_layer_input"] = compare(local[name][0], rec[name])
                # 原生递推和两种 PyTorch 数学实现使用完全相同的 Q/K/V、门控和初始状态。
                arguments = {name: rec[name][None] for name in ("query", "key", "value", "g", "beta")}
                for name, operation in (("recurrent", torch_recurrent_gated_delta_rule),
                                        ("chunk", torch_chunk_gated_delta_rule)):
                    output, state = operation(**arguments, initial_state=rec["initial_state"],
                                              output_final_state=True, use_qk_l2norm_in_kernel=False)
                    row[f"gdn_{name}_same_operands"] = compare(output.reshape_as(captured["core"]),
                                                                              captured["core"])
                    row[f"gdn_{name}_state_finite"] = bool(torch.isfinite(state).all())
            else:
                import torch_npu
                operation = torch_npu.npu_fusion_attention
                full = {}

                def observed_full(q, k, v, *args, captured_full=full, original=operation, **kwargs):
                    captured_full.update(query=q.clone(), key=k.clone(), value=v.clone())
                    result = original(q, k, v, *args, **kwargs)
                    captured_full["output"] = result[0].clone()
                    return result

                torch_npu.npu_fusion_attention = observed_full
                try:
                    mixed, _ = mixer(inputs, position_embeddings=rotary, attention_mask=allowed,
                                     past_key_values=scratch)
                finally:
                    torch_npu.npu_fusion_attention = operation
                for name in ("query", "key", "value"):
                    actual = full[name][:, :, -length:].transpose(1, 2).reshape(length, -1)
                    expected = captured[f"full_{name}"]
                    if actual.shape[-1] != expected.shape[-1]:
                        expected = expected.reshape(length, -1, mixer.head_dim).repeat_interleave(
                            mixer.num_key_value_groups, dim=1).reshape(length, -1)
                    row[f"full_{name}"] = compare(actual, expected)
                row["full_output"] = compare(full["output"].transpose(1, 2).reshape(length, -1),
                                              captured["full_output"])
                q = captured["full_query"].reshape(length, -1, mixer.head_dim).transpose(0, 1)[None]
                history = pending["cache"][index]
                kv = [torch.cat((history[i], captured[f"full_{name}"].reshape(length, -1, mixer.head_dim)
                      .transpose(0, 1)[None]), dim=2).repeat_interleave(mixer.num_key_value_groups, dim=1)
                      for i, name in enumerate(("key", "value"))]
                direct = operation(q.contiguous(), kv[0].contiguous(), kv[1].contiguous(), q.shape[1],
                    input_layout="BNSD", atten_mask=torch.isneginf(allowed), scale=mixer.scaling,
                    keep_prob=1., sparse_mode=0)[0].transpose(1, 2).reshape(length, -1)
                row["full_kernel_same_operands"] = compare(direct, captured["full_output"])
                for name in ("q_norm", "k_norm"):
                    if name + "_input" in captured:
                        row[name + "_same_input"] = compare(getattr(mixer, name)(captured[name + "_input"]),
                                                            captured[name + "_output"])
                    else:
                        row[name + "_bypassed_by_native_fusion"] = True
                qkv = torch.cat((mixer.q_proj(inputs), mixer.k_proj(inputs), mixer.v_proj(inputs)), dim=-1)[0]
                row["full_packed_projection"] = compare(qkv, captured["full_packed"])
            row["mixer_same_input"] = compare(mixed[0], captured["mixed"])
            self._module_results[str(index)] = row
        self._module_inputs.clear()

    def orthrus_capture_mixer_outputs(self):
        native = getattr(self.model_runner.get_model(), "language_model", self.model_runner.get_model()).model

        def capture(index, module, args):
            if self._layerwise_active and self._layerwise_label == "k32":
                self._module_inputs.setdefault(index, {})["mixed"] = args[0].detach().clone()

        for index, layer in enumerate(native.layers):
            layer.post_attention_layernorm.register_forward_pre_hook(partial(capture, index))
        return "enabled"

    def orthrus_module_report(self):
        return dict(rank=self.model_runner.drafter.tp.rank, modules=self._module_results,
                    candidates=self._module_candidates)


def main():
    prepare_runtime_environment()
    from vllm import LLM, SamplingParams

    args = arguments()
    if args.online or args.block_size != 32:
        raise ValueError("module replay requires a fixed K32 head")
    options = native_options(args) | speculative_options(args)
    options["worker_extension_cls"] = "probe_qwen38_module_replay.ModuleReplayWorker"
    llm = LLM(**options)
    prompt = os.environ.get("ORTHRUS_MODULE_PROMPT", 'Return JSON with keys "name" and "age":')
    history = len(llm.get_tokenizer().encode(prompt))
    if not history:
        raise ValueError("a nonempty diagnostic prefix is required")
    llm.collective_rpc("orthrus_enable_logits_audit", args=(0,))
    llm.collective_rpc("orthrus_enable_module_replay", args=(history,))
    llm.collective_rpc("orthrus_capture_mixer_outputs")
    report = dict(diagnostic_only=True, model=args.model, prompt=prompt, head_dir=args.head_dir,
                  history=history, tp=args.tp, outputs=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fixed_source = os.environ.get("ORTHRUS_MODULE_FIXED_SOURCE")
    fixed_candidates = None
    if fixed_source:
        fixed_candidates = json.loads(Path(fixed_source).read_text())["workers"][0]["candidates"]
        report["fixed_source_sha256"] = hashlib.sha256(Path(fixed_source).read_bytes()).hexdigest()
    matrix = [("decode", 0), ("k32", 31)]
    if os.environ.get("ORTHRUS_MODULE_VARIANTS") == "1":
        matrix.extend((name, 31) for name in ("fp32_l2", "fp32_gate_norm", "fused_mlp", "fp32_conv", "combined"))
    if os.environ.get("ORTHRUS_MODULE_FOLLOWUP") == "1":
        matrix.extend((name, 31) for name in ("combined", "combined_no_conv", "combined_native_projection"))
    if os.environ.get("ORTHRUS_MODULE_NATIVE_OPS") == "1":
        matrix.extend((name, 31) for name in ("combined_native_conv", "combined_native_ops"))
    if os.environ.get("ORTHRUS_MODULE_RESIDUAL") == "1":
        matrix.extend((name, 31) for name in ("combined_native_ops", "combined_native_residual"))
    projection_modes = {"packed_full", "packed_all", "packed_all_gates"}
    if os.environ.get("ORTHRUS_MODULE_PROJECTIONS") == "1":
        matrix.extend((name, 31) for name in ("packed_full", "packed_all", "packed_all_gates"))
    if os.environ.get("ORTHRUS_MODULE_RESIDUAL_ONLY") == "1":
        matrix.extend((name, 31) for name in ("residual_only", "packed_residual"))
    for label, limit in matrix:
        llm.collective_rpc("orthrus_set_proposal_limit", args=(limit,))
        precision = ("residual_only" if label == "packed_residual" else label
                     if label not in {"decode", "k32"} | projection_modes else "original")
        projection = ("packed_all_gates" if label == "packed_residual" else label
                      if label in projection_modes else "original")
        llm.collective_rpc("orthrus_module_precision", args=(precision,))
        llm.collective_rpc("orthrus_projection_diagnostic", args=(projection,))
        llm.collective_rpc("orthrus_diagnostic_candidates", args=("head" if label in {"decode", "k32"} else "fixed",))
        if label == "k32" and fixed_candidates is not None:
            llm.collective_rpc("orthrus_set_fixed_module_candidates", args=(fixed_candidates,))
        llm.collective_rpc("orthrus_arm_layerwise", args=(label,))
        output = llm.generate([prompt], SamplingParams(temperature=0, seed=7,
            max_tokens=2, ignore_eos=True), use_tqdm=False)[0].outputs[0]
        report["outputs"].append(dict(label=label, tokens=list(output.token_ids), text=output.text))
        print(json.dumps(report["outputs"][-1]), flush=True)
    report["workers"] = llm.collective_rpc("orthrus_module_report")
    report["layerwise"] = llm.collective_rpc("orthrus_layerwise_report")
    Path(os.environ["ORTHRUS_MODULE_REPLAY_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"module_replay_output": os.environ["ORTHRUS_MODULE_REPLAY_OUTPUT"]}), flush=True)


if __name__ == "__main__":
    main()
