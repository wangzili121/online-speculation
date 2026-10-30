"""Opt-in numerical boundary audit for native-engine validation only."""

import functools

import torch

from blockspec.vllm_ascend.qwen38_proposer import OrthrusWorkerExtension


def _check(name, module, args, output):
    values = output if isinstance(output, tuple) else (output,)
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"native target nonfinite boundary: {name}, "
                                     f"shape={tuple(value.shape)}, nan={int(value.isnan().sum())}")


def _inputs(name, module, args):
    _check(name + ".input", module, (), args)


def _gdn_metadata(module, args):
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata[module.prefix]
    masks = metadata.spec_sequence_masks
    print("GDN_AUDIT", {"spec_masks": None if masks is None else masks.cpu().tolist(),
                        "prefills": metadata.num_prefills, "decodes": metadata.num_decodes,
                        "state_shape": tuple(module.kv_cache[1].shape),
                        "state_dtype": str(module.kv_cache[1].dtype),
                        "state_indices": None if metadata.non_spec_state_indices_tensor is None
                        else metadata.non_spec_state_indices_tensor.cpu().tolist()}, flush=True)


def _cache_snapshot(reader):
    values = []
    for kind, module, name in reader.layers:
        meta = reader.metadata[name]
        if kind == "full_attention":
            for cache in module.kv_cache:
                pages = (meta.seq_lens_list[0] + cache.shape[1] - 1) // cache.shape[1]
                indices = meta.block_tables[0, :pages]
                values.append(cache.index_select(0, indices.to(cache.device, torch.long)))
        else:
            if meta.spec_sequence_masks is not None:
                indices = meta.spec_state_indices_tensor[0]
                conv_index = meta.spec_decode_metadata.spec_causal_conv1d.cache_indices[0, :1]
            else:
                indices = meta.non_spec_state_indices_tensor[:1]
                conv_meta = (meta.non_spec_prefill_metadata.causal_conv1d if meta.num_prefills
                             else meta.non_spec_decode_metadata.causal_conv1d)
                conv_index = conv_meta.cache_indices.reshape(1, -1)[0, :1]
            for cache, index in zip(module.kv_cache, (conv_index, indices), strict=True):
                values.append(cache.index_select(0, index.to(cache.device, torch.long)))
    return values


class ValidationWorker(OrthrusWorkerExtension):
    def orthrus_enable_cache_audit(self):
        draft = self.model_runner.drafter
        self._cache_audited = set()
        for method in ("propose_batch", "observe_verified"):
            original = getattr(draft, method)

            def checked(*args, _method=method, _original=original, **kwargs):
                if _method == "propose_batch" and not args[0][0]:
                    return _original(*args, **kwargs)
                reader = draft.history_reader
                speculative = any(reader.metadata[name].spec_sequence_masks is not None
                                  for kind, _, name in reader.layers if kind == "linear_attention")
                key = (reader.request_id, _method, speculative)
                # 每个请求检查首次 prefill/draft、投机 draft 和在线更新，避免每轮复制所有状态。
                if key in self._cache_audited:
                    return _original(*args, **kwargs)
                before = _cache_snapshot(reader)
                result = _original(*args, **kwargs)
                after = _cache_snapshot(reader)
                if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                           for a, b in zip(before, after, strict=True)):
                    raise AssertionError(f"Orthrus modified native target cache during {_method}")
                self._cache_audited.add(key)
                return result

            setattr(draft, method, checked)
        return "enabled"

    def orthrus_stats(self):
        result = super().orthrus_stats()
        result["cache_readonly_checks"] = len(getattr(self, "_cache_audited", ()))
        return result

    def orthrus_enable_numeric_audit(self):
        target = self.model_runner.get_model()
        target = getattr(target, "language_model", target).model
        self._orthus_audit_handles = []
        for index, layer in enumerate(target.layers):
            for name in ("input_layernorm", "linear_attn", "self_attn", "mlp"):
                module = getattr(layer, name, None)
                if module is not None:
                    hook = functools.partial(_check, f"layer{index}.{name}")
                    self._orthus_audit_handles.append(module.register_forward_hook(hook))
                    if index == 0 and name == "linear_attn":
                        self._orthus_audit_handles.append(module.register_forward_pre_hook(_gdn_metadata))
                        for part in ("in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
                            child = getattr(module, part)
                            label = f"layer0.linear_attn.{part}"
                            self._orthus_audit_handles.append(child.register_forward_hook(functools.partial(_check, label)))
                            self._orthus_audit_handles.append(child.register_forward_pre_hook(functools.partial(_inputs, label)))
        return "enabled"
