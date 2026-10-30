"""Opt-in numerical boundary audit for native-engine validation only."""

import functools
import hashlib

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
    def orthrus_memory(self):
        torch.npu.synchronize()
        return dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved())

    def orthrus_target_fingerprint(self):
        """Hash actual native weights, not pointer identity or draft parameters."""
        digest = hashlib.sha256()
        size = 0
        for name, parameter in self.model_runner.get_model().named_parameters():
            raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(name.encode())
            digest.update(memoryview(raw))
            size += raw.nbytes
        return dict(sha256=digest.hexdigest(), bytes=size)

    def orthrus_enable_logits_audit(self):
        runner = self.model_runner
        metadata, sample = runner._build_attention_metadata, runner._sample
        self._logit_trace, self._logit_boundary = [], None

        def capture(*args, **kwargs):
            result = metadata(*args, **kwargs)
            full = next((item for item in result[0].values()
                         if hasattr(item, "seq_lens_list")), None)
            self._logit_boundary = None if full is None else (full.seq_lens_list, full.actual_seq_lengths_q)
            return result

        def traced(logits, spec_decode_metadata):
            if (logits is not None and self._logit_boundary is not None
                    and self._logit_boundary[0][0] >= 3000 and len(self._logit_trace) < 64):
                values, indices = logits.float().topk(4, dim=-1)
                self._logit_trace.append(dict(seq_lens=list(self._logit_boundary[0]),
                                             query_lens=list(self._logit_boundary[1]),
                                             tokens=indices.cpu().tolist(), values=values.cpu().tolist()))
            return sample(logits, spec_decode_metadata)

        runner._build_attention_metadata, runner._sample = capture, traced
        return "enabled"

    def orthrus_validate_online_replay(self, steps):
        """Repeat one reached training block; this is not a serving throughput test."""
        draft = self.model_runner.drafter
        learner = draft.learner
        if type(steps) is not int or not 1 <= steps <= 64 or learner is None or not learner.replay:
            raise ValueError("1..64 steps and an online learner with reached feedback required")
        with torch.inference_mode(False), torch.no_grad():
            before = {name: parameter.detach().clone() for name, parameter in learner.execution.items()}
            cache = _cache_snapshot(draft.history_reader)
            initial = learner.backward()
            start = learner.updates
            losses = [learner.update()["loss"] for _ in range(steps)]
            final = learner.backward()
            learner.optimizer.zero_grad(set_to_none=True)
            changed = [name for name, parameter in learner.execution.items()
                       if not torch.equal(parameter, before[name])]
            unchanged = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                            for a, b in zip(cache, _cache_snapshot(draft.history_reader), strict=True))
            learner._check()
        if not changed or not unchanged or not final < initial:
            raise AssertionError(f"fixed replay failed: loss={initial}->{final}, "
                                 f"changed={len(changed)}, cache_unchanged={unchanged}")
        return dict(rank=draft.tp.rank, steps=learner.updates - start, initial_loss=initial,
                    final_loss=final, losses=losses, changed_parameters=changed,
                    trainable_parameters=learner.trainable_parameters, cache_unchanged=unchanged)

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
        if hasattr(self, "_logit_trace"):
            result["logit_trace"] = self._logit_trace
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
