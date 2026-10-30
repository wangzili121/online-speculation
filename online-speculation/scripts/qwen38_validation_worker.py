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


def _fingerprint(parameters):
    digest = hashlib.sha256()
    size = 0
    for name, parameter in parameters:
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(name.encode())
        digest.update(memoryview(raw))
        size += raw.nbytes
    return dict(sha256=digest.hexdigest(), bytes=size)


class ValidationWorker(OrthrusWorkerExtension):
    def orthrus_enable_projection_audit(self):
        target = self.model_runner.get_model()
        module = getattr(target, "language_model", target).model.layers[0].linear_attn.in_proj_qkvz
        self._projection_rows, seen = [], set()

        def capture(module, args, output):
            boundary = self._logit_boundary
            request = self.model_runner.input_batch.req_ids[0]
            if boundary is None or boundary[0][0] - boundary[1][0] != 11 or request in seen:
                return
            seen.add(request)
            value = output[0] if isinstance(output, tuple) else output
            self._projection_rows.append((args[0].shape[0], args[0][0].detach().cpu(), value[0].detach().cpu()))

        module.register_forward_hook(capture)
        self._projection_module = module
        return "enabled"

    def orthrus_projection_audit(self):
        module = self._projection_module
        rows = self._projection_rows
        if not rows:
            raise AssertionError("no first-decode projection input captured")
        reference = rows[0][1].to(module.weight.device)
        direct, baseline = [], None
        # 完全相同的输入和主模型权重，只改变已有 linear 算子的 batch 行数。
        for width in (2, 8, 16, 18, 32):
            value = torch.nn.functional.linear(reference[None].expand(width, -1).contiguous(), module.weight)[0].cpu()
            if baseline is None:
                baseline = value
            direct.append(dict(width=width, max_abs=float((value.float() - baseline.float()).abs().max()),
                               equal=torch.equal(value, baseline)))
        return dict(native_calls=[dict(width=width, input_equal=torch.equal(x, rows[0][1]),
            output_max_abs=float((y.float() - rows[0][2].float()).abs().max())) for width, x, y in rows],
            direct_linear=direct)

    def orthrus_set_proposal_limit(self, limit):
        draft = self.model_runner.drafter
        if not 0 <= limit < draft.block_size:
            raise ValueError("proposal limit must fit the configured block")
        self._proposal_limit = limit
        if not hasattr(self, "_unlimited_propose"):
            self._unlimited_propose = draft.propose_batch

            def limited(*args, **kwargs):
                return [row[:self._proposal_limit] for row in self._unlimited_propose(*args, **kwargs)]

            draft.propose_batch = limited
        return limit

    def orthrus_take_logits_audit(self):
        rows, self._logit_trace = self._logit_trace, []
        return rows

    def orthrus_enable_sampling_calibration(self, allowed):
        """Capture conditional probabilities before native sampling mutates logits."""
        runner, self._sampling_rows = self.model_runner, []
        sample = runner._sample

        def checked(logits, metadata):
            rows = logits if metadata is None else torch.cat(
                (logits[metadata.target_logits_indices], logits[metadata.bonus_logits_indices]))
            probabilities = rows[:, allowed].float().softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise AssertionError("nonfinite native calibration probabilities")
            probabilities = probabilities.cpu().tolist()
            result = sample(logits, metadata)
            tokens = result.sampled_token_ids[0].cpu().tolist()
            self._sampling_rows.extend(dict(token=token, probabilities=probability)
                for token, probability in zip(tokens, probabilities) if token >= 0)
            return result

        runner._sample = checked
        return "enabled"

    def orthrus_sampling_rows(self):
        rows, self._sampling_rows = self._sampling_rows, []
        return rows

    def orthrus_set_proposal_fixture(self, tokens, mode):
        """Drive cache branches using deterministic proposals, not a trained head."""
        if mode not in {"oracle", "reject"} or len(tokens) < 64:
            raise ValueError("oracle/reject fixture requires a known long greedy continuation")
        draft = self.model_runner.drafter
        propose, read = draft.propose_batch, draft.history_reader.read
        self._fixture_mode, self._fixture_reads = mode, []

        def history(*args, **kwargs):
            if kwargs["draft_count"]:
                self._fixture_reads.append(kwargs["sampled_count"])
            return read(*args, **kwargs)

        def controlled(*args, **kwargs):
            result = propose(*args, **kwargs)
            pending = draft.pending
            if pending is None:
                return result
            batch = draft.runner.input_batch
            generated = int(batch.num_tokens_no_spec[0] - batch.num_prompt_tokens[0])
            width = pending["candidates"].numel()
            values = tokens[generated:generated + width]
            values += [tokens[-1]] * (width - len(values))
            if mode == "reject":
                values = [(token + 1) % draft.model.config.vocab_size for token in values]
            pending["candidates"] = pending["candidates"].new_tensor(values)
            pending["q"] = None  # The fixture proposes deterministically; the target is greedy.
            return [values]

        draft.propose_batch, draft.history_reader.read = controlled, history
        return mode

    def orthrus_enable_ar_view_audit(self):
        """Compare mapped AR weights with native verifier logits on real prefixes."""
        draft = self.model_runner.drafter
        self._ar_view_checks = []
        requests = set()
        observe = draft.observe_verified

        def teacher(logits, metadata):
            count = draft._verification_count(metadata)
            if not count:
                return None
            with torch.inference_mode(False), torch.no_grad():
                return logits[metadata.target_logits_indices].detach().clone()

        def checked(metadata, target, sampled):
            pending = draft.pending
            if pending is not None and pending["request_id"] not in requests and len(requests) < 4:
                count = metadata.num_draft_tokens[0]
                with torch.inference_mode(False), torch.no_grad():
                    tokens = torch.cat((pending["inputs"][:, :1], pending["candidates"][None, :count-1]), 1)
                    actual = draft.model(tokens, view="ar", cache=pending["cache"]).logits[0].float()
                    expected = target[:count].float()
                    if actual.shape != expected.shape or not torch.isfinite(actual).all():
                        raise AssertionError("mapped AR view has invalid logits")
                    tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
                    maximum = float(tv.max())
                    self._ar_view_checks.append(dict(history=pending["cache"].length, rows=count,
                        max_tv=maximum, max_logit_abs=float((actual - expected).abs().max()),
                        top1_matches=int((actual.argmax(-1) == expected.argmax(-1)).sum()),
                        passed=maximum <= .05, row_tv=tv.cpu().tolist(),
                        first_native_tokens=expected[0].topk(4).indices.cpu().tolist(),
                        first_native_logits=expected[0].topk(4).values.cpu().tolist(),
                        first_mapped_tokens=actual[0].topk(4).indices.cpu().tolist(),
                        first_mapped_logits=actual[0].topk(4).values.cpu().tolist()))
                    requests.add(pending["request_id"])
            return observe(metadata, target, sampled)

        draft.teacher_rows = teacher
        draft.observe_verified = checked
        return "enabled"

    def orthrus_memory(self):
        torch.npu.synchronize()
        return dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved())

    def orthrus_target_fingerprint(self):
        """Hash actual native weights, not pointer identity or draft parameters."""
        return _fingerprint(self.model_runner.get_model().named_parameters())

    def orthrus_head_fingerprint(self):
        return _fingerprint(self.model_runner.drafter._draft_parameters().items())

    def orthrus_enable_logits_audit(self, min_seq_len=3000):
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
                    and self._logit_boundary[0][0] >= min_seq_len and len(self._logit_trace) < 64):
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
        return dict(rank=draft.tp.rank, steps=learner.updates - start, initial_loss=initial,
                    final_loss=final, losses=losses, changed_parameters=changed,
                    trainable_parameters=learner.trainable_parameters, cache_unchanged=unchanged,
                    passed=bool(changed) and unchanged and final < initial)

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
        if hasattr(self, "_ar_view_checks"):
            result["ar_view_checks"] = self._ar_view_checks
        if hasattr(self, "_fixture_mode"):
            result["proposal_fixture"] = self._fixture_mode
            result["fixture_accepted_reads"] = self._fixture_reads
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
