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


class ValidationWorker(OrthrusWorkerExtension):
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
