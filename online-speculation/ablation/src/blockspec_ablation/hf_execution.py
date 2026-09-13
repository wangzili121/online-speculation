"""Frozen, caller-selected HF backbone for the independently implemented samplers.

The model author's SDPA layers provide a shared execution reference for AR,
parallel drafting and PrefixRelay. Token routing, functional cache ownership,
sampling, verification and head learning remain in blockspec.
"""

from types import SimpleNamespace
import json
from pathlib import Path

from blockspec.reference import checked_reference

import torch
from torch import nn

from .adapter_io import load_peft_adapter, peft_config
from .checkpoint import config_from_hf
from .model import DraftBoundary, GatedLinear




class _RoutedLinear(GatedLinear):
    def __init__(self, linear, rank, alpha, routing):
        nn.Module.__init__(self)
        self.weight, self.bias = linear.weight, linear.bias
        self.rank, self.scale, self.routing = rank, alpha / rank, routing
        self.lora_A = nn.Parameter(linear.weight.new_zeros(rank, linear.in_features), requires_grad=False)
        self.lora_B = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank), requires_grad=False)

    def forward(self, x):
        return super().forward(x, self.routing.mask)


class FrozenHFDecoder(nn.Module):
    """Causal batch-one bridge; every call owns its returned KV container.

    A fresh DynamicCache imports the caller's immutable K/V tensors. HF appends
    through concatenation, allowing our rejection path to retain clean slices.
    The routing context is scoped to a single synchronous backbone call.
    """
    def __init__(self, reference, config):
        super().__init__()
        self.config = config
        self.model, self.lm_head = reference.model, reference.lm_head
        self.routing = SimpleNamespace(mask=None, active=False)
        from transformers.cache_utils import DynamicCache
        self.cache_type = DynamicCache
        targets = [(name, layer) for name, layer in self.model.named_modules()
                   if name.rsplit(".", 1)[-1] in config.adapter_targets]
        if len(targets) != config.num_hidden_layers * len(config.adapter_targets):
            raise ValueError("one complete set of adapter projections per decoder layer required")
        for name, layer in targets:
            if not isinstance(layer, nn.Linear):
                raise ValueError("dense HF projection required")
            parent, attribute = name.rsplit(".", 1)
            setattr(self.model.get_submodule(parent), attribute,
                    _RoutedLinear(layer, config.adapter_rank, config.adapter_alpha, self.routing))
        self.eval().requires_grad_(False)

    def attention_signature(self):
        return (self.model.config._attn_implementation,) * self.config.num_hidden_layers

    def set_attention_backend(self, backend):
        if backend != "sdpa" or self.model.config._attn_implementation != "sdpa":
            raise ValueError("frozen HF reference requires its pinned SDPA backend")
        return self

    @torch.no_grad()
    def forward(self, tokens, *, cache=None, adapter_mask=None, return_cache=False,
                capture_layer=None, positions=None, allowed=None):
        if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1
                or allowed is not None or capture_layer not in (None, self.config.num_hidden_layers)):
            raise ValueError("causal batch-one queries and optional terminal feature capture required")
        if adapter_mask is not None and (adapter_mask.shape != tokens.shape or adapter_mask.dtype != torch.bool):
            raise ValueError("one Boolean adapter gate per query token required")
        if self.routing.active:
            raise RuntimeError("a routing context serves one backbone call at a time")
        if cache is not None and len(cache) != self.config.num_hidden_layers:
            raise ValueError("one functional K/V pair per layer required")
        past_length = 0 if cache is None else cache[0][0].shape[2]
        kv = self.cache_type()
        if cache is not None:
            for index, (key, value) in enumerate(cache):
                kv.update(key, value, index)
        captured = []
        hook = (self.model.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
                if capture_layer is not None else None)
        self.routing.mask, self.routing.active = adapter_mask, True
        try:
            output = self.model(tokens, past_key_values=kv, use_cache=True, position_ids=positions,
                                output_hidden_states=False)
            logits = self.lm_head(output.last_hidden_state)
        finally:
            self.routing.mask, self.routing.active = None, False
            if hook:
                hook.remove()
        if not return_cache:
            return logits
        result = logits, tuple((layer.keys, layer.values) for layer in kv.layers)
        if capture_layer is not None:
            n = tokens.shape[1]
            pos = (torch.arange(past_length, past_length + n, device=tokens.device)[None]
                   if positions is None else positions)
            causal = torch.arange(past_length + n, device=tokens.device)[None] <= (
                torch.arange(n, device=tokens.device)[:, None] + past_length)
            result += (DraftBoundary(captured[0], pos, causal, adapter_mask, capture_layer),)
        return result


def load_frozen_hf(directory, adapter, *, reference_manifest, expected_sha256=None, device="cuda", dtype=torch.bfloat16):
    spec = checked_reference(directory, reference_manifest)
    import transformers
    if spec.get("reference_transformers") and transformers.__version__ != spec["reference_transformers"]:
        raise RuntimeError(f"frozen HF execution requires isolated Transformers {spec['reference_transformers']}")
    peft = peft_config(adapter)
    raw = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
    config = config_from_hf(raw, rank=peft["r"], alpha=peft["lora_alpha"])
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    model = FrozenHFDecoder(reference, config)
    provenance = load_peft_adapter(adapter, model, expected_sha256=expected_sha256)
    provenance.update(backbone="hf_sdpa", transformers=str(transformers.__version__),
                      base_weight_sha256=spec["weight_sha256"], reference_lf_sha256=spec["reference_lf_sha256"])
    return model, provenance
