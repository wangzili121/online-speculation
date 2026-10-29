"""Qwen3.5/3.8 text-only dual view, using existing Transformers mixers.

The GDN extension follows Orthrus' Qwen3.5 reference, not an attention-mask
replacement: its draft recurrence is causal within each masked block. The
explicit policy records this model adaptation rather than silently replacing GDN.
"""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from .backbone import BackboneOutput, DraftBoundary
from .branches import MaskedAttentionBranch


class HybridHistory(tuple):
    """Owned AR snapshots: full-attention KV or GDN convolution/recurrent state."""

    def __new__(cls, states, length):
        obj = super().__new__(cls, states)
        obj.length = length
        return obj


class _ScratchCache:
    """Minimal mixer cache; writes never reach the supplied AR snapshots."""

    def __init__(self, states, layer_types):
        self.layers = []
        for state, kind in zip(states, layer_types, strict=True):
            row = SimpleNamespace(keys=None, values=None, conv_states=None, recurrent_states=None)
            if state is not None:
                if kind == "full_attention":
                    row.keys, row.values = state
                else:
                    # Existing single-token GDN kernels may update these in place.
                    row.conv_states = (state[0].clone(),)
                    row.recurrent_states = (state[1].clone(),)
            self.layers.append(row)

    def has_previous_state(self, layer_idx):
        return self.layers[layer_idx].recurrent_states is not None

    def update(self, key, value, layer_idx, *args, **kwargs):
        row = self.layers[layer_idx]
        if row.keys is not None:
            key = torch.cat((row.keys, key), dim=2)
            value = torch.cat((row.values, value), dim=2)
        row.keys, row.values = key, value
        return key, value

    def update_conv_state(self, state, layer_idx, **kwargs):
        self.layers[layer_idx].conv_states = (state,)
        return state

    def update_recurrent_state(self, state, layer_idx, **kwargs):
        self.layers[layer_idx].recurrent_states = (state,)
        return state

    def snapshot(self, length, layer_types):
        states = []
        for row, kind in zip(self.layers, layer_types, strict=True):
            pair = ((row.keys, row.values) if kind == "full_attention"
                    else (row.conv_states[0], row.recurrent_states[0]))
            states.append(tuple(x.detach().clone() for x in pair))
        return HybridHistory(states, length)


class _HybridLayer(nn.Module):
    def __init__(self, original, kind):
        super().__init__()
        self.kind = kind
        mixer = original.self_attn if kind == "full_attention" else original.linear_attn
        self.attention = nn.Module()
        self.attention.ar = mixer
        self.attention.draft = copy.deepcopy(mixer)
        self.input_norm = original.input_layernorm
        self.post_norm = original.post_attention_layernorm
        self.mlp = original.mlp

    def forward(self, hidden, view, rotary, allowed, cache):
        device = self.input_norm.weight.device
        hidden = hidden.to(device)
        rotary = tuple(x.to(device) for x in rotary)
        allowed = allowed.to(device)
        normalized = self.input_norm(hidden)
        mixer = getattr(self.attention, view)
        if self.kind == "full_attention":
            mixed, _ = mixer(normalized, position_embeddings=rotary,
                             attention_mask=allowed, past_key_values=cache)
        else:
            mixed = mixer(normalized, cache_params=cache, attention_mask=None)
        hidden = hidden + mixed
        return hidden + self.mlp(self.post_norm(hidden))


class Qwen38DualView(nn.Module):
    """Shared frozen backbone, independent draft mixers, no second full model.

    This is an eager reference executor, not yet a vLLM paged-cache adapter.
    Mask token and block size are explicit; neither is guessed from the name.
    """

    def __init__(self, target, *, block_size, mask_token_id, hybrid_policy):
        super().__init__()
        if hybrid_policy != "orthrus_qwen35_reference":
            raise ValueError("explicit orthrus_qwen35_reference hybrid policy required")
        text = getattr(target.model, "language_model", target.model)
        config = copy.deepcopy(text.config)
        kinds = tuple(config.layer_types)
        if (config.model_type != "qwen3_5_text" or len(kinds) != config.num_hidden_layers
                or set(kinds) - {"linear_attention", "full_attention"}):
            raise ValueError("Qwen3.5/3.8 text hybrid architecture required")
        if (type(block_size) is not int or block_size < 2 or type(mask_token_id) is not int
                or not 0 <= mask_token_id < config.vocab_size):
            raise ValueError("explicit valid block size and mask token required")
        if getattr(target, "hf_device_map", None):
            raise ValueError("dispatched/offloaded models need a separate device-placement adapter")
        parameters = list(text.parameters()) + list(target.lm_head.parameters())
        if (len({p.device for p in parameters}) != 1 or any(
                p.dtype not in (torch.float32, torch.bfloat16) for p in parameters)):
            raise ValueError("single-device BF16/FP32 text weights required; INT8 is inference-only here")
        config.block_size, config.mask_token_id = block_size, mask_token_id
        self.config = config
        self.layer_types = kinds
        self.hybrid_policy = hybrid_policy
        self.embedding, self.norm, self.head = text.embed_tokens, text.norm, target.lm_head
        self.rotary = text.rotary_emb
        self.layers = nn.ModuleList(_HybridLayer(layer, kind)
                                    for layer, kind in zip(text.layers, kinds, strict=True))
        self.source = {}
        self.set_backend("sdpa")
        self.eval().requires_grad_(False)

    def set_backend(self, backend):
        if backend not in {"sdpa", "eager"}:
            raise ValueError("hybrid reference supports sdpa or eager")
        self.backend = backend
        for layer in self.layers:
            if layer.kind == "full_attention":
                for view in ("ar", "draft"):
                    getattr(layer.attention, view).config._attn_implementation = backend

    def _apply(self, fn, recurse=True):
        # RoPE frequencies must not lose precision when execution weights cast.
        frequencies = {name: getattr(self.rotary, name).detach().clone()
                       for name in ("inv_freq", "original_inv_freq")
                       if isinstance(getattr(self.rotary, name, None), torch.Tensor)}
        result = super()._apply(fn, recurse)
        for name, value in frequencies.items():
            setattr(self.rotary, name, value.to(device=self.rotary.inv_freq.device, dtype=torch.float32))
        return result

    def place_layers(self, devices):
        """Contiguous layer placement for low-throughput reference validation.

        This is not tensor parallelism. Embedding and LM head stay together so
        the existing proposal/correction loop keeps a single token/logit device.
        """
        if not devices or len(devices) > len(self.layers):
            raise ValueError("one to num_hidden_layers devices required")
        devices = [torch.device(device) for device in devices]
        self.embedding.to(devices[0])
        self.head.to(devices[0])
        self.rotary.to(devices[0])
        for i, layer in enumerate(self.layers):
            layer.to(devices[i * len(devices) // len(self.layers)])
        self.norm.to(devices[-1])
        return self

    def _logits(self, hidden):
        normalized = self.norm(hidden.to(self.norm.weight.device))
        return self.head(normalized.to(self.head.weight.device))

    def _cache(self, cache, start=0):
        if cache is None:
            states = [None] * len(self.layers)
        else:
            if len(cache) != len(self.layers) - start:
                raise ValueError("one history state per executed layer required")
            states = [None] * start + list(cache)
        return _ScratchCache(states, self.layer_types)

    def forward(self, tokens, *, view="ar", cache=None, capture_layer=None,
                logits_to_keep=0):
        if (view not in {"ar", "draft"} or tokens.ndim != 2 or tokens.shape[0] != 1
                or tokens.shape[1] < 1):
            raise ValueError("nonempty batch-one token block and ar/draft view required")
        if cache is not None and not isinstance(cache, HybridHistory):
            raise TypeError("hybrid AR history required, not an ordinary KV tuple")
        if capture_layer is not None and (view != "draft" or type(capture_layer) is not int
                                          or not 0 <= capture_layer < len(self.layers)):
            raise ValueError("valid draft capture layer required")
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("nonnegative logits_to_keep required")
        past = 0 if cache is None else cache.length
        hidden = self.embedding(tokens)
        positions = torch.arange(past, past + tokens.shape[1], device=tokens.device)[None]
        rotary = self.rotary(hidden, positions[None].expand(3, -1, -1))
        # Orthrus: every draft query sees the strict AR prefix and the whole
        # temporary block. AR instead uses each row's own causal boundary.
        allowed = hidden.new_zeros((1, 1, tokens.shape[1], past + tokens.shape[1]))
        if view == "ar":
            future = torch.arange(past + tokens.shape[1], device=tokens.device)[None] > positions[0, :, None]
            allowed.masked_fill_(future, float("-inf"))
        scratch = self._cache(cache)
        boundary = None
        for index, layer in enumerate(self.layers):
            if index == capture_layer:
                boundary = DraftBoundary(hidden, rotary, allowed, index, past, self.backend).detached()
            hidden = layer(hidden, view, rotary, allowed, scratch)
        logits = self._logits(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        history = scratch.snapshot(past + tokens.shape[1], self.layer_types) if view == "ar" else None
        return BackboneOutput(hidden, history, logits, boundary)

    def forward_suffix(self, boundary, *, cache=None, logit_range=None, draft_weights=None):
        if (not isinstance(boundary, DraftBoundary) or boundary.backend != self.backend
                or not 0 <= boundary.start_layer < len(self.layers)):
            raise ValueError("compatible captured draft boundary required")
        start = boundary.start_layer
        scratch = self._cache(cache, start)
        overrides = {} if draft_weights is None else draft_weights
        permitted = {f"layers.{i}.attention.draft.{name}" for i in range(start, len(self.layers))
                     for name, _ in self.layers[i].attention.draft.named_parameters()}
        if set(overrides) - permitted:
            raise ValueError("only suffix draft parameters may be overridden")
        hidden = boundary.hidden
        for i in range(start, len(self.layers)):
            prefix = f"layers.{i}."
            weights = {name[len(prefix):]: value for name, value in overrides.items() if name.startswith(prefix)}
            args = (hidden, "draft", boundary.rotary, boundary.allowed, scratch)
            hidden = (torch.func.functional_call(self.layers[i], weights, args)
                      if weights else self.layers[i](*args))
        projected = hidden if logit_range is None else hidden[:, slice(*logit_range)]
        return BackboneOutput(hidden, None, self._logits(projected))


class Qwen38Branch(MaskedAttentionBranch):
    """Use the same proposals, exact correction and reached-row feedback loop.

    A recurrent state cannot be cropped like KV. After rejection we replay only
    accepted AR inputs from the pre-verification snapshot. This adds work, but
    is a correctness baseline for a later vLLM state-checkpoint implementation.
    """

    @staticmethod
    def history_length(cache):
        return 0 if cache is None else cache.length

    def commit_history(self, proposal, verified_cache, committed):
        previous = self.history_length(proposal.cache)
        count = committed - previous
        if not 0 < count <= proposal.verifier_inputs.shape[1]:
            raise ValueError("committed AR inputs must be a nonempty verified prefix")
        if committed == verified_cache.length:
            return verified_cache
        _, clean = self.ar(proposal.verifier_inputs[:, :count], cache=proposal.cache, logits_to_keep=1)
        return clean
