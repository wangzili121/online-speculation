"""Fixed-shape inference on our Decoder, optionally replayed as CUDA graphs.

Private padded KV workspaces serve the attention layers. Each returned cache,
logit tensor and draft feature owns a fresh snapshot for functional online replay.
The executor uses batch-one, fixed-capacity FP32/BF16 CUDA graphs and eager long
prefill. AR, static speculation and online continuation share this engine.
"""

import time

import torch

from .model import DraftBoundary, PackedCache, cache_length, is_adapter


class _ForwardSlot:
    @torch.no_grad()
    def __init__(self, model, capacity, length, adapted, capture_layer, use_cuda_graph, past):
        self.model, self.capacity = model, capacity
        device = next(model.parameters()).device
        self.ids = torch.zeros(1, length, dtype=torch.long, device=device)
        self.positions = torch.arange(length, device=device)[None]
        self.allowed = torch.zeros(1, 1, length, capacity + length, dtype=torch.bool, device=device)
        self.allowed[..., capacity:] = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        self.mask = torch.ones_like(self.ids, dtype=torch.bool) if adapted else None
        if self.mask is not None:
            self.mask[:, 0] = False
        self.past = past
        self.cache = PackedCache(self.past)
        self.capture_layer = capture_layer
        self.graph = None
        if use_cuda_graph:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.run()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.run()
            self.graph = graph

    def run(self):
        output = self.model(self.ids, positions=self.positions, allowed=self.allowed,
                            adapter_mask=self.mask, cache=self.cache, return_cache=True,
                            capture_layer=self.capture_layer)
        self.logits = output[0]
        # Attention returns the input prefix followed by this query's KV.
        # Pack the new rows; the input workspace already holds the prefix.
        self.new_cache = torch.stack([torch.stack([kv[..., self.capacity:, :] for kv in layer])
                                      for layer in output[1]])
        self.boundary = output[2] if self.capture_layer is not None else None

    @torch.no_grad()
    def __call__(self, ids, positions, allowed, adapter_mask, cache):
        prefix = cache_length(cache)
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        if self.mask is not None:
            self.mask.copy_(adapter_mask)
        self.allowed.zero_()
        self.allowed[..., :prefix].copy_(allowed[..., :prefix])
        self.allowed[..., self.capacity:].copy_(allowed[..., prefix:])
        if cache is not None:
            if isinstance(cache, PackedCache):
                self.past[..., :prefix, :].copy_(cache.packed)
            else:
                for dst, (k, v) in zip(self.past, cache):
                    dst[0, :, :, :prefix].copy_(k)
                    dst[1, :, :, :prefix].copy_(v)
        if self.graph is None:
            self.run()
        else:
            self.graph.replay()
        # Joining the valid input prefix and new rows creates an owned snapshot.
        packed = torch.cat((self.past[..., :prefix, :], self.new_cache), dim=4).detach()
        result = (self.logits.clone(), PackedCache(packed))
        if self.boundary is not None:
            boundary = DraftBoundary(self.boundary.hidden.clone(), positions.clone(), allowed.clone(),
                                     None if adapter_mask is None else adapter_mask.clone(), self.capture_layer)
            result += (boundary,)
        return result


class FixedShapeExecutor:
    """One owner, one stream, in-place adapter publication between decode rounds.

    prepare() must capture every allowed (length, adapted, boundary) signature
    before measured generation. Unknown short signatures FAIL, never silently
    capture inside a timed request. Inputs longer than max_query use eager
    prefill with the same Decoder. Replacing model storage invalidates graphs.
    """

    def __init__(self, model, *, capacity, max_query, use_cuda_graph=True):
        if type(capacity) is not int or capacity < 1 or type(max_query) is not int or max_query < 1:
            raise ValueError("positive integer prefix capacity and maximum query required")
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        if use_cuda_graph and (device.type != "cuda" or dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError("CUDA graph executor requires an FP32 or BF16 CUDA model")
        if any(p.device != device or (p.dtype != dtype and not (
                dtype == torch.bfloat16 and is_adapter(n) and p.dtype == torch.float32))
               for n, p in model.named_parameters()):
            raise ValueError("executor needs one base dtype and device; BF16 permits FP32 adapter masters")
        self.model, self.capacity, self.max_query = model, capacity, max_query
        self.use_cuda_graph = use_cuda_graph
        self.slots, self.setup_seconds = {}, 0.0
        self._past = None
        self.signature_seconds = {}
        self._storage = [(n, id(p), p.data_ptr(), p.dtype, p.device) for n, p in model.named_parameters()]
        self._buffers = [(n, id(b), b.data_ptr(), b._version) for n, b in model.named_buffers()]
        self._attention = model.attention_signature()

    def validate(self, model):
        if model is not self.model:
            raise ValueError("executor and decoder must share a model")
        if model.attention_signature() != self._attention:
            raise RuntimeError("attention execution changed; rebuild executor")
        if [(n, id(p), p.data_ptr(), p.dtype, p.device) for n, p in model.named_parameters()] != self._storage:
            raise RuntimeError("model storage changed; discard executor and recapture")
        if [(n, id(b), b.data_ptr(), b._version) for n, b in model.named_buffers()] != self._buffers:
            raise RuntimeError("model buffers changed; discard executor and recapture")

    @torch.no_grad()
    def prepare(self, signatures):
        """Capture explicit signatures; this setup time is reported separately."""
        self.validate(self.model)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for length, adapted, capture_layer in signatures:
            if type(length) is not int or not 1 <= length <= self.max_query or type(adapted) is not bool:
                raise ValueError("invalid graph query signature")
            if capture_layer is not None and (type(capture_layer) is not int
                                              or not 0 <= capture_layer <= self.model.config.num_hidden_layers):
                raise ValueError("invalid captured boundary layer")
            key = (length, adapted, capture_layer)
            if key not in self.slots:
                before = time.perf_counter()
                if self._past is None:
                    c = self.model.config
                    self._past = torch.zeros(c.num_hidden_layers, 2, 1, c.num_key_value_heads,
                                             self.capacity, c.head_dim, device=device,
                                             dtype=next(self.model.parameters()).dtype)
                self.slots[key] = _ForwardSlot(self.model, self.capacity, length, adapted,
                                               capture_layer, self.use_cuda_graph, self._past)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                self.signature_seconds[key] = time.perf_counter() - before
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.setup_seconds += time.perf_counter() - start

    def __call__(self, tokens, **kwargs):
        self.validate(self.model)
        return self._forward(tokens, **kwargs)

    def _forward(self, tokens, *, positions=None, allowed=None, adapter_mask=None,
                 cache=None, return_cache=False, capture_layer=None):
        # Decoders validate once per request and then use this private method.
        # Only the synchronous learner's IN-PLACE updates are allowed in between.
        if torch.is_grad_enabled():
            raise RuntimeError("fixed-shape executor is inference-only; train through the Decoder")
        if torch.is_autocast_enabled(tokens.device.type):
            raise RuntimeError("autocast is outside the captured inference contract")
        if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1
                or tokens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("nonempty batch-one input required")
        prefix, length = cache_length(cache), tokens.shape[1]
        if prefix > self.capacity:
            raise ValueError("prefix exceeds prepared capacity")
        if capture_layer is not None and not return_cache:
            raise ValueError("boundary capture requires return_cache")
        if length > self.max_query:
            return self.model(tokens, positions=positions, allowed=allowed, adapter_mask=adapter_mask,
                              cache=cache, return_cache=return_cache, capture_layer=capture_layer)
        key = (length, adapter_mask is not None, capture_layer)
        if key not in self.slots:
            raise ValueError(f"unprepared decode signature: {key}; prepare outside measurement")
        if positions is None:
            positions = torch.arange(prefix, prefix + length, device=tokens.device)[None]
        if allowed is None:
            allowed = (torch.arange(prefix + length, device=tokens.device)[None, :] <=
                       (prefix + torch.arange(length, device=tokens.device))[:, None])[None, None]
        if (positions.shape != tokens.shape or positions.dtype not in (torch.int32, torch.int64)
                or allowed.dtype != torch.bool
                or allowed.shape != (1, 1, length, prefix + length)
                or (adapter_mask is not None and (adapter_mask.shape != tokens.shape
                                                  or adapter_mask.dtype != torch.bool))):
            raise ValueError("invalid fixed-shape forward layout")
        c = self.model.config
        expected = (1, c.num_key_value_heads, prefix, c.head_dim)
        if cache is not None and (len(cache) != c.num_hidden_layers
                                  or any(k.shape != expected or v.shape != expected for k, v in cache)):
            raise ValueError("invalid fixed-shape prefix cache")
        parameter = next(self.model.parameters())
        if cache is not None and any(value.dtype != parameter.dtype or value.device != parameter.device
                                     for layer in cache for value in layer):
            raise ValueError("prefix cache must match the execution base dtype and device")
        result = self.slots[key](tokens, positions, allowed, adapter_mask, cache)
        return result if return_cache else result[0]
