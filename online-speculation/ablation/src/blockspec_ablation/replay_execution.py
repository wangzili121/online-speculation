"""Prepared suffix forward/loss/gradient graphs with explicit replay costs.

Each signature owns padded base KV and detached draft inputs. Gradients are
computed into graph-owned outputs, then accumulated into separate optimizer
buffers. Repeated signatures therefore preserve earlier feedback contributions.
"""

import time

import torch

from .distillation import LOSS_KINDS, divergence
from .model import DraftBoundary, PackedCache, cache_length, is_adapter


class _GradientSlot:
    def __init__(self, owner, length, valid):
        model, capacity = owner.model, owner.capacity
        # Slots retain their compute inputs, while ownership flows executor ->
        # slots only. Dropping the executor immediately releases graph storage.
        self.model, self.capacity, self.valid = model, capacity, valid
        self.loss_kind, self.parameters = owner.loss, owner.parameters
        c = model.config
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.hidden = torch.zeros(1, length, c.hidden_size, device=device, dtype=dtype)
        self.positions = torch.arange(length, device=device)[None]
        self.allowed = torch.zeros(1, 1, length, capacity + length, device=device, dtype=torch.bool)
        self.allowed[..., capacity:] = True
        self.mask = torch.ones(1, length, device=device, dtype=torch.bool)
        self.mask[:, 0] = False
        self.past = torch.zeros(c.num_hidden_layers - owner.start_layer, 2, 1,
                                c.num_key_value_heads, capacity, c.head_dim, device=device, dtype=dtype)
        self.cache = PackedCache(self.past)
        self.boundary = DraftBoundary(self.hidden, self.positions, self.allowed, self.mask, owner.start_layer)
        self.teacher = torch.zeros(valid, c.vocab_size, device=device, dtype=dtype)
        self.denominator = torch.ones((), device=device, dtype=dtype)
        self.graph = None
        if owner.use_cuda_graph:
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

    @torch.enable_grad()
    def run(self):
        logits = self.model.forward_suffix(self.boundary, cache=self.cache,
                                            logit_range=(1, 1 + self.valid))
        loss = divergence(logits[0], self.teacher, self.loss_kind).sum() / self.denominator
        self.gradients = torch.autograd.grad(loss, self.parameters)
        # Graph replay owns tensor storage; release Python autograd nodes so
        # later captures create their leaf nodes on their own capture stream.
        self.logits, self.loss = logits.detach(), loss.detach()

    @torch.no_grad()
    def __call__(self, item, positions):
        prefix = cache_length(item.cache)
        boundary = item.boundary
        self.hidden.copy_(boundary.hidden)
        self.positions.copy_(boundary.positions)
        self.mask.copy_(boundary.adapter_mask)
        self.allowed.zero_()
        self.allowed[..., :prefix].copy_(boundary.allowed[..., :prefix])
        self.allowed[..., self.capacity:].copy_(boundary.allowed[..., prefix:])
        if item.cache is not None:
            for dst, (k, v) in zip(self.past, item.cache):
                dst[0, :, :, :prefix].copy_(k)
                dst[1, :, :, :prefix].copy_(v)
        self.teacher.copy_(item.teacher_logits)
        self.denominator.fill_(positions)
        if self.graph is None:
            self.run()
        else:
            self.graph.replay()
        return self.loss, self.gradients


class SuffixReplayExecutor:
    """One model, one stream, prepared (block length, valid rows) signatures.

    Construct after selecting the learner's trainable suffix. prepare() computes
    gradients only; model weights, existing .grad fields and Adam remain intact.
    Parameter storage is fixed, while values may be updated in-place between uses.
    """

    def __init__(self, model, *, start_layer, loss, capacity, max_query, use_cuda_graph=True):
        count = model.config.num_hidden_layers
        if type(start_layer) is not int or not 0 <= start_layer < count:
            raise ValueError("prepared replay requires a valid suffix layer")
        if type(capacity) is not int or capacity < 1 or type(max_query) is not int or max_query < 2:
            raise ValueError("positive prefix capacity and block length >= 2 required")
        if loss not in LOSS_KINDS:
            raise ValueError("unknown replay loss")
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        if use_cuda_graph and (device.type != "cuda" or dtype != torch.float32):
            raise ValueError("prepared graph replay requires an FP32 CUDA model")
        if any(p.device != device or p.dtype != dtype for p in model.parameters()):
            raise ValueError("prepared replay needs uniform parameter device and dtype")
        expected = [p for i, layer in enumerate(model.model.layers) if i >= start_layer
                    for n, p in layer.named_parameters() if is_adapter(n)]
        actual = [p for p in model.parameters() if p.requires_grad]
        if not expected or [id(p) for p in actual] != [id(p) for p in expected]:
            raise ValueError("trainable parameters must equal the selected suffix adapters")
        self.model, self.start_layer, self.loss = model, start_layer, loss
        self.capacity, self.max_query, self.use_cuda_graph = capacity, max_query, use_cuda_graph
        self.parameters = expected
        self.gradients = [torch.zeros_like(p) for p in expected]
        self.slots, self.setup_seconds = {}, 0.0
        self._storage = [(n, id(p), p.data_ptr(), p.requires_grad) for n, p in model.named_parameters()]
        self._buffers = [(n, id(b), b.data_ptr(), b._version) for n, b in model.named_buffers()]
        self._attention = model.attention_signature()

    def validate(self, model, start_layer, loss):
        if model is not self.model or start_layer != self.start_layer or loss != self.loss:
            raise ValueError("learner and prepared replay must share model, suffix and loss")
        if model.attention_signature() != self._attention:
            raise RuntimeError("attention execution changed; rebuild replay graphs")
        if [(n, id(p), p.data_ptr(), p.requires_grad) for n, p in model.named_parameters()] != self._storage:
            raise RuntimeError("model storage or trainable scope changed; rebuild replay graphs")
        if [(n, id(b), b.data_ptr(), b._version) for n, b in model.named_buffers()] != self._buffers:
            raise RuntimeError("model buffers changed; rebuild replay graphs")

    def prepare(self, signatures):
        self.validate(self.model, self.start_layer, self.loss)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for length, valid in signatures:
            if (type(length) is not int or type(valid) is not int
                    or not 2 <= length <= self.max_query or not 1 <= valid < length):
                raise ValueError("invalid replay signature")
            if (length, valid) not in self.slots:
                self.slots[length, valid] = _GradientSlot(self, length, valid)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.setup_seconds += time.perf_counter() - start

    def _validate_item(self, item):
        boundary = item.boundary
        if boundary is None or boundary.start_layer != self.start_layer:
            raise ValueError("matching draft boundary required")
        length, prefix = item.inputs.shape[1], cache_length(item.cache)
        key = (length, item.valid)
        if key not in self.slots:
            raise ValueError("unprepared replay signature; prepare before generation")
        c = self.model.config
        if prefix > self.capacity:
            raise ValueError("replay prefix exceeds prepared capacity")
        if (item.inputs.shape != (1, length) or boundary.hidden.shape != (1, length, c.hidden_size)
                or boundary.positions.shape != (1, length)
                or boundary.positions.dtype not in (torch.int32, torch.int64)
                or boundary.allowed.shape != (1, 1, length, prefix + length)
                or boundary.allowed.dtype != torch.bool or boundary.adapter_mask is None
                or boundary.adapter_mask.shape != (1, length) or boundary.adapter_mask.dtype != torch.bool
                or item.teacher_logits.shape != (item.valid, c.vocab_size)):
            raise ValueError("invalid prepared replay layout")
        expected = (1, c.num_key_value_heads, prefix, c.head_dim)
        if item.cache is not None and (len(item.cache) != c.num_hidden_layers - self.start_layer
                                      or any(k.shape != expected or v.shape != expected for k, v in item.cache)):
            raise ValueError("invalid suffix prefix cache")
        device, dtype = self.parameters[0].device, self.parameters[0].dtype
        values = [boundary.hidden, item.teacher_logits]
        if item.cache is not None:
            values += [value for pair in item.cache for value in pair]
        if (any(value.device != device or value.dtype != dtype for value in values)
                or any(value.device != device for value in (
                    item.inputs, boundary.positions, boundary.allowed, boundary.adapter_mask))):
            raise ValueError("replay tensors must match the prepared device and dtype")
        return key

    @torch.no_grad()
    def backward(self, replay):
        """Write the full replay gradient; clipping and Adam are performed by the learner."""
        self.validate(self.model, self.start_layer, self.loss)
        if torch.is_autocast_enabled(next(self.model.parameters()).device.type):
            raise RuntimeError("autocast is outside the prepared replay contract")
        keys = [self._validate_item(item) for item in replay]
        if not keys:
            raise ValueError("positive replay feedback required")
        positions = sum(item.valid for item in replay)
        torch._foreach_zero_(self.gradients)
        total = 0.0
        for key, item in zip(keys, replay):
            loss, gradients = self.slots[key](item, positions)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite online loss")
            torch._foreach_add_(self.gradients, gradients)
            total += float(loss.detach())
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        return total
