"""K32 neural draft executor sharing a native worker and target backbone banks."""

import time

import torch

from .baseline import native_module
from .programs import make_programs
from .target import OrthrusRunner


def metadata(head, native_samples):
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    hidden, result = empty((32, 5120)), empty((32, 5120))
    gdn = [empty(value.shape, torch.float32 if index == 5 else value.dtype)
           for index, value in enumerate(head.banks(0))]
    full = [empty(value.shape, value.dtype) for value in head.banks(3)]
    # Native samples: weights occupy 1..23; resident states/cache occupy 29..32.
    q, k, v, gate = [empty((32, width)) for width in (6144, 1024, 1024, 6144)]
    cos, sin = empty((32, 64)), empty((32, 64))
    return {
        "embedding": [empty((32,), torch.int32), native_samples[1], hidden],
        "gdn": [hidden, *native_samples[4:9], *gdn, *native_samples[29:31], result, 0, 0],
        "full_prepare": [hidden, native_samples[4], *full[:5], cos, sin, q, k, v, gate, 3],
        "full": [hidden, *native_samples[5:9], full[-1], q, k, v, gate, *native_samples[31:33],
                 empty((32,), torch.int32), empty((16, 512), torch.float32), result, 3, 0, 32],
        "finish": [hidden, native_samples[2], native_samples[3], empty((32, 248320))],
    }


class NeuralDraft:
    context = 512

    def __init__(self, source, head):
        self.source, self.head = source, head
        self.block_size = head.block_size
        self.programs, self.handles = {}, []
        self.native = None
        self.rounds = 0
        self.last_seconds = None

    def compile(self, samples, config):
        inputs = metadata(self.head, samples)
        for name, program in make_programs(self.source).items():
            print(f"compiling Orthrus {name}", flush=True)
            self.programs[name] = program.compile(*inputs[name], config=config)
        return list(self.programs.values())

    def _allocate(self, shape, dtype=torch.bfloat16):
        handle = self.native.worker.alloc_tensor(shape, dtype)
        self.handles.append(handle)
        return handle

    def start(self, native):
        if self.native is not None or len(self.programs) != 5:
            raise RuntimeError("draft programs must be registered before starting the native worker")
        self.native = native
        self.hidden = [self._allocate((32, 5120)), self._allocate((32, 5120))]
        self.query, self.key, self.value, self.gate = [self._allocate((32, width))
                                                     for width in (6144, 1024, 1024, 6144)]
        self.ids = self._allocate((32,), torch.int32)
        self.cos, self.sin = self._allocate((32, 64)), self._allocate((32, 64))
        self.table = self._allocate((32,), torch.int32)
        self.mask = self._allocate((16, 512), torch.float32)
        self.logits = self._allocate((32, 248320))

    def close(self):
        if self.native is not None and self.native.worker is not None:
            for handle in reversed(self.handles):
                self.native.worker.free_tensor(handle)
        self.handles.clear()
        self.native = None

    def _upload(self, tensor, handle):
        tensor = tensor.contiguous()
        self.native.worker.copy_to(handle.data_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())

    def _run(self, name, arguments):
        self.native.worker.run(self.programs[name], *arguments, config=self.native.config)

    def propose(self, request, anchor, width):
        native = self.native
        if native is None or not 2 <= width <= self.block_size:
            raise ValueError("a started draft and an active block width in 2..K are required")
        resident = native.requests[request.request_id]
        past = resident.pages.length
        expected = len(request.prompt_ids) + len(request.generated) - 1
        if past != expected or past + width > self.context:
            raise RuntimeError("draft history boundary does not match the request anchor")
        start = time.monotonic()
        ids = torch.full((32,), self.head.mask_token_id, dtype=torch.int32)
        ids[0] = anchor
        self._upload(ids, self.ids)
        # Positions outside the active width are masked; clamping only pads RoPE.
        positions = torch.arange(past, past + 32).clamp_max(self.context - 1)
        self._upload(native.rope["cos"][positions], self.cos)
        self._upload(native.rope["sin"][positions], self.sin)
        table = torch.zeros(32, dtype=torch.int32)
        table[:len(resident.pages.pages)] = torch.tensor(resident.pages.pages, dtype=torch.int32)
        self._upload(table, self.table)
        mask = torch.full((16, 512), -10000.0)
        mask[:, :past + width] = 0
        self._upload(mask, self.mask)
        worker, banks = native.worker, native.banks
        self._run("embedding", [self.ids, banks[0], self.hidden[0]])
        for layer in range(64):
            # Stream only attention weights, keeping the target's MLP/norm/LM
            # allocations shared. This initial path prioritizes bounded memory.
            host = self.head.banks(layer)
            if layer % 4 != 3:
                host[5] = host[5].float()  # BF16 initialization also promotes exactly.
            weights = []
            try:
                for value in host:
                    weights.append(worker.alloc_tensor(tuple(value.shape), value.dtype, init=value))
                current, output = self.hidden[layer % 2], self.hidden[1 - layer % 2]
                if layer % 4 != 3:
                    self._run("gdn", [current, *banks[3:8], *weights,
                                      *native.resident[:2], output, layer, resident.slot])
                else:
                    self._run("full_prepare", [current, banks[3], *weights[:5], self.cos, self.sin,
                                               self.query, self.key, self.value, self.gate, layer])
                    self._run("full", [current, *banks[4:8], weights[-1], self.query, self.key,
                                       self.value, self.gate, *native.resident[2:], self.table,
                                       self.mask, output, layer, past, width])
            finally:
                for value in reversed(weights):
                    worker.free_tensor(value)
        self._run("finish", [self.hidden[0], banks[1], banks[2], self.logits])
        logits = torch.empty((32, 248320), dtype=torch.bfloat16)
        worker.copy_from(logits.data_ptr(), self.logits.data_ptr, logits.numel() * logits.element_size())
        if not logits[:width - 1].isfinite().all():
            raise FloatingPointError("nonfinite PyPTO draft logits")
        self.rounds += 1
        self.last_seconds = time.monotonic() - start
        return logits[:width - 1].float().argmax(dim=-1).tolist()


def make_runner(source, model_root, device, head, **callbacks):
    """Create a native runner with additional graphs; leave native source intact."""
    runtime = native_module(source, "runtime")
    draft = NeuralDraft(source, head)

    class PreparedRunner(runtime.ModelRunner):
        def start(self):
            if self.worker is not None:
                raise RuntimeError("model runner is already started")
            samples = runtime.single_meta_samples()
            self.prefill = runtime.prefill_fwd.compile(*samples, 1, config=self.config)
            self.decode = runtime.decode_fwd.compile(*samples, 1, config=self.config)
            programs = draft.compile(samples, self.config)
            self.rope = runtime.rotary_tables(self.model_root)
            if any(value.shape != (512, 64) for value in self.rope.values()):
                raise ValueError("invalid native RoPE table")
            self._buffers()
            self.worker = runtime.DistributedWorker([self.prefill, self.decode, *programs], config=self.config)
            try:
                for _, host in runtime.TextWeights(self.model_root).banks():
                    self.banks.append(self.worker.alloc_tensor(tuple(host.shape), host.dtype, init=host))
                pool_shape = (16, 4, runtime.POOL_BLOCKS, 16, 256)
                for shape, dtype in (((8, *runtime.CONV_SHAPE), torch.bfloat16),
                                     ((8, *runtime.RECURRENT_SHAPE), torch.float32),
                                     (pool_shape, torch.bfloat16), (pool_shape, torch.bfloat16)):
                    self.resident.append(self.worker.alloc_tensor(shape, dtype, init=torch.zeros(shape, dtype=dtype)))
                self.traces = self.worker.alloc_tensor((8, 66, 16, 5120), torch.bfloat16)
                self.logits = self.worker.alloc_tensor((8, 16, 248320), torch.bfloat16)
                self._sample_memory()
            except BaseException:
                self.close()
                raise

    native = PreparedRunner(model_root, device, **callbacks)
    return OrthrusRunner(native, draft, native_module(source, "schedule").BatchSlot)


def make_engine(source, model_root, device, head):
    engine = native_module(source, "engine").InferenceEngine(model_root, device, max_active=1, prefix_cache=False)
    engine.runner = make_runner(source, model_root, device, head, on_cache_event=engine._on_cache)
    engine.scheduler.runner = engine.runner
    return engine
