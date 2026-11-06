"""K32 neural draft executor sharing a native worker and target backbone banks."""

import time

import torch

from .baseline import native_module
from .greedy import host_greedy_tokens
from .programs import make_programs
from .target import OrthrusRunner, SnapshotPool
from .weights import DraftWeights


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
        "full_layer": [hidden, *native_samples[4:9], *full, cos, sin, q, k, v, gate,
                       *native_samples[31:33], empty((32,), torch.int32), empty((16, 512), torch.float32), result, 3, 0, 32],
        "finish": [hidden, native_samples[2], native_samples[3], empty((32, 248320))],
    }


class NeuralDraft:
    context = 512

    def __init__(self, source, head, *, head_storage="auto", head_transfer="individual", attention_backend="native"):
        if attention_backend not in {"native", "fused", "paged"}:
            raise ValueError("attention backend must be native, fused or paged")
        if getattr(getattr(head, "layout", None), "size", 1) != 1:
            raise ValueError("rank-local heads require a full-model TP executor; the pinned native executor is TP1")
        self.source, self.head = source, head
        self.attention_backend = attention_backend
        self.block_size = head.block_size
        self.programs, self.handles = {}, []
        self.native = None
        self.rounds = 0
        self.last_seconds = None
        self.last_timings = {}
        self.weights = DraftWeights(head, head_storage, head_transfer)

    def compile(self, samples, config):
        inputs = metadata(self.head, samples)
        for name, program in make_programs(self.source, attention_backend=self.attention_backend).items():
            print(f"compiling Orthrus {name}", flush=True)
            self.programs[name] = program.compile(*inputs[name], config=config)
        return list(self.programs.values())

    def prepare_host(self):
        self.head.prepare()
        self.host_logits = torch.empty((32, 248320), dtype=torch.bfloat16).share_memory_()

    def _allocate(self, shape, dtype=torch.bfloat16):
        handle = self.native.worker.alloc_tensor(shape, dtype)
        self.handles.append(handle)
        return handle

    def start(self, native):
        expected = ({"embedding", "gdn", "full_layer", "finish"} if self.attention_backend == "paged" else
                    {"embedding", "gdn", "full_prepare", "full", "finish"})
        if self.native is not None or set(self.programs) != expected:
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
        self.weights.start(native.worker)

    @property
    def storage(self):
        return self.weights.storage

    def close(self):
        try:
            self.weights.close()
            if self.native is not None and self.native.worker is not None:
                for handle in reversed(self.handles):
                    self.native.worker.free_tensor(handle)
        finally:
            self.handles.clear()
            self.native = None

    def _upload(self, tensor, handle):
        tensor = tensor.contiguous()
        self.native.worker.copy_to(handle.data_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())

    def _run(self, name, arguments):
        self.last_timings["compute_dispatches"] += 1
        start = time.monotonic()
        try:
            self.native.worker.run(self.programs[name], *arguments, config=self.native.config)
        finally:
            elapsed = time.monotonic() - start
            self.last_timings["device_seconds"] += elapsed
            label = name + "_seconds"
            self.last_timings[label] = self.last_timings.get(label, 0.0) + elapsed

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
        self.last_timings = dict(device_seconds=0.0, weights_seconds=0.0, compute_dispatches=0)
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
        self.last_timings["inputs_seconds"] = time.monotonic() - start
        worker, banks = native.worker, native.banks
        self._run("embedding", [self.ids, banks[0], self.hidden[0]])
        for layer in range(64):
            weight_start = time.monotonic()
            weights = self.weights.for_layer(layer)
            self.last_timings["weights_seconds"] += time.monotonic() - weight_start
            current, output = self.hidden[layer % 2], self.hidden[1 - layer % 2]
            if layer % 4 != 3:
                self._run("gdn", [current, *banks[3:8], *weights,
                                  *native.resident[:2], output, layer, resident.slot])
            elif self.attention_backend == "paged":
                self._run("full_layer", [current, *banks[3:8], *weights, self.cos, self.sin,
                                        self.query, self.key, self.value, self.gate, *native.resident[2:],
                                        self.table, self.mask, output, layer, past, width])
            else:
                self._run("full_prepare", [current, banks[3], *weights[:5], self.cos, self.sin,
                                           self.query, self.key, self.value, self.gate, layer])
                self._run("full", [current, *banks[4:8], weights[-1], self.query, self.key,
                                   self.value, self.gate, *native.resident[2:], self.table,
                                   self.mask, output, layer, past, width])
        self._run("finish", [self.hidden[0], banks[1], banks[2], self.logits])
        copy_start = time.monotonic()
        logits = self.host_logits[:width - 1]
        worker.copy_from(logits.data_ptr(), self.logits.data_ptr, logits.numel() * logits.element_size())
        self.last_timings["readback_seconds"] = time.monotonic() - copy_start
        select_start = time.monotonic()
        tokens = host_greedy_tokens(logits)
        self.last_timings["select_seconds"] = time.monotonic() - select_start
        self.rounds += 1
        self.last_seconds = time.monotonic() - start
        self.last_timings["total_seconds"] = self.last_seconds
        return tokens


def make_runner(source, model_root, device, head, *, head_storage="auto", head_transfer="individual",
                attention_backend="native", **callbacks):
    """Create a native runner with additional graphs; leave native source intact."""
    runtime = native_module(source, "runtime")
    draft = NeuralDraft(source, head, head_storage=head_storage, head_transfer=head_transfer,
                        attention_backend=attention_backend)

    class PreparedRunner(runtime.ModelRunner):
        def start(self):
            if self.worker is not None:
                raise RuntimeError("model runner is already started")
            samples = runtime.single_meta_samples()
            self.prefill = runtime.prefill_fwd.compile(*samples, 1, config=self.config)
            self.decode = runtime.decode_fwd.compile(*samples, 1, config=self.config)
            programs = draft.compile(samples, self.config)
            draft.prepare_host()
            self.rope = runtime.rotary_tables(self.model_root)
            if any(value.shape != (512, 64) for value in self.rope.values()):
                raise ValueError("invalid native RoPE table")
            self._buffers()
            self.verification_logits = torch.empty((16, 248320), dtype=torch.bfloat16).share_memory_()
            self.verification_states = SnapshotPool(self.zero_conv.shape, self.zero_recurrent.shape)
            from pypto.runtime import ReadOnlyHostTensor
            self.worker = runtime.DistributedWorker([self.prefill, self.decode, *programs], config=self.config,
                inherited_host_tensors=[ReadOnlyHostTensor(head.shared_weights), draft.host_logits,
                                        self.verification_logits, self.verification_states.storage])
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


def make_engine(source, model_root, device, head, *, head_storage="auto", head_transfer="individual",
                attention_backend="native"):
    class Engine(native_module(source, "engine").InferenceEngine):
        def snapshot(self):
            result = super().snapshot()
            runner = self.runner
            result["orthrus"] = dict(block_size=runner.draft.block_size, rounds=runner.rounds,
                                     accepted=runner.accepted, restores=runner.restores,
                                     neural_draft_rounds=runner.draft.rounds,
                                     last_draft_seconds=runner.draft.last_seconds,
                                     head_storage=runner.draft.storage,
                                     head_transfer=runner.draft.weights.transfer,
                                     attention_backend=runner.draft.attention_backend,
                                     head_preparation_seconds=head.preparation_seconds,
                                     last_draft_timings=dict(runner.draft.last_timings))
            return result

    engine = Engine(model_root, device, max_active=1, prefix_cache=False)
    engine.runner = make_runner(source, model_root, device, head, head_storage=head_storage,
                                head_transfer=head_transfer,
                                attention_backend=attention_backend,
                                on_cache_event=engine._on_cache)
    engine.scheduler.runner = engine.runner
    return engine
