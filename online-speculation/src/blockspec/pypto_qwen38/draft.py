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
    inputs = {
        "embedding": [empty((32,), torch.int32), native_samples[1], hidden],
        "gdn": [hidden, *native_samples[4:9], *gdn, *native_samples[29:31], result, 0, 0],
        "full_prepare": [hidden, native_samples[4], *full[:5], cos, sin, q, k, v, gate, 3],
        "full": [hidden, *native_samples[5:9], full[-1], q, k, v, gate, *native_samples[31:33],
                 empty((32,), torch.int32), empty((16, 512), torch.float32), result, 3, 0, 32],
        "full_layer": [hidden, *native_samples[4:9], *full, cos, sin, q, k, v, gate,
                       *native_samples[31:33], empty((32,), torch.int32), empty((16, 512), torch.float32), result, 3, 0, 32],
        "finish": [hidden, native_samples[2], native_samples[3], empty((32, 248320))],
    }
    group_weights = [empty(value.shape, torch.float32 if index == 5 else value.dtype)
                     for layer in range(3) for index, value in enumerate(head.banks(layer))]
    inputs["group"] = [hidden, *native_samples[4:9], *group_weights, *full, cos, sin, q, k, v, gate,
                       *native_samples[29:33], empty((32,), torch.int32), empty((16, 512), torch.float32),
                       empty((32, 5120)), empty((32, 5120)), result, 0, 0, 0, 32]
    inputs["group_prefix"] = [hidden, *native_samples[4:9], *group_weights, *native_samples[29:31],
                               empty((32, 5120)), empty((32, 5120)), result, 60, 0]
    inputs["hidden_capture"] = [hidden,result]
    return inputs


class NeuralDraft:
    context = 512

    def __init__(self, source, head, *, head_storage="auto", head_transfer="individual", attention_backend="native", group_size=1, gdn_backend="native"):
        if attention_backend not in {"native", "fused", "paged"}:
            raise ValueError("attention backend must be native, fused or paged")
        if gdn_backend not in {"native", "vector"}:
            raise ValueError("gdn_backend must be native or vector")
        if (type(group_size) is not int or group_size not in {1, 4}
                or group_size == 4 and (attention_backend != "paged" or head_transfer != "packed")):
            raise ValueError("group_size=4 requires paged attention and packed head transfer")
        if getattr(getattr(head, "layout", None), "size", 1) != 1:
            raise ValueError("rank-local heads require a full-model TP executor; the pinned native executor is TP1")
        self.source, self.head = source, head
        self.attention_backend = attention_backend
        self.gdn_backend = gdn_backend
        self.group_size = group_size
        self.block_size = head.block_size
        self.programs, self.handles = {}, []
        self.native = None
        self.rounds = 0
        self.last_seconds = None
        self.last_timings = {}
        self.learner = None
        self.weights = DraftWeights(head, head_storage, head_transfer, group_size=group_size)

    def compile(self, samples, config):
        inputs = metadata(self.head, samples)
        for name, program in make_programs(self.source, attention_backend=self.attention_backend, group_size=self.group_size,
                                         gdn_backend=self.gdn_backend, online=self.learner is not None,
                                         online_layers=self.learner.options.last_layers if self.learner is not None else 1).items():
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
        if self.group_size == 4:
            expected = {"embedding", "group", "finish"}
            if self.learner is not None:
                expected |= {"full_layer"}
                if self.learner.options.last_layers > 1:
                    expected |= {"gdn", "hidden_capture"}
                else:
                    expected |= {"group_prefix"}
        if self.native is not None or set(self.programs) != expected:
            raise RuntimeError("draft programs must be registered before starting the native worker")
        self.native = native
        self.hidden = [self._allocate((32, 5120)), self._allocate((32, 5120))]
        self.group_scratch = [self._allocate((32, 5120)), self._allocate((32, 5120))] if self.group_size == 4 else []
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
        serving = getattr(native, "device_serving", None)
        if serving is not None:
            serving.prepare(self, resident.pages.pages, anchor, past, width)
        else:
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
        feedback = self.learner.begin(self, past, width, slot=resident.slot) if self.learner is not None else None
        banks = native.banks
        self._run("embedding", [self.ids, banks[0], self.hidden[0]])
        for layer in range(0, 64, self.group_size):
            weight_start = time.monotonic()
            weights = self.weights.for_group(layer) if self.group_size == 4 else self.weights.for_layer(layer)
            self.last_timings["weights_seconds"] += time.monotonic() - weight_start
            index = layer // self.group_size % 2
            current, output = self.hidden[index], self.hidden[1 - index]
            if self.group_size == 4:
                capture = self.learner.graph.capture_layer if self.learner is not None and self.learner.options.last_layers > 1 else 63
                if layer <= capture < layer+4 and feedback is not None:
                    if self.learner.options.last_layers > 1:
                        for offset in range(3):
                            if layer+offset == self.learner.graph.capture_layer:
                                self._run("hidden_capture",[current,feedback.inputs["hidden"]])
                                current = feedback.inputs["hidden"]
                            following = self.group_scratch[offset%2]
                            self._run("gdn",[current,*banks[3:8],*weights[offset],*native.resident[:2],
                                             following,layer+offset,resident.slot])
                            current = following
                        if capture == layer+3:
                            self._run("hidden_capture",[current,feedback.inputs["hidden"]])
                            current = feedback.inputs["hidden"]
                        self._run("full_layer",[current,*banks[3:8],*weights[3],self.cos,self.sin,
                            self.query,self.key,self.value,self.gate,*native.resident[2:],self.table,
                            self.mask,output,layer+3,past,width])
                        continue
                    self._run("group_prefix", [current, *banks[3:8],
                        *(handle for group in weights[:3] for handle in group), *native.resident[:2],
                        *self.group_scratch, feedback.inputs["hidden"], layer, resident.slot])
                    self._run("full_layer", [feedback.inputs["hidden"], *banks[3:8], *weights[3], self.cos, self.sin,
                        self.query, self.key, self.value, self.gate, *native.resident[2:], self.table,
                        self.mask, output, 63, past, width])
                    continue
                flat = [handle for group in weights for handle in group]
                self._run("group", [current, *banks[3:8], *flat, self.cos, self.sin,
                                    self.query, self.key, self.value, self.gate, *native.resident,
                                    self.table, self.mask, *self.group_scratch, output, layer, resident.slot, past, width])
            elif layer % 4 != 3:
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
        select_start = time.monotonic()
        if serving is not None:
            tokens = serving.select(self.logits, 0, width - 1)
            self.last_timings["readback_seconds"] = 0.0
        else:
            copy_start = time.monotonic()
            logits = self.read_logits(width - 1)
            self.last_timings["readback_seconds"] = time.monotonic() - copy_start
            select_start = time.monotonic()
            tokens = host_greedy_tokens(logits)
        self.last_timings["select_seconds"] = time.monotonic() - select_start
        self.rounds += 1
        self.last_seconds = time.monotonic() - start
        self.last_timings["total_seconds"] = self.last_seconds
        return tokens

    def read_logits(self, rows):
        """诊断时显式读取完整 logits；正常推理仅回传 token 和有限值标志。"""
        if type(rows) is not int or not 1 <= rows <= 32 or self.native is None:
            raise ValueError("a started draft and 1..32 logits rows required")
        logits = self.host_logits[:rows]
        self.native.worker.copy_from(logits.data_ptr(), self.logits.data_ptr, logits.numel() * logits.element_size())
        return logits


def make_runner(source, model_root, device, head, *, head_storage="auto", head_transfer="individual",
                attention_backend="native", group_size=1, gdn_backend="native", device_serving=True,
                weight_backend="pypto", online_config=None, **callbacks):
    """Create a native runner with additional graphs; leave native source intact."""
    runtime = native_module(source, "runtime")
    if weight_backend not in {"pypto", "host"}:
        raise ValueError("weight backend must be pypto or host")
    draft = NeuralDraft(source, head, head_storage=head_storage, head_transfer=head_transfer,
                        attention_backend=attention_backend, group_size=group_size, gdn_backend=gdn_backend)
    learner = None
    if online_config is not None:
        if (attention_backend, group_size, gdn_backend, device_serving) != ("paged", 4, "vector", True):
            raise ValueError("online updates require paged attention, vector GDN, groups of four and device serving")
        from .online import OnlineLearner
        if online_config.last_layers > 1:
            from .online_suffix import SuffixLearner
            learner = draft.learner = SuffixLearner(head,online_config)
        else:
            learner = draft.learner = OnlineLearner(head, online_config)

    class PreparedRunner(runtime.ModelRunner):
        device_serving = None

        def start(self):
            if self.worker is not None:
                raise RuntimeError("model runner is already started")
            if weight_backend == "pypto":
                # 原生交付包的文件身份校验先执行，避免路径有误时仍准备 head 和编译训练图。
                runtime.TextWeights(self.model_root)
                head.prepare_on_device(self.config)
            samples = runtime.single_meta_samples()
            self.prefill = runtime.prefill_fwd.compile(*samples, 1, config=self.config)
            self.decode = runtime.decode_fwd.compile(*samples, 1, config=self.config)
            programs = draft.compile(samples, self.config)
            if learner is not None:
                programs += learner.compile(samples, self.config)
            draft.prepare_host()
            self.rope = runtime.rotary_tables(self.model_root)
            if any(value.shape != (512, 64) for value in self.rope.values()):
                raise ValueError("invalid native RoPE table")
            self._buffers()
            self.verification_logits = torch.empty((16, 248320), dtype=torch.bfloat16).share_memory_()
            self.verification_states = SnapshotPool(self.zero_conv.shape, self.zero_recurrent.shape)
            if device_serving:
                from .serving import DeviceServing
                self.device_serving = DeviceServing(head.mask_token_id)
                programs += self.device_serving.compile(self.config, self.zero_conv.shape, self.zero_recurrent.shape)
            base_loading = None
            if weight_backend == "pypto":
                from .base_loading import BaseLoading
                base_loading = BaseLoading(runtime, self.model_root, self.config)
                programs += list(base_loading.registry.programs.values())
            from pypto.runtime import ReadOnlyHostTensor
            if learner is not None:
                count = sum(sum((path/"kernel_config.py").exists() for path in (program.output_dir/"next_levels").iterdir())
                            for program in (self.prefill, self.decode, *programs))
                print(f"online runtime: {count} registered chip entries", flush=True)
                if count > 64:
                    raise RuntimeError(f"online program set exceeds pinned runtime capacity: {count} > 64")
            self.worker = runtime.DistributedWorker([self.prefill, self.decode, *programs], config=self.config,
                inherited_host_tensors=[ReadOnlyHostTensor(head.shared_weights), draft.host_logits,
                                        self.verification_logits, self.verification_states.storage,
                                        *([learner.bridge.scratch] if learner is not None and learner.dedicated else []),
                                        *(self.device_serving.host_tensors if self.device_serving else [])])
            try:
                if learner is not None:
                    print("online runtime: loading shared target banks", flush=True)
                if base_loading is not None:
                    for handle in base_loading.banks(self.worker):
                        self.banks.append(handle)
                else:
                    for _, host in runtime.TextWeights(self.model_root).banks():
                        self.banks.append(self.worker.alloc_tensor(tuple(host.shape), host.dtype, init=host))
                pool_shape = (16, 4, runtime.POOL_BLOCKS, 16, 256)
                for shape, dtype in (((8, *runtime.CONV_SHAPE), torch.bfloat16),
                                     ((8, *runtime.RECURRENT_SHAPE), torch.float32),
                                     (pool_shape, torch.bfloat16), (pool_shape, torch.bfloat16)):
                    self.resident.append(self.worker.alloc_tensor(shape, dtype, init=torch.zeros(shape, dtype=dtype)))
                self.traces = self.worker.alloc_tensor((8, 66, 16, 5120), torch.bfloat16)
                self.logits = self.worker.alloc_tensor((8, 16, 248320), torch.bfloat16)
                if self.device_serving is not None:
                    self.device_serving.start(self)
                self._sample_memory()
            except BaseException:
                self.close()
                raise

        def close(self):
            try:
                try:
                    if learner is not None:
                        learner.close()
                finally:
                    if self.device_serving is not None:
                        self.device_serving.close()
            finally:
                super().close()

        def run_batch(self, slots):
            if self.device_serving is None:
                return super().run_batch(slots)
            return self.device_serving.run_batch(runtime, slots)

        def run_verifier(self, slots):
            return self.device_serving.run_batch(runtime, slots, select=False)

        def _compact(self, slots):
            if self.device_serving is None:
                return super()._compact(slots)
            return self.device_serving.compact(slots)

    native = PreparedRunner(model_root, device, **callbacks)
    native.weight_backend = weight_backend
    native.learner = learner
    return OrthrusRunner(native, draft, native_module(source, "schedule").BatchSlot)


def make_engine(source, model_root, device, head, *, head_storage="auto", head_transfer="individual",
                attention_backend="native", group_size=1, gdn_backend="native", device_serving=True, weight_backend="pypto",
                online_config=None):
    class Engine(native_module(source, "engine").InferenceEngine):
        def start(self):
            if online_config is None:
                return super().start()
            # 在线入口还需编译/启动训练 worker；不受原生纯推理的十分钟启动窗口限制。
            self.scheduler.thread.start()
            if not self.scheduler.ready.wait(timeout=1800):
                raise RuntimeError("online model runner startup timed out")
            if self.scheduler.error is not None:
                raise RuntimeError("online model runner failed to start") from self.scheduler.error

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
                                     gdn_backend=runner.draft.gdn_backend,
                                     serving_backend="device" if runner.native.device_serving is not None else "host",
                                     weight_backend=runner.native.weight_backend,
                                     head_preparation_backend=head.preparation_backend,
                                     draft_group_size=runner.draft.group_size,
                                     head_preparation_seconds=head.preparation_seconds,
                                     last_draft_timings=dict(runner.draft.last_timings))
            if runner.native.learner is not None:
                learner = runner.native.learner
                result["orthrus"]["online"] = dict(last_layers=online_config.last_layers, rounds=learner.rounds, updates=learner.updates,
                    version=runner.draft.weights.version, last_result=learner.last_result,
                    learner_device=online_config.learner_device)
            return result

    engine = Engine(model_root, device, max_active=1, prefix_cache=False)
    engine.runner = make_runner(source, model_root, device, head, head_storage=head_storage,
                                head_transfer=head_transfer,
                                attention_backend=attention_backend,
                                group_size=group_size,
                                gdn_backend=gdn_backend,
                                device_serving=device_serving,
                                weight_backend=weight_backend,
                                online_config=online_config,
                                on_cache_event=engine._on_cache)
    engine.scheduler.runner = engine.runner
    return engine
