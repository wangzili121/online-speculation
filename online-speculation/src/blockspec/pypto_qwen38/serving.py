"""设备端推理辅助计算。文件读取、页分配与请求调度仍由主机负责。"""

import ctypes
from dataclasses import dataclass
from math import prod
import threading
import time

import torch

from .serving_programs import make_draft_inputs, make_greedy, make_state_copy, make_target_inputs


@dataclass(frozen=True)
class DeviceState:
    owner: object
    index: int


class DeviceServing:
    capacity = 2

    def __init__(self, mask_token):
        self.mask_token = mask_token
        self.programs, self.handles, self.busy = {}, [], set()
        self.native = None
        self.lock = threading.Lock()
        # fork 前注册的小型主机缓冲，仅传页号和最终 token，不传完整 logits。
        self.host_result = torch.empty((2, 32), dtype=torch.int32).share_memory_()
        self.host_pages = torch.empty((32,), dtype=torch.int32).share_memory_()
        self.result_values = (ctypes.c_int32 * 64).from_address(self.host_result.data_ptr())
        self.page_values = (ctypes.c_int32 * 32).from_address(self.host_pages.data_ptr())

    def compile(self, config, conv_shape, recurrent_shape):
        empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
        conv, rec = empty((8, *conv_shape)), empty((8, *recurrent_shape), torch.float32)
        saved_conv, saved_rec = empty((self.capacity, *conv_shape)), empty((self.capacity, *recurrent_shape), torch.float32)
        self.shapes = conv_shape, recurrent_shape
        inputs = dict(
            draft_select=[empty((32, 248320)), self.host_result, 0, 31],
            target_select=[empty((8, 16, 248320)), self.host_result, 0, 16],
            save=[conv, rec, saved_conv, saved_rec, 0, 0],
            restore=[saved_conv, saved_rec, conv, rec, 0, 0],
            inputs=[empty((512, 64)), empty((512, 64)), empty((1, 512), torch.int32), self.host_pages,
                    empty((32,), torch.int32), empty((32, 64)), empty((32, 64)), empty((32,), torch.int32),
                    empty((16, 512), torch.float32), 0, 0, 32],
            target_inputs=[empty((512, 64)), empty((512, 64)), empty((1, 512), torch.int32),
                           empty((8,), torch.int32), empty((8,), torch.int32),
                           empty((8, 16, 64)), empty((8, 16, 64)), empty((8, 16, 512), torch.float32)])
        copy = dict(conv_elements=prod(conv_shape), recurrent_elements=prod(recurrent_shape))
        programs = dict(draft_select=make_greedy()[1], target_select=make_greedy(source_rows=128)[1],
                        save=make_state_copy(**copy, source_slots=8, destination_slots=self.capacity)[1],
                        restore=make_state_copy(**copy, source_slots=self.capacity, destination_slots=8)[1],
                        inputs=make_draft_inputs(mask_token=self.mask_token)[1], target_inputs=make_target_inputs()[1])
        for name, program in programs.items():
            print(f"compiling Orthrus serving {name}", flush=True)
            self.programs[name] = program.compile(*inputs[name], config=config)
        return list(self.programs.values())

    @property
    def host_tensors(self):
        return [self.host_result, self.host_pages]

    def start(self, native):
        self.native = native
        self.result = self.allocate(tuple(self.host_result.shape), self.host_result.dtype)
        self.rope = [self.allocate(tuple(native.rope[key].shape), native.rope[key].dtype, init=native.rope[key])
                     for key in ("cos", "sin")]
        columns = torch.arange(512, dtype=torch.int32).reshape(1, 512)
        self.columns = self.allocate((1, 512), torch.int32, init=columns)
        self.states = [self.allocate((self.capacity, *shape), dtype)
                       for shape, dtype in zip(self.shapes, (torch.bfloat16, torch.float32), strict=True)]
        self.target_cos, self.target_sin = [self.allocate((8, 16, 64), torch.bfloat16) for _ in range(2)]
        self.target_mask = self.allocate((8, 16, 512), torch.float32)

    def allocate(self, shape, dtype, **options):
        handle = self.native.worker.alloc_tensor(shape, dtype, **options)
        self.handles.append(handle)
        return handle

    def close(self):
        if self.native is not None and self.native.worker is not None:
            for handle in reversed(self.handles):
                self.native.worker.free_tensor(handle)
        self.handles.clear()
        self.native = None

    def run(self, name, *arguments):
        if self.native is None or self.native.worker is None:
            raise RuntimeError("device serving buffers are not started")
        self.native.worker.run(self.programs[name], *arguments, config=self.native.config)

    def select(self, source, begin, width, *, target=False):
        rows = 128 if target else 32
        if type(begin) is not int or type(width) is not int or not 1 <= width <= 32 or not 0 <= begin <= rows - width:
            raise ValueError("invalid active logits rows")
        self.run("target_select" if target else "draft_select", source, self.result, begin, width)
        self.native.worker.copy_from(self.host_result.data_ptr(), self.result.data_ptr, self.host_result.numel() * 4)
        tokens = []
        for row in range(width):
            if self.result_values[32 + row] != 1:
                raise FloatingPointError("nonfinite target logits" if target else "nonfinite draft logits")
            tokens.append(self.result_values[row])
        return tokens

    def prepare(self, draft, pages, anchor, past, width):
        for i in range(32):
            self.page_values[i] = pages[i] if i < len(pages) else 0
        self.run("inputs", *self.rope, self.columns, self.host_pages,
                 draft.ids, draft.cos, draft.sin, draft.table, draft.mask, anchor, past, width)

    def acquire(self, slot):
        with self.lock:
            index = next((i for i in range(self.capacity) if i not in self.busy), None)
            if index is None:
                return None
            self.busy.add(index)
        try:
            self.run("save", *self.native.resident[:2], *self.states, slot, index)
            return DeviceState(self, index)
        except BaseException:
            self.release(index)
            raise

    def release(self, index):
        with self.lock:
            self.busy.discard(index)

    def restore(self, state, slot):
        if state.owner is not self or state.index not in self.busy:
            raise ValueError("snapshot does not belong to this live device pool")
        self.run("restore", *self.states, *self.native.resident[:2], state.index, slot)

    def run_batch(self, runtime, slots, *, select=True):
        """沿用固定版本的 cache/回调协议，输入计算和 logits 归约留在设备。"""
        native = self.native
        if native is None or native.worker is None or len(slots) != 1:
            raise RuntimeError("device serving requires one active request")
        if native.allow_prefix_hits:
            raise RuntimeError("prefix reuse requires separate device serving acceptance")
        native._compact(slots)
        ids = (ctypes.c_int32 * (8 * 16)).from_address(native.ids.data_ptr())
        counts = (ctypes.c_int32 * 8).from_address(native.counts.data_ptr())
        offsets = (ctypes.c_int32 * 8).from_address(native.offsets.data_ptr())
        tables = (ctypes.c_int32 * (8 * 32)).from_address(native.tables.data_ptr())
        # CPU 只填调度器提供的整数元数据，不构造 RoPE/mask 张量。
        for values in (ids, counts, offsets, tables):
            ctypes.memset(ctypes.addressof(values), 0, ctypes.sizeof(values))
        slot = slots[0]
        count, position = len(slot.token_ids), slot.position
        resident = native.requests[slot.request.request_id]
        if not 1 <= count <= 16 or position < 0 or position + count > 512 or resident.pages.length != position:
            raise RuntimeError("request KV length and scheduled position disagree")
        try:
            native.cache.reserve(resident.pages, position + count)
        except runtime.CacheCapacityError as error:
            raise runtime.RequestCapacityError(slot.request, str(error)) from error
        counts[0], offsets[0] = count, position
        for i, token in enumerate(slot.token_ids):
            ids[i] = token
        for i, page in enumerate(resident.pages.pages):
            tables[i] = page
        self.run("target_inputs", *self.rope, self.columns, native.counts, native.offsets,
                 self.target_cos, self.target_sin, self.target_mask)
        graph = native.prefill if slot.phase == "prefill" else native.decode
        if native.on_activation is not None:
            native.on_activation("graph_start", dict(time_ns=time.monotonic_ns(), device=native.device_id, phases=[slot.phase]))
        native.worker.run(graph, native.ids, *native.banks, self.target_cos, self.target_sin, self.target_mask,
                          native.counts, native.offsets, *native.resident, native.tables, native.traces, native.logits, 1,
                          config=native.config)
        if native.on_activation is not None:
            native.on_activation("graph_end", dict(time_ns=time.monotonic_ns(), device=native.device_id))
        native.cache.commit(resident.pages, position + count)
        output = None
        if select and (slot.phase != "prefill" or slot.completes_prefill):
            output = self.select(native.logits, count - 1, 1, target=True)[0]
        native._sample_memory()
        return [output]

    def compact(self, slots):
        """单请求换槽也在设备内复制；不改变框架的页所有权规则。"""
        native = self.native
        if len(slots) != 1:
            raise ValueError("device state compaction requires one active request")
        request_id = slots[0].request.request_id
        if native.slot_ids[0] == request_id:
            return
        resident = native.requests[request_id]
        if resident.slot != 0:
            saved = self.acquire(resident.slot)
            if saved is None:
                raise RuntimeError("device snapshot pool is full during slot compaction")
            try:
                self.restore(saved, 0)
            finally:
                self.release(saved.index)
        native.slot_ids = [request_id] + [None] * 7
        resident.slot = 0
