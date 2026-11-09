"""Keep head parameters resident when possible; reuse buffers when streaming."""


class DraftWeights:
    def __init__(self, head, mode="auto", transfer="individual", *, worker_id=0, group_size=1):
        if mode not in {"auto", "resident", "stream"}:
            raise ValueError("head storage must be auto, resident or stream")
        if transfer not in {"individual", "packed"}:
            raise ValueError("head transfer must be individual or packed")
        if type(worker_id) is not int or worker_id < 0:
            raise ValueError("head worker_id must be a nonnegative integer")
        if type(group_size) is not int or group_size not in {1, 4} or group_size == 4 and transfer != "packed":
            raise ValueError("head group_size must be 1 or 4; groups of four require packed transfer")
        self.worker_id = worker_id
        self.group_size = group_size
        self.head, self.mode = head, mode
        self.transfer, self.packed = transfer, {}
        self.worker = None
        self.handles, self.resident, self.buffers = [], {}, {}
        self.storage = None
        self.overrides = {}
        self.version = 0

    def publish(self, layer, values, *, expected_version):
        """在同步起草轮次之间替换整层引用；不原地改正在被读取的 bank。"""
        if self.worker is None or expected_version != self.version or layer != 63:
            raise RuntimeError("a started last-layer head and its current version required")
        values = tuple(values)
        expected = self.head.banks(layer)
        if len(values) != len(expected) or any(tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype
                                              for a, b in zip(values, expected, strict=True)):
            raise ValueError("published head layout changed")
        # 所有检查均在提交前完成；bank 的 owner 由 learner 保留至下一轮安全边界。
        self.overrides = {**self.overrides, layer: values}
        self.version += 1

    def _allocate(self, value, *, initialize):
        handle = self.worker.alloc_tensor(
            tuple(value.shape), value.dtype, init=value if initialize else None, worker_id=self.worker_id
        )
        self.handles.append(handle)
        return handle

    def start(self, worker):
        if self.worker is not None or self.head.shared_weights is None:
            raise RuntimeError("prepare the head before starting its device weights")
        self.worker = worker
        free, _ = worker.device_memory_info(worker_id=self.worker_id)
        required = self.head.shared_weights.numel() + (256 << 20)
        if self.mode == "resident" and required > free:
            raise RuntimeError(
                f"resident head requires {required} free bytes, available {free}"
            )
        self.storage = (
            "resident"
            if self.mode == "resident" or self.mode == "auto" and required <= free
            else "stream"
        )
        if self.storage == "resident":
            for layer in range(64):
                self.resident[layer] = tuple(
                    self._allocate(value, initialize=True)
                    for value in self.head.banks(layer)
                )
        else:
            # GDN 与 Full Attention 各用一组缓冲，不在每层/每轮重新分配。
            minimum = sum(
                value.numel() * value.element_size()
                for layer in (range(4) if self.group_size == 4 else (0, 3))
                for value in self.head.banks(layer)
            ) + (256 << 20)
            if minimum > free:
                raise RuntimeError(
                    f"stream head buffers require {minimum} free bytes, available {free}"
                )
            if self.transfer == "packed":
                self._prepare_packed_stream()
                return
            for kind, layer in (("gdn", 0), ("full", 3)):
                self.buffers[kind] = tuple(
                    self._allocate(value, initialize=False)
                    for value in self.head.banks(layer)
                )

    def _prepare_packed_stream(self):
        if self.group_size == 4:
            return self._prepare_packed_groups()
        import torch
        from pypto.runtime.task_interface import DataType

        pool = self.head.shared_weights
        if pool.dtype != torch.uint8 or pool.ndim != 1 or not pool.is_contiguous():
            raise ValueError("packed transfer requires one contiguous byte head storage")
        layouts, maximum = {}, {"gdn": 0, "full": 0}
        dtypes = {torch.bfloat16: DataType.BFLOAT16, torch.float32: DataType.FLOAT32}
        for layer in range(64):
            values = self.head.banks(layer)
            offsets = [value.data_ptr() - pool.data_ptr() for value in values]
            begin = min(offsets)
            end = max(offset + value.numel() * value.element_size()
                      for offset, value in zip(offsets, values, strict=True))
            if begin < 0 or end > pool.numel() or any(not value.is_contiguous() for value in values):
                raise ValueError(f"head layer {layer} is outside its packed storage")
            kind = "full" if layer % 4 == 3 else "gdn"
            layouts[layer] = (kind, pool[begin:end], values, [offset - begin for offset in offsets])
            maximum[kind] = max(maximum[kind], end - begin)
        for kind, size in maximum.items():
            # 仅父缓冲拥有设备内存；子张量通过公开的 Buffer.tensor 保留同一 owner。
            self.buffers[kind] = self._allocate(torch.empty(size, dtype=torch.uint8), initialize=False)
        for layer, (kind, span, values, offsets) in layouts.items():
            parent = self.buffers[kind]
            views = tuple(parent.buffer.tensor(tuple(value.shape), dtypes[value.dtype], byte_offset=offset)
                          for value, offset in zip(values, offsets, strict=True))
            self.packed[layer] = (span, parent, views)

    def _prepare_packed_groups(self):
        import torch
        from pypto.runtime.task_interface import DataType

        pool = self.head.shared_weights
        if pool.dtype != torch.uint8 or pool.ndim != 1 or not pool.is_contiguous():
            raise ValueError("packed groups require one contiguous byte head storage")
        dtypes = {torch.bfloat16: DataType.BFLOAT16, torch.float32: DataType.FLOAT32}
        layouts = {}
        for base in range(0, 64, 4):
            banks = [self.head.banks(layer) for layer in range(base, base + 4)]
            offsets = [[value.data_ptr() - pool.data_ptr() for value in values] for values in banks]
            begin = min(offset for row in offsets for offset in row)
            end = max(offset + value.numel() * value.element_size()
                      for row, values in zip(offsets, banks, strict=True)
                      for offset, value in zip(row, values, strict=True))
            if begin < 0 or end > pool.numel() or any(not v.is_contiguous() for row in banks for v in row):
                raise ValueError(f"head group {base} is outside its packed storage")
            layouts[base] = (pool[begin:end], banks, [[offset - begin for offset in row] for row in offsets])
        # 三个 GDN 层必须拥有各自的参数视图；不能连续上传到同一个逐层缓冲后再执行。
        size = max(span.numel() for span, _, _ in layouts.values())
        self.buffers["group"] = parent = self._allocate(torch.empty(size, dtype=torch.uint8), initialize=False)
        for base, (span, banks, offsets) in layouts.items():
            views = tuple(tuple(parent.buffer.tensor(tuple(value.shape), dtypes[value.dtype], byte_offset=offset)
                                for value, offset in zip(values, row, strict=True))
                          for values, row in zip(banks, offsets, strict=True))
            self.packed[base] = (span, parent, views)

    def for_group(self, base):
        if self.group_size != 4 or type(base) is not int or base not in range(0, 64, 4):
            raise ValueError("a four-layer head and an aligned group base are required")
        if self.storage == "resident":
            return tuple(self.overrides.get(layer, self.resident[layer]) for layer in range(base, base + 4))
        if self.storage != "stream":
            raise RuntimeError("head device weights are not started")
        span, parent, views = self.packed[base]
        length = span.numel()
        if base + 3 in self.overrides:
            length = self.head.banks(base + 3)[0].data_ptr() - span.data_ptr()
        self.worker.copy_to(parent.data_ptr, span.data_ptr(), length, worker_id=self.worker_id)
        return tuple(self.overrides.get(base + i, values) for i, values in enumerate(views))

    def for_layer(self, layer):
        if self.group_size != 1:
            raise RuntimeError("four-layer weights must be consumed through for_group")
        if layer in self.overrides:
            return self.overrides[layer]
        if self.storage == "resident":
            return self.resident[layer]
        if self.storage != "stream":
            raise RuntimeError("head device weights are not started")
        if self.transfer == "packed":
            span, parent, views = self.packed[layer]
            self.worker.copy_to(parent.data_ptr, span.data_ptr(), span.numel(), worker_id=self.worker_id)
            return views
        values = self.head.banks(layer)
        handles = self.buffers["full" if layer % 4 == 3 else "gdn"]
        for value, handle in zip(values, handles, strict=True):
            if tuple(value.shape) != handle.shape or value.dtype != handle.dtype:
                raise ValueError(f"head buffer layout changed at layer {layer}")
            self.worker.copy_to(
                handle.data_ptr, value.data_ptr(), value.numel() * value.element_size(), worker_id=self.worker_id
            )
        return handles

    def close(self):
        # 首次 free 失败时保留未释放的 owner/handle，允许同一 worker 上重试。
        self.storage = None
        self.resident.clear()
        self.buffers.clear()
        self.packed.clear()
        self.overrides.clear()
        self.version = 0
        while self.handles:
            self.worker.free_tensor(self.handles[-1], worker_id=self.worker_id)
            self.handles.pop()
        self.worker = None
