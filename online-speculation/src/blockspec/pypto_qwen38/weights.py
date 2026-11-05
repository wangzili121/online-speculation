"""Keep head parameters resident when possible; reuse buffers when streaming."""


class DraftWeights:
    def __init__(self, head, mode="auto"):
        if mode not in {"auto", "resident", "stream"}:
            raise ValueError("head storage must be auto, resident or stream")
        self.head, self.mode = head, mode
        self.worker = None
        self.handles, self.resident, self.buffers = [], {}, {}
        self.storage = None

    def _allocate(self, value, *, initialize):
        handle = self.worker.alloc_tensor(
            tuple(value.shape), value.dtype, init=value if initialize else None
        )
        self.handles.append(handle)
        return handle

    def start(self, worker):
        if self.worker is not None or self.head.shared_weights is None:
            raise RuntimeError("prepare the head before starting its device weights")
        self.worker = worker
        free, _ = worker.device_memory_info()
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
                for layer in (0, 3)
                for value in self.head.banks(layer)
            ) + (256 << 20)
            if minimum > free:
                raise RuntimeError(
                    f"stream head buffers require {minimum} free bytes, available {free}"
                )
            for kind, layer in (("gdn", 0), ("full", 3)):
                self.buffers[kind] = tuple(
                    self._allocate(value, initialize=False)
                    for value in self.head.banks(layer)
                )

    def for_layer(self, layer):
        if self.storage == "resident":
            return self.resident[layer]
        if self.storage != "stream":
            raise RuntimeError("head device weights are not started")
        values = self.head.banks(layer)
        handles = self.buffers["full" if layer % 4 == 3 else "gdn"]
        for value, handle in zip(values, handles, strict=True):
            if tuple(value.shape) != handle.shape or value.dtype != handle.dtype:
                raise ValueError(f"head buffer layout changed at layer {layer}")
            self.worker.copy_to(
                handle.data_ptr, value.data_ptr(), value.numel() * value.element_size()
            )
        return handles

    def close(self):
        try:
            if self.worker is not None:
                for handle in reversed(self.handles):
                    self.worker.free_tensor(handle)
        finally:
            self.handles.clear()
            self.resident.clear()
            self.buffers.clear()
            self.worker = None
            self.storage = None
