"""两个独立 NPU worker 之间的有界字节搬运；不在主机计算模型值。"""

from math import prod


class TensorBridge:
    def __init__(self):
        import torch
        # 在两个 worker 启动前登记同一共享缓冲，避免每次复制临时分配 shm 和主机 memcpy。
        self.scratch = torch.empty(16 << 20, dtype=torch.uint8).share_memory_()
        self.bytes = 0

    def copy(self, source_worker, source, target_worker, target, *, source_offset=0, nbytes=None):
        target_capacity = prod(target.shape)*target.dtype.itemsize
        size = target_capacity if nbytes is None else nbytes
        capacity = prod(source.shape)*source.dtype.itemsize
        if (type(size) is not int or type(source_offset) is not int
                or source.dtype != target.dtype or size < 0 or size > target_capacity
                or source_offset < 0 or source_offset+size > capacity
                or size % source.dtype.itemsize or source_offset % source.dtype.itemsize):
            raise ValueError("incompatible byte transfer bounds or dtype")
        for offset in range(0, size, self.scratch.numel()):
            count = min(self.scratch.numel(), size-offset)
            source_worker.copy_from(self.scratch.data_ptr(), source.data_ptr, count, src_offset=source_offset+offset)
            target_worker.copy_to(target.data_ptr, self.scratch.data_ptr(), count, dst_offset=offset)
        self.bytes += size


class HeadPublisher:
    def __init__(self, learner, target):
        self.learner, self.target, self.active = learner, target, 0

    @property
    def version(self):
        return self.target.version

    @version.setter
    def version(self, value):
        self.target.version = value

    def publish(self, layer, values, *, expected_version):
        if self.version != expected_version:
            raise RuntimeError("stale cross-device head publication")
        index = 1-self.active
        banks = self.learner.serving_banks[index]
        for source, target in zip(values, banks, strict=True):
            self.learner.bridge.copy(self.learner.trainer, source, self.learner.native.worker, target)
        self.target.publish(layer, banks, expected_version=expected_version)
        self.active = index


class SuffixPublisher(HeadPublisher):
    def publish_flat(self, layout, values, *, expected_version):
        if self.version != expected_version:
            raise RuntimeError("stale cross-device suffix publication")
        index = 1-self.active
        buffers = self.learner.serving_banks[index]
        for source,target in zip(values,buffers,strict=True):
            self.learner.bridge.copy(self.learner.trainer,source,self.learner.native.worker,target)
        self.target.publish_many(layout.banks(*buffers),expected_version=expected_version)
        self.active = index
