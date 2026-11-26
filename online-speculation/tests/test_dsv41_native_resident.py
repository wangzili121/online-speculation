from ctypes import addressof, create_string_buffer, memmove

from blockspec.dsv41_native_resident import NativeRankMemory


class Buffer:
    def __init__(self, size):
        self.storage = create_string_buffer(size)
        self.base = addressof(self.storage)
        self.nbytes = size


class Worker:
    def __init__(self):
        self.host_allocations = 0

    def create_buffer(self, size):
        self.host_allocations += 1
        return Buffer(size)

    def copy_to(self, device, host, *, dst_offset, nbytes):
        memmove(device.base + dst_offset, host.base, nbytes)

    def copy_from(self, host, device, *, src_offset, nbytes):
        memmove(host.base, device.base + src_offset, nbytes)


def test_rank_memory_reuses_one_staging_buffer_for_tp_and_offset_reads():
    worker = Worker()
    runtime = NativeRankMemory(worker, staging_bytes=5)
    local = [Buffer(16) for _ in range(2)]
    expected = b"0123456789"
    source = create_string_buffer(expected)
    for rank, buffer in enumerate(local):
        runtime.allocations[rank, buffer.base] = buffer
    runtime.upload_to_ranks(
        addressof(source), len(expected),
        ((rank, buffer.base, rank + 2) for rank, buffer in enumerate(local)),
    )
    for rank, buffer in enumerate(local):
        output = create_string_buffer(10)
        runtime.copy_from(addressof(output), buffer.base, 10, worker_id=rank, src_offset=rank + 2)
        assert output.raw == expected
        assert buffer.storage.raw[:rank + 2] == bytes(rank + 2)
    assert worker.host_allocations == 1
