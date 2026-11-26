"""Map graph-owned EP communication buffers into the local HCCL context."""

from dataclasses import dataclass


@dataclass
class _Binding:
    exported: object
    mapping: object = None


class ResidentHcclMappings:
    """Keep each backing leased until HCCL users finish and unmap."""

    def __init__(self, runtime, memory):
        self.runtime, self.memory = runtime, memory
        self._bindings = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def register(self, rank, tensor):
        """Export one complete graph-owned tensor before the first collective."""
        from simpler.buffer_export import BufferExportError, BufferImportError, BufferMapping

        if tensor.byte_offset or tensor.nbytes != tensor.buffer.nbytes:
            raise ValueError("An EP IPC binding must cover a complete resident allocation")
        key = rank, tensor.data_ptr
        if key in self._bindings:
            return
        try:
            exported = self.runtime.export_buffer(tensor, worker_id=rank)
        except BufferExportError as error:
            self._bindings[key] = _Binding(error.exported)
            raise
        binding = self._bindings[key] = _Binding(exported)
        self.memory.select(exported.device_id)
        try:
            binding.mapping = BufferMapping(exported)
        except BufferImportError as error:
            binding.mapping = error.mapping
            raise

    def address(self, rank, tensor):
        """Resolve a registered tensor's consumer-local address."""
        return self._bindings[rank, tensor.data_ptr].mapping.base

    def close(self):
        """Retire each mapping before releasing its graph owner's export lease."""
        first_error = None
        for key, binding in reversed(tuple(self._bindings.items())):
            try:
                if binding.mapping is not None:
                    self.memory.select(binding.exported.device_id)
                    binding.mapping.close()
                    binding.mapping = None
                self.runtime.release_export(binding.exported)
                del self._bindings[key]
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
