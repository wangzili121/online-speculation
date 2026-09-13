"""Shared persistent-history representation for every draft branch and executor."""

from torch import Tensor


Cache = tuple[tuple[Tensor, Tensor], ...]


class PackedCache(tuple):
    """Immutable views into [layer,2,batch,head,time,width] storage.

    Execution workspaces copy these views before mutation. Prefix trimming
    preserves the packed owner so the next forward can perform one KV transfer.
    """

    def __new__(cls, packed):
        if packed.ndim != 6 or packed.shape[0] < 1 or packed.shape[1] != 2:
            raise ValueError("packed cache needs [layers, 2, batch, heads, time, dim]")
        result = super().__new__(cls, ((layer[0], layer[1]) for layer in packed.unbind(0)))
        result.packed = packed
        return result


def cache_length(cache: Cache | None):
    if cache is not None and not cache:
        raise ValueError("empty cache tuple; use None for an empty prefix")
    return 0 if cache is None else cache[0][0].shape[2]


def trim_cache(cache: Cache | None, length: int) -> Cache | None:
    """Commit an AR prefix, preserving packed storage and detaching feedback history."""
    if length < 0 or length > cache_length(cache):
        raise ValueError("cannot extend cache by trimming")
    if cache is None or length == 0:
        return None
    if isinstance(cache, PackedCache):
        return PackedCache(cache.packed[..., :length, :].detach())
    return tuple((k[:, :, :length].detach(), v[:, :, :length].detach()) for k, v in cache)
