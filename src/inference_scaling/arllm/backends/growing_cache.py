"""Block-allocated storage for the legacy Transformers DynamicCache API.

The visible tensors retain exactly the used sequence length. Spare capacity is
not passed to attention, and keys/values are copied without numerical changes.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _cache_type():
    from transformers.cache_utils import DynamicCache

    example = DynamicCache()
    if not {"key_cache", "value_cache", "_seen_tokens"} <= vars(example).keys():
        raise RuntimeError("block cache growth requires the legacy DynamicCache storage API")

    class BlockAllocatedCache(DynamicCache):
        def __init__(self, block_tokens=512):
            super().__init__()
            if not isinstance(block_tokens, int) or block_tokens <= 0:
                raise ValueError("cache growth block must be a positive integer")
            self.block_tokens = block_tokens
            self._key_storage = []
            self._value_storage = []
            self.copied_prefix_elements = 0

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states.new_empty((0,)))
                self.value_cache.append(value_states.new_empty((0,)))
            while len(self._key_storage) <= layer_idx:
                self._key_storage.append(None)
                self._value_storage.append(None)
            previous_key, previous_value = self.key_cache[layer_idx], self.value_cache[layer_idx]
            previous_length = previous_key.shape[-2] if previous_key.numel() else 0
            required = previous_length + key_states.shape[-2]
            keys, values = self._key_storage[layer_idx], self._value_storage[layer_idx]
            # Parent crop retains a view into storage. Reorder/repeat may create
            # different tensors; use those public tensors as the authoritative prefix.
            reusable = (keys is not None and values is not None and previous_length > 0
                        and previous_key.data_ptr() == keys.data_ptr()
                        and previous_value.data_ptr() == values.data_ptr()
                        and keys.shape[-2] >= required
                        and keys.shape[:-2] == key_states.shape[:-2]
                        and values.shape[:-2] == value_states.shape[:-2])
            if not reusable:
                capacity = ((required + self.block_tokens - 1) // self.block_tokens) * self.block_tokens
                keys = key_states.new_empty((*key_states.shape[:-2], capacity, key_states.shape[-1]))
                values = value_states.new_empty((*value_states.shape[:-2], capacity, value_states.shape[-1]))
                if previous_length:
                    keys[..., :previous_length, :].copy_(previous_key)
                    values[..., :previous_length, :].copy_(previous_value)
                    self.copied_prefix_elements += previous_key.numel() + previous_value.numel()
                self._key_storage[layer_idx], self._value_storage[layer_idx] = keys, values
            keys[..., previous_length:required, :].copy_(key_states)
            values[..., previous_length:required, :].copy_(value_states)
            self.key_cache[layer_idx] = keys[..., :required, :]
            self.value_cache[layer_idx] = values[..., :required, :]
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    return BlockAllocatedCache


def validate_cache_growth_support() -> None:
    _cache_type()


def block_allocated_cache(cache, block_tokens: int):
    from transformers.cache_utils import DynamicCache

    if type(cache) is not DynamicCache:
        raise RuntimeError("block cache growth supports ordinary DynamicCache, without offloading or quantization")
    result = _cache_type()(block_tokens)
    result._seen_tokens = cache._seen_tokens
    result.key_cache = list(cache.key_cache)
    result.value_cache = list(cache.value_cache)
    return result
