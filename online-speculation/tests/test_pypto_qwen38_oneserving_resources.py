import pytest

from blockspec.pypto_qwen38.oneserving_resources import bounded_ring_policy


def test_heap_floor_preserves_larger_native_budgets_and_admission_reserves():
    native = dict(ring_heap_bytes=[256 << 20, 1024 << 20, 256 << 20, 256 << 20],
                  future_bytes_after_prepare=987654, version="native")
    candidate = bounded_ring_policy(native, 512)
    assert candidate["ring_heap_bytes"] == [512 << 20, 1024 << 20, 512 << 20, 512 << 20]
    assert candidate["future_bytes_after_prepare"] == native["future_bytes_after_prepare"]
    assert candidate["native_ring_heap_bytes"] == native["ring_heap_bytes"]
    candidate["ring_heap_bytes"][0] = 0
    assert native["ring_heap_bytes"][0] == 256 << 20
    assert bounded_ring_policy(native, 256)["ring_heap_bytes"] == native["ring_heap_bytes"]


@pytest.mark.parametrize("size", [0, 128, 384, True, 512.0])
def test_invalid_heap_floor(size):
    with pytest.raises(ValueError, match="power of two"):
        bounded_ring_policy(dict(ring_heap_bytes=[256 << 20] * 4), size)
