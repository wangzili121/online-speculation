import pytest
import torch

from blockspec_ablation.checkpoint_validation import all_finite


def test_all_bfloat16_bit_patterns_match_torch_finiteness():
    values = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    expected = torch.isfinite(values)
    assert int(expected.sum()) == 65280
    assert all_finite(values[expected]) is True
    for value in values[~expected]:
        assert all_finite(value) is False
    assert all_finite(values) is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index", [0, (1 << 20) - 1, 1 << 20, (1 << 20) + 7])
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_at_chunk_boundaries(dtype, index, bad):
    values = torch.ones((1 << 20) + 8, dtype=dtype)
    assert all_finite(values) is True
    values[index] = bad
    assert all_finite(values) is False


def test_random_float32_encodings_and_all_exponents():
    rng = torch.Generator().manual_seed(461)
    bits = torch.randint(-(1 << 31), 1 << 31, (1 << 20,), generator=rng, dtype=torch.int64).to(torch.int32)
    values = bits.view(torch.float32)
    expected = torch.isfinite(values)
    assert all_finite(values[expected]) is True
    assert all_finite(values) is False
    for value in values[~expected][:512]:
        assert all_finite(value) is False
    for sign in (0, 1 << 31):
        for exponent in range(256):
            encodings = torch.tensor([sign | (exponent << 23) | mantissa
                                      for mantissa in (0, 1, (1 << 23) - 1)], dtype=torch.int64).to(torch.int32)
            row = encodings.view(torch.float32)
            assert all_finite(row) is bool(torch.isfinite(row).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64, torch.float16])
def test_empty_scalar_strided_and_autograd_values(dtype):
    assert all_finite(torch.empty(0, dtype=dtype)) is True
    assert all_finite(torch.tensor(0., dtype=dtype)) is True
    values = torch.arange(35., dtype=dtype).reshape(5, 7).requires_grad_()
    for view in (values.T, values[:, ::2]):
        assert all_finite(view) is True
        assert view.grad_fn is not None
    with torch.no_grad():
        values[2, 2] = float("nan")
    assert all_finite(values[:, ::2]) is False and values.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback check")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fallback_checks_each_tensor(dtype):
    values = torch.ones(31, device="cuda", dtype=dtype)
    assert all_finite(values) is True
    values[-1] = float("inf")
    assert all_finite(values) is False
