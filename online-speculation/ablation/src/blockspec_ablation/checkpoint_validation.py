"""Bounded-scratch finite checks for dense, CPU-resident training checkpoints."""

import torch


def all_finite(value):
    """Test every element, including all IEEE infinity and NaN encodings."""
    formats = {torch.float32: (torch.int32, 0x7f800000), torch.bfloat16: (torch.int16, 0x7f80)}
    if value.device.type != "cpu" or value.layout != torch.strided or value.dtype not in formats:
        return bool(torch.isfinite(value).all())
    integer, exponent = formats[value.dtype]
    bits = value.detach().contiguous().reshape(-1).view(integer).numpy()
    # An all-ones exponent identifies infinity/NaN; sign and significand drop
    # out of the test. One chunk bounds the temporary array to four MiB.
    for start in range(0, bits.size, 1 << 20):
        if (bits[start:start + (1 << 20)] & exponent).max(initial=0) == exponent:
            return False
    return True
