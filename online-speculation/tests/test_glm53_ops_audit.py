"""对照现成融合调用，保持临时块可见性及状态所有权不变。"""

import os

import pytest
import torch
from torch.nn import functional as F

from blockspec.vllm_ascend.glm53_ops import _block_segment, _segment, kda_convolve


@pytest.mark.parametrize("length", [1, 2, 8, 32])
def test_convolution_reference_uses_only_given_tail(length):
    torch.manual_seed(5303)
    rows, tail, weight = torch.randn(length, 24), torch.randn(3, 24), torch.randn(24, 4)
    before = [value.clone() for value in (rows, tail, weight)]
    expected = F.silu(F.conv1d(torch.cat((tail, rows)).T[None], weight[:, None], groups=24))[0].T
    torch.testing.assert_close(kda_convolve(rows, tail, weight), expected)
    for value, old in zip((rows, tail, weight), before, strict=True):
        assert torch.equal(value, old)


@pytest.mark.skipif(os.environ.get("ORTHRUS_RUN_NPU_TESTS") != "1", reason="allocated NPU required")
@pytest.mark.parametrize("length,heads", [(1, 8), (2, 8), (8, 8), (8, 32), (8, 64), (32, 8)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_shared_block_and_fused_convolution(length, heads, weight_dtype):
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    assert enable_custom_op()
    torch.manual_seed(5303)
    query = torch.randn(length, heads, 512, device="npu", dtype=torch.bfloat16)
    block = torch.randn(length, 512, device="npu", dtype=torch.bfloat16)
    channels = 3 * heads * 128
    # 模拟真实融合投影的 QKV 切片，行间仍跨过 beta 和两组 gate。
    rows = torch.randn(length, channels + heads + 256, device="npu", dtype=torch.bfloat16)[:, :channels]
    tail = torch.randn(3, rows.shape[1], device="npu", dtype=rows.dtype)
    weight = torch.randn(rows.shape[1], 4, device="npu", dtype=weight_dtype) * .1
    before = [value.clone() for value in (query, block, rows, tail, weight)]
    with torch.inference_mode():
        actual, lse = _block_segment(query, block, 256**-.5, "npu")
        old, old_lse = _segment(query, block.expand(length, -1, -1), 256**-.5, "npu")
        expected, expected_lse = _block_segment(query, block, 256**-.5, "torch")
        torch.testing.assert_close(actual, old, atol=.015, rtol=.02)
        torch.testing.assert_close(lse, old_lse, atol=.002, rtol=.001)
        torch.testing.assert_close(actual, expected, atol=.015, rtol=.02)
        torch.testing.assert_close(lse, expected_lse, atol=.002, rtol=.001)
        torch.testing.assert_close(kda_convolve(rows, tail, weight, backend="npu"),
                                   kda_convolve(rows, tail, weight), atol=.015, rtol=.02)
        selected = "npu" if heads in (32, 64) and length <= 8 else "torch"
        torch.testing.assert_close(kda_convolve(rows, tail, weight, backend="auto"),
                                   kda_convolve(rows, tail, weight, backend=selected), atol=0, rtol=0)
    for value, old in zip((query, block, rows, tail, weight), before, strict=True):
        assert torch.equal(value, old)
