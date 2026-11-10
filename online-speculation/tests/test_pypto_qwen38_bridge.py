"""有效教师行传输的字节边界；未传输行必须保持不变。"""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.online import FeedbackBlock
from blockspec.pypto_qwen38.online_bridge import TensorBridge
from blockspec.pypto_qwen38.training_programs import reduction_tile


class MemoryWorker:
    def copy_from(self, target, source, count, *, src_offset=0):
        ctypes.memmove(target, source + src_offset, count)

    def copy_to(self, target, source, count, *, dst_offset=0):
        ctypes.memmove(target + dst_offset, source, count)


def handle(value):
    return SimpleNamespace(shape=value.shape, dtype=value.dtype, data_ptr=value.data_ptr())


@pytest.mark.parametrize("valid", (0, 1, 7, 31, 32))
def test_feedback_transfer_preserves_unsupervised_teacher_rows(valid):
    worker = MemoryWorker()
    bridge = TensorBridge()
    # 让搬运跨过多个 chunk，覆盖最后一个不满 chunk 的分支。
    bridge.scratch = torch.empty(80, dtype=torch.uint8)
    source = dict(teacher=torch.arange(32 * 64).reshape(32, 64).to(torch.bfloat16),
                  hidden=torch.randn(32, 64, dtype=torch.bfloat16))
    destination = {k: torch.full_like(v, float("nan")) for k, v in source.items()}
    owner = SimpleNamespace(native=SimpleNamespace(worker=worker), trainer=worker, bridge=bridge)
    block = FeedbackBlock(owner, {k: handle(v) for k, v in source.items()})
    block.training_inputs = {k: handle(v) for k, v in destination.items()}
    block.valid = valid
    block.transfer()
    assert torch.equal(destination["teacher"][:valid], source["teacher"][:valid])
    assert torch.isnan(destination["teacher"][valid:]).all()
    assert torch.equal(destination["hidden"], source["hidden"])
    assert bridge.bytes == valid * 64 * 2 + source["hidden"].numel() * 2


@pytest.mark.parametrize("kwargs", [dict(nbytes=-2), dict(nbytes=9), dict(nbytes=18),
    dict(source_offset=1), dict(source_offset=10), dict(nbytes=2.0)])
def test_invalid_partial_copy_is_rejected_before_memory_access(kwargs):
    worker = MemoryWorker()
    bridge = TensorBridge()
    source = torch.zeros(8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bounds"):
        bridge.copy(worker, handle(source), worker, handle(source), **kwargs)
    assert bridge.bytes == 0


def test_source_offset_and_full_copy_remain_supported():
    worker, bridge = MemoryWorker(), TensorBridge()
    source = torch.arange(24).to(torch.bfloat16)
    destination = torch.empty(8, dtype=torch.bfloat16)
    bridge.copy(worker, handle(source), worker, handle(destination), source_offset=16)
    assert torch.equal(destination, source[8:16])


@pytest.mark.parametrize("k,limit,expected", [(32, 512, 32), (192, 512, 64),
    (5120, 512, 512), (17408, 256, 256), (248320, 128, 128), (256, 64, 64)])
def test_reduction_tiles_cover_unpadded_k(k, limit, expected):
    assert reduction_tile(k, limit) == expected
