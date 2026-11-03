"""分页快照只复制有效前缀，不物化 arena，也不引用目标 cache。"""

import os

import pytest
import torch

from blockspec.vllm_ascend.glm53_native import read_pages
from blockspec.vllm_ascend.glm53_ops import gather_history


def reference(cache, table, length, block):
    split = cache.shape[1] // block
    pieces = [cache[i // split, i % split * block:(i % split + 1) * block, 0]
              for i in table[0].cpu().tolist()[:(length + block - 1) // block]]
    return torch.cat(pieces)[:length].contiguous() if pieces else cache.new_empty((0, cache.shape[-1]))


@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("length", [0, 1, 47, 48, 49, 191])
def test_snapshot_owns_only_valid_subpages(monkeypatch, padding, length):
    # 行末及页末分别有 padding，cache 还包含非零 storage offset。
    arena = torch.arange(34 * 15 * (7 + padding), dtype=torch.float32).reshape(34, 15, 1, 7 + padding)
    cache = arena[1:33, :12, :, :7]
    table = torch.arange(127, -1, -1).reshape(1, -1)
    expected = reference(cache, table, length, 3)
    before = arena.clone()
    original = torch.Tensor.index_select
    calls = []
    def checked(tensor, dim, indices):
        assert tensor.is_contiguous()
        assert tensor.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert indices.numel() == length
        calls.append(True)
        return original(tensor, dim, indices)
    monkeypatch.setattr(torch.Tensor, "index_select", checked)
    actual = read_pages(cache, table, length, block_size=3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert bool(calls) == (length > 45)
    actual.zero_()
    torch.testing.assert_close(arena, before, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["column_stride", "short_storage"])
def test_unrepresentable_pitched_view_uses_page_slices(monkeypatch, layout):
    if layout == "column_stride":
        cache = torch.arange(32 * 12 * 14.).reshape(32, 12, 1, 14)[..., ::2]
    else:
        # 最后一个有效元素之后没有足够存储容纳一整行 pitch。
        cache = torch.arange(32 * 12 * 9. - 2).as_strided((32, 12, 1, 7), (108, 9, 9, 1))
    table = torch.arange(127, -1, -1).reshape(1, -1)
    expected = reference(cache, table, 191, 3)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported layout must not use a pitched gather")
    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)
    torch.testing.assert_close(read_pages(cache, table, 191, block_size=3), expected, atol=0, rtol=0)


@pytest.mark.parametrize("index", [-1, 128])
def test_invalid_page_is_rejected_before_index_select(index):
    table = torch.zeros(1, 16, dtype=torch.int32)
    table[0, 15] = index
    with pytest.raises(ValueError, match="physical cache arena"):
        read_pages(torch.zeros(32, 12, 1, 7), table, 48, block_size=3)


@pytest.mark.parametrize("width,selected", [(2, 7), (2, 1024), (8, 2051)])
@pytest.mark.parametrize("padding", [0, 3])
def test_selected_history_keeps_duplicates_and_owns_output(width, selected, padding):
    torch.manual_seed(5312)
    arena = torch.randn(2055, 7 + padding)
    history = arena[:, :7]
    indices = torch.randint(2055, (width, selected))
    before = arena.clone()
    actual = gather_history(history, indices)
    torch.testing.assert_close(actual, history[indices], atol=0, rtol=0)
    actual.zero_()
    torch.testing.assert_close(arena, before, atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get("ORTHRUS_RUN_NPU_TESTS") != "1", reason="allocated NPU required")
@pytest.mark.parametrize("dim,pitch", [(128, 128), (512, 512), (512, 576)])
@pytest.mark.parametrize("length", [7, 2055, 8195])
def test_npu_snapshot_matches_page_slices_without_copying_arena(dim, pitch, length):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(5311)
    arena = torch.randn(1024, 128, 1, pitch, device="npu", dtype=torch.bfloat16)
    cache = arena[1:1023, :96, :, :dim]
    table = torch.randperm(1022 * 6).to(device="npu", dtype=torch.int32)[None]
    expected = reference(cache, table, length, 16)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated = torch.npu.memory_allocated()
    actual = read_pages(cache, table, length, block_size=16)
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - allocated
    assert torch.equal(actual, expected)
    # 峰值随本请求有效前缀增长，不得出现整个 arena 的物化副本。
    assert peak < 16 * 1024**2 + length * (pitch + dim) * cache.element_size()
    actual.zero_()
    assert torch.equal(reference(cache, table, length, 16), expected)


@pytest.mark.skipif(os.environ.get("ORTHRUS_RUN_NPU_TESTS") != "1", reason="allocated NPU required")
@pytest.mark.parametrize("width,selected", [(2, 7), (2, 2051), (8, 2051), (32, 2051)])
def test_npu_gather_is_exact_and_uses_only_output_sized_storage(width, selected):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(5312)
    history = torch.randn(8195, 512, device="npu", dtype=torch.bfloat16)
    indices = torch.randint(len(history), (width, selected), device="npu")
    expected = history[indices]
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated = torch.npu.memory_allocated()
    actual = gather_history(history, indices)
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - allocated
    assert torch.equal(actual, expected)
    if indices.numel() >= 2048:
        assert peak < actual.numel() * actual.element_size() + 1024**2
    actual.zero_()
    assert torch.equal(history[indices], expected)
