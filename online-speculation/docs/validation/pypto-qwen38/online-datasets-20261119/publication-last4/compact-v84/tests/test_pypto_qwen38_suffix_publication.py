"""发布必须覆盖整个 TP bank，不能漏层或把 padded gate 行当成有效 head。"""

from dataclasses import replace

import pytest

pytest.importorskip("serving.runtime")

from blockspec.pypto_qwen38.training_flat import FlatLayout
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.tp2_suffix_publication import suffix_publication_plan


@pytest.mark.parametrize("scope,banks,count", [
    ("full", set(range(15)), 33),
    ("norms", {8, 12, 13}, 5),
    ("output_norms", {2, 8, 12, 13, 14}, 9),
])
def test_complete_suffix_stack_and_small_scalar_boundary(scope, banks, count):
    layout = FlatLayout(SuffixGraph(4, parameter_scope=scope).parameters)
    plan = suffix_publication_plan(layout, banks)
    assert len(plan) == count
    assert {p.bank for p in plan} == banks
    for bank in banks:
        stack = [p for p in plan if p.bank == bank]
        assert {p.slot for p in stack} == ({0, 1, 2} if bank < 9 else {0})
        assert len({p.target for p in stack}) == 1
        assert sum(p.local_rows for p in stack) == stack[0].rank_rows
    if scope == "full":
        assert sum(p.mode == "scalar" for p in plan) == 6
        assert sum(p.mode == "direct" for p in plan) == 3
        assert all(p.source == (1, 64) and p.target == (6, 32)
                   for p in plan if p.mode == "scalar")
        assert {p.bank for p in plan if p.mode == "padded_rows"} == {3, 4}
        assert {p.bank for p in plan if p.mode == "qkv_columns"} == {5}


@pytest.mark.parametrize("bad", ["missing_layer", "shape", "dtype", "extra_bank"])
def test_invalid_suffix_schema_is_rejected_before_compilation(bad):
    parameters = SuffixGraph(4).parameters.copy()
    if bad == "missing_layer":
        parameters = [p for p in parameters if p.layer != 61]
    elif bad == "shape":
        parameters[0] = replace(parameters[0], shape=(10240, 128))
    elif bad == "dtype":
        index = next(i for i, p in enumerate(parameters) if p.shape == (1, 128))
        parameters[index] = replace(parameters[index], raw_dtype="fp32")
    else:
        parameters[0] = replace(parameters[0], bank=9)
    with pytest.raises(ValueError):
        suffix_publication_plan(FlatLayout(parameters), set(range(15)))


def test_device_workspace_counts_each_layer_stack_once():
    from math import prod
    from blockspec.pypto_qwen38.oneserving_tp2 import SHAPES
    from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms
    obj = object.__new__(SuffixPublicationPrograms)
    plan = suffix_publication_plan(FlatLayout(SuffixGraph(4).parameters), set(range(15)))
    obj.outputs = {p.bank: p for p in plan if p.mode not in {"direct", "scalar", "rows"}}
    expected = sum(2 * prod(shape) * dtype.itemsize for i, (shape, dtype) in enumerate(SHAPES)
                   if i in {0, 2, 3, 4, 5, 8, 12, 13, 14})
    assert obj.allocation_bytes == expected


def test_full_suffix_device_packing_is_explicit_until_stream_comparison():
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
    assert TP2Online(None, device=0, last_layers=4, publication_backend="device").device_publication
    assert not TP2Online(None, device=0, last_layers=4).device_publication


def test_scalar_readback_preserves_rank_order_offsets_and_padding():
    import ctypes
    from types import SimpleNamespace
    import torch
    from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms, Copy
    # 用独立内容模拟设备缓冲，检查 FP32/BF16 offset、三层堆叠和两 rank。
    source_bf = torch.arange(64 * 3, dtype=torch.bfloat16).reshape(1, -1)
    source_fp = (torch.arange(64 * 3, dtype=torch.float32) + 1000).reshape(1, -1)
    host_bf = torch.full_like(source_bf, -3)
    host_fp = torch.full_like(source_fp, -5)
    packed = {6: torch.full((2, 3, 32), -9, dtype=torch.float32),
              7: torch.full((2, 3, 32), -9, dtype=torch.bfloat16)}
    entries, specs, calls = [], [], []
    for bank in (6, 7):
        for slot in range(3):
            name = f'layer{slot}.bank{bank}'
            entries.append(SimpleNamespace(parameter=SimpleNamespace(name=name),
                                           offset=slot * 64, special=slot * 64 if bank == 6 else None))
            specs.append(Copy(bank, name, (1, 64), (6, 32),
                              'fp32' if bank == 6 else 'bf16', 'scalar', slot, 1, 3))
    obj = object.__new__(SuffixPublicationPrograms)
    obj.plan = specs
    obj.sources = lambda *args: {}
    def copy(dst, base, size, *, src_offset):
        calls.append((base, size, src_offset))
        ctypes.memmove(dst, base + src_offset, size)
    worker = SimpleNamespace(copy_from=copy)
    buffers = tuple(SimpleNamespace(data_ptr=x.data_ptr()) for x in (source_bf, source_fp))
    obj.readback(worker, SimpleNamespace(entries=entries), buffers, {}, packed, (host_bf, host_fp))
    for bank, source in [(6, source_fp), (7, source_bf)]:
        for rank in range(2):
            for slot in range(3):
                assert torch.equal(packed[bank][rank, slot, :24],
                                   source[0, slot * 64 + rank * 24:slot * 64 + (rank + 1) * 24])
        assert torch.count_nonzero(packed[bank][:, :, 24:]) == 0
    assert torch.equal(host_bf, source_bf) and torch.equal(host_fp, source_fp)
    assert calls == [(source_fp.data_ptr(), 256, slot * 256) for slot in range(3)] + [
        (source_bf.data_ptr(), 128, slot * 128) for slot in range(3)]


def test_stacked_row_readback_uses_contiguous_rank_slices_without_device_scratch():
    import ctypes
    from types import SimpleNamespace
    import torch
    from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms, Copy
    # 三层原参数按 layer/rank 排列，最终 bank 按 rank/layer 排列。
    source = torch.arange(48, dtype=torch.bfloat16).reshape(1, -1)
    packed = {1: torch.full((2, 6, 4), -1, dtype=torch.bfloat16)}
    obj = object.__new__(SuffixPublicationPrograms)
    obj.plan = [Copy(1, f'z{slot}', (4, 4), (12, 4), 'bf16', 'rows', slot, 2, 6)
                for slot in range(3)]
    entries = [SimpleNamespace(parameter=SimpleNamespace(name=f'z{slot}'), offset=slot * 16)
               for slot in range(3)]
    obj.sources = lambda *args: {}
    calls = []
    def copy(dst, base, size, *, src_offset):
        calls.append((base, size, src_offset))
        ctypes.memmove(dst, base + src_offset, size)
    obj.readback(SimpleNamespace(copy_from=copy), SimpleNamespace(entries=entries),
                 (SimpleNamespace(data_ptr=source.data_ptr()), None), {}, packed, (None, None))
    expected = source.reshape(3, 2, 2, 4).permute(1, 0, 2, 3).reshape(2, 6, 4)
    assert torch.equal(packed[1], expected)
    assert calls == [(source.data_ptr(), 16, (slot * 16 + rank * 8) * 2)
                     for slot in range(3) for rank in range(2)]


def test_publication_arena_uses_owned_base_with_nonoverlapping_aligned_views():
    from types import SimpleNamespace
    from math import prod
    import torch
    from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms
    obj = object.__new__(SuffixPublicationPrograms)
    layout = FlatLayout(SuffixGraph(4).parameters)
    obj.plan = suffix_publication_plan(layout, set(range(15)))
    obj.outputs = {p.bank: p for p in obj.plan if p.mode not in {'direct', 'scalar', 'rows'}}
    views = []
    def view(shape, dtype, *, byte_offset):
        views.append((shape, dtype, byte_offset))
        return object()
    scratch = SimpleNamespace(shape=(1, obj.allocation_bytes // 4), dtype=torch.float32,
                              data_ptr=8192, buffer=SimpleNamespace(tensor=view))
    worker = SimpleNamespace(alloc_tensor=lambda *args: pytest.fail('arena must not allocate another buffer'))
    destinations = obj.allocate(worker, scratch=scratch)
    offset = 0
    for bank, spec in obj.outputs.items():
        dst = destinations[bank]
        assert dst.base == 8192 and dst.offset == offset and offset % 64 == 0
        size = prod(spec.target) * (4 if spec.dtype == 'fp32' else 2)
        offset += (size + 63) // 64 * 64
    assert offset == obj.allocation_bytes and len(views) == len(obj.outputs)
    bf16_arena = SimpleNamespace(shape=(1, obj.allocation_bytes // 2), dtype=torch.bfloat16,
                                 data_ptr=16384, buffer=scratch.buffer)
    assert all(x.base == 16384 for x in obj.allocate(worker, scratch=bf16_arena).values())
    for bank, (base, offset) in obj.sources(layout, scratch, destinations).items():
        assert base == 8192
        if bank in destinations:
            assert offset == destinations[bank].offset
    for bad in [SimpleNamespace(shape=(1, 8), dtype=torch.float32),
                SimpleNamespace(shape=scratch.shape, dtype=torch.float16)]:
        with pytest.raises(ValueError, match='sufficient'):
            obj.allocate(worker, scratch=bad)


def test_reused_gradient_scratch_is_cleared_before_next_update_including_padding():
    from types import SimpleNamespace
    import torch
    from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer
    opt = object.__new__(SuffixOptimizer)
    opt.programs = SimpleNamespace(zero=object())
    opt.scratch = torch.full((1, 64), 9., dtype=torch.float32)
    opt.candidate = [torch.full((1, 64), 3.) for _ in range(3)]
    opt.state = [torch.full((1, 64), float(i + 4)) for i in range(3)]
    original = [x.clone() for x in opt.state + opt.candidate[1:]]
    calls = []
    def run(program, target):
        assert program is opt.programs.zero
        calls.append(target)
        target.zero_()
    opt.run = run
    opt.scratch_dirty = True
    opt.zero_grad()
    assert len(calls) == 2 and calls[0] is opt.scratch and calls[1] is opt.candidate[0]
    assert not opt.scratch_dirty and torch.count_nonzero(opt.scratch) == 0
    for a, b in zip(original, opt.state + opt.candidate[1:]):
        assert torch.equal(a, b)
    # 下一步没有发布时，无需重复清空由反向完整覆盖的真实参数区间。
    opt.scratch.fill_(2.)
    opt.zero_grad()
    assert len(calls) == 3 and calls[-1] is opt.candidate[0]
    assert torch.all(opt.scratch == 2.)
