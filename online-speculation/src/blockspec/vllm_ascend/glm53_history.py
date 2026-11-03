"""GLM 只读分页历史：复制完整前缀，或在同步 draft 内按需收集。"""

import torch


def _pages(cache, table, length, block_size):
    if (cache.ndim != 4 or cache.shape[2] != 1 or table.ndim != 2
            or table.shape[0] != 1 or type(length) is not int or length < 0
            or table.dtype not in (torch.int32, torch.int64)):
        raise ValueError("unsupported GLM batch-one paged latent layout")
    block_size = cache.shape[1] if block_size is None else block_size
    if block_size < 1 or cache.shape[1] % block_size:
        raise ValueError("operator pages must divide the physical cache page")
    count = (length + block_size - 1) // block_size
    if count > table.shape[1]:
        raise ValueError("prefix exceeds GLM page table")
    ids = tuple(table[0, :count].cpu().tolist()) if count else ()
    if any(not 0 <= index < cache.shape[0] * (cache.shape[1] // block_size) for index in ids):
        raise ValueError("operator page outside the physical cache arena")
    return block_size, ids


def _flat_rows(cache):
    pitch = cache.stride(1)
    if (pitch < max(1, cache.shape[-1]) or cache.stride(-1) != 1
            or cache.stride(0) < cache.shape[1] * pitch or cache.stride(0) % pitch):
        return None
    rows = (cache.shape[0] - 1) * (cache.stride(0) // pitch) + cache.shape[1]
    if (cache.storage_offset() + rows * pitch) * cache.element_size() > cache.untyped_storage().nbytes():
        return None
    if cache.device.type == "npu":
        import torch_npu

        # 四维输入可为普通 NCHW(0)，通用输入可为 ND(2)，均非分块存储。
        if torch_npu.get_npu_format(cache) not in (0, 2):
            return None
    # 包含物理 padding 的连续视图不分配存储；不能 reshape 被裁窄的 arena。
    return cache.as_strided((rows, pitch), (pitch, 1))


def _starts(cache, ids, block_size):
    split = cache.shape[1] // block_size
    page_pitch = cache.stride(0) // cache.stride(1)
    return torch.tensor([i // split * page_pitch + i % split * block_size for i in ids],
                        dtype=torch.long, device="cpu")


def _copy_prefix(cache, ids, length, block_size, rows):
    if not ids:
        return cache.new_empty((0, cache.shape[-1]))
    if len(ids) >= 16 and rows is not None:
        indices = (_starts(cache, ids, block_size)[:, None]
                   + torch.arange(block_size, device="cpu")).flatten()[:length].to(cache.device)
        return rows.index_select(0, indices)[:, :cache.shape[-1]].contiguous()
    split = cache.shape[1] // block_size
    pieces = []
    for page, index in enumerate(ids):
        physical, subpage = divmod(index, split)
        offset = subpage * block_size
        width = min(block_size, length - page * block_size)
        pieces.append(cache[physical, offset:offset + width, 0].clone(memory_format=torch.contiguous_format))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def read_pages(cache, table, length, *, block_size=None):
    """复制有效分页前缀，返回独立 [L,R] 快照。"""
    block_size, ids = _pages(cache, table, length, block_size)
    return _copy_prefix(cache, ids, length, block_size, _flat_rows(cache) if len(ids) >= 16 else None)


def _gather_dense(history, indices):
    if history.is_contiguous() and indices.numel() >= 2048:
        return history.index_select(0, indices.flatten()).view(*indices.shape, history.shape[-1])
    return history[indices]


class PagedLatent:
    """借用目标存储和自有页表元数据；release 后不得再次读取。

    仅用于 eager、同步单请求：目标不能在 gather 期间推进或回收页。
    gather 的结果拥有独立存储，历史引用本身不是跨步骤快照。
    """

    ndim = 2

    def __init__(self, cache, table, length, *, block_size=None, mode="auto", counters=None):
        if mode not in ("auto", "snapshot", "paged"):
            raise ValueError("history read mode must be auto, snapshot or paged")
        self.block_size, self._ids = _pages(cache, table, length, block_size)
        self.shape = (length, cache.shape[-1])
        self._cache, self._rows = cache, _flat_rows(cache)
        self.mode, self.counters = mode, counters
        self.last_path = None
        if mode == "paged" and self._rows is None:
            raise ValueError("paged history requires a supported contiguous physical row pitch")

    def release(self):
        self._cache = self._rows = None
        self._ids = ()
        self.counters = None

    def gather(self, indices):
        cache = self._cache
        if cache is None:
            raise RuntimeError("paged history was released after synchronous draft execution")
        if (indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64)
                or indices.device != cache.device):
            raise ValueError("history indices must be a device-local integer [K,S] tensor")
        length, dim = self.shape
        if not indices.numel():
            return cache.new_empty((*indices.shape, dim))
        if length == 0:
            raise IndexError("cannot select from an empty committed history")
        # 本机计时中短前缀的映射开销高于复制；长前缀才选择省内存的按页读取。
        paged = self._rows is not None and (self.mode == "paged" or self.mode == "auto"
                and length >= max(32768, indices.numel()))
        path = "paged" if paged else "snapshot"
        self.last_path = path
        if self.counters is not None:
            self.counters[path] += 1
        if not paged:
            return _gather_dense(_copy_prefix(cache, self._ids, length, self.block_size, self._rows), indices)
        starts = _starts(cache, self._ids, self.block_size).to(cache.device)
        pages = torch.div(indices, self.block_size, rounding_mode="floor")
        physical = starts.index_select(0, pages.flatten()).view_as(indices) + indices.remainder(self.block_size)
        # 最后一个有效页也可能含未提交的行；越界索引必须失败，不能读到它们。
        physical = torch.where(indices < length, physical, self._rows.shape[0])
        selected = self._rows.index_select(0, physical.flatten())[:, :dim].contiguous()
        return selected.view(*indices.shape, dim)


def gather_history(history, indices):
    """保持索引顺序与重复项；返回每行 query 自己的 [K,S,R]。"""
    return history.gather(indices) if isinstance(history, PagedLatent) else _gather_dense(history, indices)
