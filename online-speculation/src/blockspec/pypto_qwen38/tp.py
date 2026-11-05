"""Rank-local layout for the dense Qwen3.8-27B head and target history."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TPLayout:
    size: int = 1
    rank: int = 0

    def __post_init__(self):
        if type(self.size) is not int or self.size not in {1, 2, 4}:
            raise ValueError("Qwen3.8 draft layout requires TP1/TP2/TP4")
        if type(self.rank) is not int or not 0 <= self.rank < self.size:
            raise ValueError("TP rank must be in [0, size)")

    def interval(self, count):
        if type(count) is not int or count < 1 or count % self.size:
            raise ValueError("the head/channel count must be divisible by TP")
        width = count // self.size
        return self.rank * width, (self.rank + 1) * width

    @property
    def head_ranges(self):
        return {name: self.interval(count) for name, count in (
            ("full_query", 24), ("full_kv", 4), ("gdn_key", 16), ("gdn_value", 48))}

    def history_shapes(self, *, slots, pages):
        if any(type(n) is not int or n < 1 for n in (slots, pages)):
            raise ValueError("positive slot and physical page counts required")
        # 与 native 的 layer/head/page/token/dim 顺序一致，仅 head/channel 分片。
        kv = (16, 4 // self.size, pages, 16, 256)
        return dict(key=kv, value=kv, conv=(slots, 48, 4, 10240 // self.size),
                    recurrent=(slots, 48, 48 // self.size, 128, 128))

    def validate_history_shapes(self, shapes, *, slots, pages):
        expected = self.history_shapes(slots=slots, pages=pages)
        if set(shapes) != set(expected) or any(tuple(shapes[name]) != shape for name, shape in expected.items()):
            raise ValueError("target KV/GDN history does not match the rank-local TP layout")


def reshard_parameter(layer, name, *, source_tp, layout, read):
    """只读取当前 rank 的区间；read(checkpoint_rank, slices) 返回 CPU 张量。"""
    import torch
    from .head import KEY_WIDTH, VALUE_WIDTH, tensor_schema

    TPLayout(source_tp)
    if type(layer) is not int or not 0 <= layer < 64:
        raise ValueError("layer must be in 0..63")
    full_shape = tensor_schema(layer)[name]
    replicated = name in {"q_norm.weight", "k_norm.weight", "norm.weight"}
    dtype = None

    def checked(rank, selection, shape):
        nonlocal dtype
        value = read(rank, selection)
        permitted = {torch.bfloat16, torch.float32} if name == "A_log" else {torch.bfloat16}
        if (tuple(value.shape) != tuple(shape) or value.dtype not in permitted
                or dtype is not None and value.dtype != dtype or not value.isfinite().all()):
            raise ValueError(f"invalid rank-local draft parameter {layer}.{name}")
        dtype = value.dtype
        return value

    if replicated:
        values = [checked(rank, (slice(None),) * len(full_shape), full_shape) for rank in range(source_tp)]
        if any(not torch.equal(values[0], other) for other in values[1:]):
            raise ValueError(f"replicated draft parameter differs across ranks: {layer}.{name}")
        return values[0].contiguous()

    axis = 1 if name in {"o_proj.weight", "out_proj.weight"} else 0
    sections = ((KEY_WIDTH, KEY_WIDTH, VALUE_WIDTH)
                if name in {"in_proj_qkv.weight", "conv1d.weight"} else (full_shape[axis],))
    pieces, offset = [], 0
    for width in sections:
        begin, end = layout.interval(width)
        shard_width = width // source_tp
        for rank in range(begin // shard_width, (end - 1) // shard_width + 1):
            start, stop = max(begin, rank * shard_width), min(end, (rank + 1) * shard_width)
            selection = [slice(None)] * len(full_shape)
            selection[axis] = slice(offset + start - rank * shard_width,
                                    offset + stop - rank * shard_width)
            shape = list(full_shape)
            shape[axis] = stop - start
            pieces.append(checked(rank, tuple(selection), shape))
        offset += shard_width
    result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=axis)
    if tuple(result.shape) != tensor_schema(layer, layout.size)[name]:
        raise ValueError(f"incorrect TP shard shape for {layer}.{name}")
    return result.contiguous()
