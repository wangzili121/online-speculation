"""Orthrus 分页 Attention 的接口与 Torch 参考，不依赖 PyPTO 环境。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionShape:
    block_size: int = 32
    q_heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    context: int = 512
    page_size: int = 16
    query_tile: int = 16

    def __post_init__(self):
        values = tuple(vars(self).values())
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("attention dimensions must be positive integers")
        if (self.q_heads % self.kv_heads or self.context % self.page_size
                or self.block_size % self.query_tile or self.block_size > self.context):
            raise ValueError("incompatible GQA, page, or query-tile dimensions")

    @property
    def query_width(self):
        return self.q_heads * self.head_dim

    @property
    def kv_width(self):
        return self.kv_heads * self.head_dim


def validate_inputs(shape, query, key, value, gate, key_pool, value_pool, table, past, width):
    if type(past) is not int or type(width) is not int or not 0 <= past <= shape.context - width:
        raise ValueError("invalid history boundary")
    if not 1 <= width <= shape.block_size:
        raise ValueError("invalid active block width")
    for tensor, expected in ((query, (shape.block_size, shape.query_width)),
                             (gate, (shape.block_size, shape.query_width)),
                             (key, (shape.block_size, shape.kv_width)),
                             (value, (shape.block_size, shape.kv_width))):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"expected tensor shape {expected}")
    if (key_pool.ndim != 4 or value_pool.shape != key_pool.shape
            or key_pool.shape[0] != shape.kv_heads
            or tuple(key_pool.shape[2:]) != (shape.page_size, shape.head_dim)):
        raise ValueError("invalid paged KV layout")
    pages = (past + shape.page_size - 1) // shape.page_size
    if table.ndim != 1 or table.numel() < pages or table.dtype not in (torch.int32, torch.int64):
        raise ValueError("invalid block table")
    used = table[:pages]
    if pages and (used.min().item() < 0 or used.max().item() >= key_pool.shape[1]):
        raise ValueError("physical KV page is out of range")


def attention_golden(shape, query, key, value, gate, key_pool, value_pool, table, past, width,
                     *, native_rounding=False):
    """所有 query 读取同一段真实历史和整个有效块；不修改任何输入。

    默认以 FP32 SDPA 表达数学参考。native_rounding 复现原生 Attention
    的 BF16 中间舍入，供替换实现做兼容性比较；不以设备输出生成参考。
    """
    validate_inputs(shape, query, key, value, gate, key_pool, value_pool, table, past, width)
    pages = (past + shape.page_size - 1) // shape.page_size
    history = [pool.index_select(1, table[:pages].long()).flatten(1, 2)[:, :past]
               for pool in (key_pool, value_pool)]
    block = [x[:width].reshape(width, shape.kv_heads, shape.head_dim).transpose(0, 1)
             for x in (key, value)]
    keys, values = [torch.cat((old, new), dim=1) for old, new in zip(history, block, strict=True)]
    repeats = shape.q_heads // shape.kv_heads
    keys, values = [x.repeat_interleave(repeats, dim=0).float() for x in (keys, values)]
    q = query.reshape(shape.block_size, shape.q_heads, shape.head_dim).transpose(0, 1).float()
    if native_rounding:
        scores = (q @ keys.transpose(-1, -2)).bfloat16().float()
        scores = (scores * shape.head_dim ** -.5).bfloat16().float()
        probabilities = scores.softmax(-1).bfloat16().float()
        # 原生 PV 按 page 累加 FP32，再转 BF16。
        attended = torch.zeros_like(q)
        for start in range(0, past + width, shape.page_size):
            attended += probabilities[..., start:start + shape.page_size] @ values[:, start:start + shape.page_size]
        attended = attended.bfloat16().float()
        activated_gate = gate.float().sigmoid().bfloat16().float()
    else:
        attended = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
        activated_gate = gate.float().sigmoid()
    attended = attended.transpose(0, 1).reshape(shape.block_size, shape.query_width)
    return (attended * activated_gate).to(query.dtype)
