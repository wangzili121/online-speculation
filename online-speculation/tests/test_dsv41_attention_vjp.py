import torch

from blockspec.dsv41_attention_vjp import (
    attention_forward,
    attention_vjp,
    c1a_sparse_forward,
    c1a_sparse_vjp,
)


def test_explicit_attention_vjp_matches_autograd_and_splits_frozen_history():
    torch.manual_seed(41)
    rows, heads, dimension, history = 32, 4, 16, 11
    total = history + rows
    query = torch.randn(rows, heads, dimension, dtype=torch.float64, requires_grad=True)
    key = torch.randn(total, dimension, dtype=torch.float64, requires_grad=True)
    value = torch.randn(total, dimension, dtype=torch.float64, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float64, requires_grad=True)
    allowed = torch.ones(rows, total, dtype=torch.bool)
    allowed[:, history - 2:history] = False
    upstream = torch.randn(rows, heads, dimension, dtype=torch.float64)

    output, gradients = attention_vjp(query, key, value, sink, allowed, upstream)
    expected, _ = attention_forward(query, key, value, sink, allowed)
    reference = torch.autograd.grad(expected, (query, key, value, sink), upstream)
    torch.testing.assert_close(output, expected)
    for actual, wanted in zip(
        (gradients.query, gradients.key, gradients.value, gradients.sink), reference, strict=True,
    ):
        torch.testing.assert_close(actual, wanted.float(), atol=2e-6, rtol=2e-6)

    block = gradients.trainable_block(history)
    assert block.key.shape == block.value.shape == (rows, dimension)
    torch.testing.assert_close(block.key, gradients.key[history:])
    torch.testing.assert_close(block.value, gradients.value[history:])


def test_c1a_sparse_vjp_matches_autograd_with_per_query_history():
    torch.manual_seed(42)
    rows, heads, dimension, history = 8, 3, 16, 13
    query = torch.randn(rows, heads, dimension, dtype=torch.float64, requires_grad=True)
    history_key = torch.randn(rows, history, dimension, dtype=torch.float64, requires_grad=True)
    history_value = torch.randn_like(history_key, requires_grad=True)
    block_key = torch.randn(rows, dimension, dtype=torch.float64, requires_grad=True)
    block_value = torch.randn_like(block_key, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float64, requires_grad=True)
    allowed = torch.ones(rows, history + rows, dtype=torch.bool)
    allowed[0, 2:5] = False
    allowed[1, :3] = False
    upstream = torch.randn(rows, heads, dimension, dtype=torch.float64)

    output, actual = c1a_sparse_vjp(
        query, history_key, history_value, block_key, block_value, sink, allowed, upstream,
    )
    expected, _ = c1a_sparse_forward(
        query, history_key, history_value, block_key, block_value, sink, allowed,
    )
    reference = torch.autograd.grad(
        expected, (query, block_key, block_value, sink), upstream,
    )
    torch.testing.assert_close(output, expected)
    for result, wanted in zip(
        (actual.query, actual.block_key, actual.block_value, actual.sink), reference, strict=True,
    ):
        torch.testing.assert_close(result, wanted.float(), atol=2e-6, rtol=2e-6)
