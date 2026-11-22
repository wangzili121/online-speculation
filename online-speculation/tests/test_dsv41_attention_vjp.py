import torch

from blockspec.dsv41_attention_vjp import attention_forward, attention_vjp


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
