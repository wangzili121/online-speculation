import pytest
import torch

from blockspec.pypto_qwen38.gdn import recurrence_golden, convolution_golden


def test_one_step_matches_explicit_outer_product_update_and_preserves_history():
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 3, 8) for _ in range(3)]
    g, b = -torch.rand(1, 3), torch.rand(1, 3)
    initial = torch.randn(3, 8, 8)
    before = initial.clone()
    actual, state = recurrence_golden(q, k, v, g, b, initial, 1)
    for head in range(3):
        kh = k[0, head] / (k[0, head].square().sum() + 1e-6).sqrt()
        qh = q[0, head] / (q[0, head].square().sum() + 1e-6).sqrt() / 8 ** .5
        decayed = initial[head] * g[0, head].exp()
        expected = decayed + torch.outer((v[0, head] - decayed @ kh) * b[0, head], kh)
        torch.testing.assert_close(state[head], expected)
        torch.testing.assert_close(actual[0, head], expected @ qh)
    assert torch.equal(initial, before)


@pytest.mark.parametrize("boundary", [1, 7, 16, 17, 31])
def test_splitting_the_block_continues_from_the_returned_state(boundary):
    torch.manual_seed(17)
    q, k, v = [torch.randn(32, 3, 8).bfloat16() for _ in range(3)]
    g, b, initial = -torch.rand(32, 3), torch.rand(32, 3), torch.randn(3, 8, 8)
    full, state = recurrence_golden(q, k, v, g, b, initial, 32)
    first, mid = recurrence_golden(q[:boundary], k[:boundary], v[:boundary], g[:boundary], b[:boundary], initial, boundary)
    second, final = recurrence_golden(q[boundary:], k[boundary:], v[boundary:], g[boundary:], b[boundary:], mid, 32 - boundary)
    assert torch.equal(torch.cat((first, second)), full)
    assert torch.equal(final, state)


def test_unused_poisoned_rows_do_not_enter_outputs_or_final_state():
    q, k, v = [torch.zeros(8, 1, 8) for _ in range(3)]
    g, b, initial = torch.zeros(8, 1), torch.ones(8, 1), torch.ones(1, 8, 8)
    for tensor in (q, k, v, g, b):
        tensor[3:] = float("nan")
    result, state = recurrence_golden(q, k, v, g, b, initial, 3)
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(state, initial)


@pytest.mark.parametrize("options", [{"backend": "unknown"}, {"value_tile": 8}])
def test_invalid_kernel_options_fail_before_loading_the_runtime(options):
    from blockspec.pypto_qwen38.gdn_programs import make_recurrence
    with pytest.raises(ValueError):
        make_recurrence(None, **options)


def test_model_rejects_unknown_recurrence_before_loading_the_runtime():
    from blockspec.pypto_qwen38.programs import make_programs
    from blockspec.pypto_qwen38.draft import NeuralDraft
    with pytest.raises(ValueError, match="gdn_backend"):
        make_programs(None, gdn_backend="unknown")
    with pytest.raises(ValueError, match="gdn_backend"):
        NeuralDraft(None, None, gdn_backend="unknown")


def test_convolution_uses_three_previous_inputs_and_preserves_initial_state():
    source = torch.tensor([[5., 6.], [7., 8.]])
    history = torch.tensor([[99., 99.], [1., 2.], [3., 4.], [5., 6.]])
    before = history.clone()
    weight = torch.ones(4, 2)
    expected = torch.nn.functional.silu(torch.tensor([[14., 18.], [20., 24.]]))
    torch.testing.assert_close(convolution_golden(source, weight, history), expected)
    assert torch.equal(history, before)
