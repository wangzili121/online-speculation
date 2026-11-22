import pytest
import torch

from blockspec.dsv41_ep_vjp import DSV41EPBackwardPlan


def test_ep16_backward_plan_round_trips_token_slot_order():
    ids = torch.tensor([
        [0, 23, 24, 25, 383, 192],
        [48, 49, 72, 96, 120, 144],
    ], dtype=torch.int32)
    plan = DSV41EPBackwardPlan(ids)
    values = torch.arange(2 * 6 * 5, dtype=torch.float32).reshape(2, 6, 5)
    packets = plan.dispatch(values)
    torch.testing.assert_close(plan.restore(packets), values)
    torch.testing.assert_close(plan.reduce_to_tokens(packets), values.sum(1))
    assert plan.metadata(0).tolist() == [[0, 0, 0, 0], [0, 1, 23, 23]]
    assert plan.metadata(1).tolist() == [[0, 2, 24, 0], [0, 3, 25, 1]]


def test_ep16_plan_keeps_repeated_expert_across_tokens_as_distinct_packets():
    ids = torch.tensor([
        [7, 24, 48, 72, 96, 120],
        [7, 25, 49, 73, 97, 121],
    ], dtype=torch.int64)
    plan = DSV41EPBackwardPlan(ids)
    assert plan.metadata(0).tolist() == [[0, 0, 7, 7], [1, 0, 7, 7]]
    values = torch.randn(2, 6, 3)
    responses = {owner: packet + owner for owner, packet in plan.dispatch(values).items()}
    restored = plan.restore(responses)
    for route in plan.routes:
        torch.testing.assert_close(
            restored[route.token, route.slot], values[route.token, route.slot] + route.owner,
        )


def test_ep16_plan_rejects_duplicate_experts_within_one_token():
    with pytest.raises(ValueError, match="distinct"):
        DSV41EPBackwardPlan(torch.tensor([[1, 2, 3, 4, 5, 5]], dtype=torch.int32))
