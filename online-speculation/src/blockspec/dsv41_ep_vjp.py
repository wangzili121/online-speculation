"""Deterministic EP16 packet contract for the frozen DSV4.1 MoE VJP.

The serving forward records global expert IDs in token/slot order.  Backward
must send each selected-expert gradient to the rank that owns that expert and
return one input gradient to the original token/slot without re-running Top-K.
This module fixes that ordering independently of the HCCL transport backend.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertRoute:
    token: int
    slot: int
    expert: int
    owner: int
    local_expert: int


class DSV41EPBackwardPlan:
    """Stable route packets for 384 experts sharded over EP16 ranks."""

    def __init__(self, route_ids, *, experts=384, ep_size=16, topk=6):
        if (
            not torch.is_tensor(route_ids) or route_ids.ndim != 2
            or route_ids.dtype not in (torch.int32, torch.int64)
            or route_ids.shape[1] != topk or experts % ep_size
            or bool(((route_ids < 0) | (route_ids >= experts)).any())
        ):
            raise ValueError("expected valid [rows, 6] global DSV4.1 expert IDs")
        if any(torch.unique(row).numel() != topk for row in route_ids):
            raise ValueError("each token must select six distinct experts")
        self.rows, self.topk = route_ids.shape
        self.experts, self.ep_size = experts, ep_size
        self.experts_per_rank = experts // ep_size
        self.route_ids = route_ids.detach().cpu().long().contiguous()
        self.routes = tuple(
            ExpertRoute(
                token, slot, expert,
                expert // self.experts_per_rank,
                expert % self.experts_per_rank,
            )
            for token, row in enumerate(self.route_ids.tolist())
            for slot, expert in enumerate(row)
        )
        self.by_owner = {
            owner: tuple(route for route in self.routes if route.owner == owner)
            for owner in range(ep_size)
        }

    def dispatch(self, route_values):
        """Pack ``[rows, 6, width]`` values in deterministic owner order."""
        if (
            not torch.is_tensor(route_values) or route_values.ndim < 2
            or tuple(route_values.shape[:2]) != (self.rows, self.topk)
        ):
            raise ValueError("route values must start with the replay [rows, 6] geometry")
        return {
            owner: route_values[
                torch.tensor([route.token for route in routes], device=route_values.device),
                torch.tensor([route.slot for route in routes], device=route_values.device),
            ].contiguous()
            for owner, routes in self.by_owner.items() if routes
        }

    def metadata(self, owner):
        """Return ``[token, slot, global expert, local expert]`` for one peer."""
        if type(owner) is not int or not 0 <= owner < self.ep_size:
            raise ValueError("EP owner rank is outside the world")
        return torch.tensor([
            (route.token, route.slot, route.expert, route.local_expert)
            for route in self.by_owner[owner]
        ], dtype=torch.int32).reshape(-1, 4)

    def restore(self, owner_values):
        """Undo owner packing and restore one gradient per token/route slot."""
        if set(owner_values) != {owner for owner, routes in self.by_owner.items() if routes}:
            raise ValueError("expert-owner responses do not cover the fixed route plan")
        sample = next(iter(owner_values.values()))
        if not torch.is_tensor(sample) or sample.ndim < 1:
            raise ValueError("expert-owner responses must be tensors")
        output = torch.empty(
            (self.rows, self.topk, *sample.shape[1:]), dtype=sample.dtype, device=sample.device,
        )
        for owner, values in owner_values.items():
            routes = self.by_owner[owner]
            if (
                not torch.is_tensor(values) or values.shape[0] != len(routes)
                or values.shape[1:] != sample.shape[1:] or values.dtype != sample.dtype
                or values.device != sample.device
            ):
                raise ValueError("expert-owner response changed packet shape, dtype or device")
            for index, route in enumerate(routes):
                output[route.token, route.slot].copy_(values[index])
        return output

    def reduce_to_tokens(self, owner_values):
        """Return the sum of all six expert input gradients for each token."""
        return self.restore(owner_values).sum(dim=1)
