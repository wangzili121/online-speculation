import pytest
import torch

from blockspec.dsv41_tp_gradient_layout import (
    DSV41TPGradientLayout,
    PARAMETERS,
    validate_artifact_shapes,
)


CONFIG = {
    "hidden_size": 128,
    "q_lora_rank": 32,
    "num_attention_heads": 8,
    "head_dim": 16,
    "o_groups": 4,
    "o_lora_rank": 8,
}


def rows(dp=2):
    result = {}
    for lane in range(4):
        value = float(lane + 1)
        result[dp * 4 + lane] = {
            "wq_a.weight": torch.full((128, 32), value),
            "q_norm.weight": torch.full((32,), value),
            "wq_b.weight": torch.full((32, 32), value),
            "wkv.weight": torch.full((128, 16), value),
            "kv_norm.weight": torch.full((16,), value),
            "attn_sink": torch.full((2,), value),
            "wo_a.weight": torch.full((1, 8, 32), value),
            "wo_b.weight": torch.full((8, 128), value),
        }
    return result


def test_tp4_gradient_assembly_restores_exchange_artifact_layout():
    assembled = DSV41TPGradientLayout().assemble(rows())
    assert validate_artifact_shapes(assembled, CONFIG)
    # Replicated branches sum all four lanes.
    assert torch.all(assembled["wq_a.weight"] == 10)
    assert torch.all(assembled["wkv.weight"] == 10)
    # True shards retain lane order and are transformed back from execution
    # orientation to the public head checkpoint orientation.
    assert assembled["attn_sink"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert assembled["wq_b.weight"][:, 0].tolist() == [1] * 32 + [2] * 32 + [3] * 32 + [4] * 32
    assert assembled["wo_b.weight"][0].tolist() == [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8


def test_tp4_gradient_assembly_rejects_partial_or_mixed_dp_partition():
    layout = DSV41TPGradientLayout()
    broken = rows()
    del broken[8]
    with pytest.raises(ValueError, match="exactly four"):
        layout.assemble(broken)
    mixed = rows()
    mixed[12] = mixed.pop(8)
    with pytest.raises(ValueError, match="contiguous"):
        layout.assemble(mixed)


def test_tp4_gradient_assembly_rejects_parameter_abi_drift():
    broken = rows()
    broken[8].pop(PARAMETERS[0])
    with pytest.raises(ValueError, match="gradient ABI"):
        DSV41TPGradientLayout().assemble(broken)
