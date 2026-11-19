"""用不同形状的小张量核对所有发布 bank 的拆分和原始参数格式。"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.tp2_head_checkpoint import export_served_head, restore_tensor_layout


def key(layer, name):
    return f"layers.{layer}.attention.draft.{name}"


def fixture():
    tensors = {}
    banks = [torch.arange(3 * n * 2, dtype=torch.bfloat16).reshape(3 * n, 2)
             for n in (5, 3, 5, 4, 4)]
    banks += [torch.arange(3 * 4 * 5, dtype=torch.bfloat16).reshape(12, 5),
              torch.arange(12, dtype=torch.float32).reshape(3, 4),
              torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
              torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)]
    for layer in (60, 61, 62):
        for name, shape in (("in_proj_qkv.weight", (5, 2)), ("in_proj_z.weight", (3, 2)),
                            ("out_proj.weight", (5, 2)), ("in_proj_a.weight", (2, 2)),
                            ("in_proj_b.weight", (2, 2)), ("conv1d.weight", (5, 1, 4)),
                            ("A_log", (2,)), ("dt_bias", (2,)), ("norm.weight", (2,))):
            tensors[key(layer, name)] = torch.zeros(shape, dtype=torch.float32 if name == "A_log" else torch.bfloat16)
    for i, name in enumerate(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        tensors[key(63, name)] = torch.zeros(2, 2, dtype=torch.bfloat16)
        banks.append(torch.full((2, 2), i + 4, dtype=torch.bfloat16))
    for name in ("q_norm.weight", "k_norm.weight"):
        tensors[key(63, name)] = torch.zeros(2, dtype=torch.bfloat16)
        banks.append(torch.tensor([[1.25, .75]], dtype=torch.float32))
    tensors[key(63, "o_proj.weight")] = torch.zeros(2, 2, dtype=torch.bfloat16)
    banks.append(torch.full((2, 2), 9, dtype=torch.bfloat16))
    return tensors, banks


def test_all_three_gdn_layers_keep_their_own_slices_and_remove_padding():
    tensors, banks = fixture()
    actual = restore_tensor_layout(tensors, banks)
    for i, layer in enumerate((60, 61, 62)):
        assert torch.equal(actual[key(layer, "in_proj_qkv.weight")], banks[0][5 * i:5 * (i + 1)])
        assert torch.equal(actual[key(layer, "in_proj_a.weight")], banks[3][4 * i:4 * i + 2])
        assert torch.equal(actual[key(layer, "conv1d.weight")].squeeze(1).T, banks[5][4 * i:4 * (i + 1)])
        assert torch.equal(actual[key(layer, "A_log")], banks[6][i, :2])
        assert torch.equal(actual[key(layer, "norm.weight")], banks[8][i])


def test_last_one_export_preserves_original_gdn_tensors():
    tensors, banks = fixture()
    original = tensors[key(61, "in_proj_qkv.weight")]
    result = restore_tensor_layout(tensors, [None] * 9 + banks[9:])
    assert result[key(61, "in_proj_qkv.weight")] is original
    assert torch.equal(result[key(63, "q_norm.weight")], torch.tensor([.25, -.25], dtype=torch.bfloat16))
    assert torch.equal(result[key(63, "o_proj.weight")], banks[14])


@pytest.mark.parametrize("active,pending,stride,published", [(True, None, None, 1),
    (False, object(), None, 1), (False, None, 16, 1), (False, None, None, 0)])
def test_export_rejects_live_or_unpublished_state_before_writing(tmp_path, active, pending, stride, published):
    trainer = SimpleNamespace(runner=SimpleNamespace(requests={"r": {}} if active else {}),
                              stride=stride, pending_update=pending, published_version=published, initial_step=0)
    with pytest.raises(RuntimeError, match="idle serving"):
        export_served_head(trainer, tmp_path / "head")
    assert not (tmp_path / "head").exists()
