import importlib.util
import math
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/replay_dsv4_c4_operator.py"
SPEC = importlib.util.spec_from_file_location("operator_replay", PATH)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


@pytest.mark.parametrize("transform", [REPLAY.reverse_physical_pages, REPLAY.canonical_physical_pages,
                                       REPLAY.compact_physical_pages])
def test_page_repacking_preserves_logical_kv_and_aliases(transform):
    cache = torch.arange(20).reshape(5, 2, 1, 2)
    table = torch.tensor([[3, 1, -1], [1, 4, 3]], dtype=torch.int32)
    kwargs = {kind + suffix: value for kind in ("ori", "cmp")
              for suffix, value in (("_kv", cache), ("_block_table", table))}
    repacked = transform(kwargs)
    for kind in ("ori", "cmp"):
        actual_table = repacked[kind + "_block_table"]
        valid = table >= 0
        assert torch.equal(cache[table[valid].long()], repacked[kind + "_kv"][actual_table[valid].long()])
        assert torch.equal(actual_table[~valid], table[~valid])
        assert actual_table[0, 1] == actual_table[1, 0]
    assert torch.equal(kwargs["ori_block_table"], table)


def test_reference_includes_sink_and_masks_unavailable_compressed_keys():
    snapshot = dict(args=(torch.ones(1, 1, 1),), kwargs=dict(
        cu_seqlens_q=torch.tensor([0, 1]), seqused_kv=torch.tensor([1]), ori_win_left=127,
        cmp_ratio=4, cmp_sparse_indices=torch.tensor([[[0, -1]]]),
        ori_kv=torch.tensor([[[[2.]]]]), cmp_kv=torch.tensor([[[[99.]]]]),
        ori_block_table=torch.tensor([[0]]), cmp_block_table=torch.tensor([[0]]),
        softmax_scale=1., sinks=torch.tensor([0.])), output=(torch.zeros(1, 1, 1),))
    result = REPLAY.reference_value(snapshot, 0, 0, 0)
    assert result["position"] == 0 and result["compressed_keys"] == 0
    assert result["fp64"] == pytest.approx(2 * math.exp(2) / (1 + math.exp(2)))


def test_tensor_description_handles_scalars_and_noncontiguous_views():
    for tensor in (torch.tensor(2.), torch.arange(12).reshape(3, 4).T):
        summary = REPLAY.tensor_summary(tensor)
        assert summary["shape"] == list(tensor.shape)
        assert len(summary["sha256"]) == 64
