"""CPU dispatch tests; numerical NPU equivalence is checked on loaded weights."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("quantized", [(True, True), (False, False), (True, False)])
def test_qkv_reuses_quantization_only_for_two_dynamic_projections(monkeypatch, quantized):
    source = Path(__file__).resolve().parents[1] / "src/blockspec/vllm_ascend/dsv4.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_qkv_readonly")
    dynamic_quant = Mock(side_effect=lambda x: (x, torch.ones(x.shape[0])))
    quant_matmul = Mock(side_effect=lambda x, weight, scale, **kwargs: x @ weight)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_dynamic_quant=dynamic_quant, npu_quant_matmul=quant_matmul))
    rotary = Mock()
    namespace = {
        "torch": SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(
            inplace_partial_rotary_mul=rotary))),
        "_dsa_runtime": SimpleNamespace(_is_w8a8_dynamic=lambda module: module.dynamic),
        "_linear": lambda module, x: x @ module.weight,
        "_query_readonly": lambda impl, x, cos, sin, q_a: (q_a, q_a, None),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    q = SimpleNamespace(dynamic=quantized[0], weight=torch.ones(4, 2), weight_scale=None, bias=None)
    kv = SimpleNamespace(dynamic=quantized[1], weight=2 * torch.ones(4, 2), weight_scale=None, bias=None)
    impl = SimpleNamespace(wq_a=q, wkv=kv, kv_norm=lambda x: x, head_dim=2, nope_head_dim=0)
    x = torch.ones(3, 4)
    query, _, _, local = namespace["_qkv_readonly"](impl, x, None, None)
    assert torch.equal(query, x @ q.weight)
    assert torch.equal(local, (x @ kv.weight).view(3, 1, 2))
    assert dynamic_quant.call_count == int(all(quantized))
    assert quant_matmul.call_count == 2 * int(all(quantized))
    assert rotary.call_count == 1


@pytest.mark.parametrize("page,backend,selected", [(32, "auto", "sparse"),
    (128, "auto", "paged"), (128, "sparse", "sparse"), (32, "gather", "gather")])
def test_c128_backend_preserves_all_history_rows(page, backend, selected):
    source = Path(__file__).resolve().parents[1] / "src/blockspec/vllm_ascend/dsv4.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_c128_segment")
    sparse, fia, gather = Mock(return_value="sparse"), Mock(return_value="fia"), Mock(return_value="kv")
    namespace = {"torch": torch, "os": os, "BLOCK_SIZE": 32,
                 "_sparse_history_segment": sparse, "_fia_segment": fia, "gather_paged_cache": gather}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    query = torch.empty(2, 32, 8, 512)
    cache = torch.empty(4, page, 1, 512)
    lengths, table = torch.tensor([0, 5]), torch.tensor([[0, 1], [2, 3]])
    call = namespace["_c128_segment"]
    call(query, cache, table, lengths, 5, 512**-0.5, backend=backend)
    assert sparse.called == (selected == "sparse")
    assert gather.called == (selected == "gather")
    if selected == "sparse":
        indices = sparse.call_args.args[3]
        assert indices.shape == (2, 32, 128)
        assert (indices[0] == -1).all()
        assert torch.equal(indices[1, :, :5], torch.arange(5).expand(32, -1))
        assert (indices[1, :, 5:] == -1).all()
    elif selected == "paged":
        assert fia.call_args.kwargs["block_table"] is table
    with pytest.raises(ValueError, match="C128_BACKEND"):
        call(query, cache, table, lengths, 5, 1, backend="unsupported")
