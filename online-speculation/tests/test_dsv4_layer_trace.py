import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/trace_dsv4_native_layers.py"
SPEC = importlib.util.spec_from_file_location("layer_trace", PATH)
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def test_operator_snapshot_detaches_and_copies_nested_inputs():
    original = torch.randn(2, 3, requires_grad=True)
    snapshot = TRACE.snapshot_tree(dict(args=(original,), kwargs={"scale": 0.5, "metadata": None}))
    assert not snapshot["args"][0].requires_grad
    assert torch.equal(snapshot["args"][0], original)
    with torch.no_grad():
        original.zero_()
    assert not torch.equal(snapshot["args"][0], original)
    assert snapshot["kwargs"] == {"scale": 0.5, "metadata": None}
    with pytest.raises(TypeError, match="unsupported operator input"):
        TRACE.snapshot_tree(object())


def test_block_trace_selects_request_not_duplicate_position():
    positions = [512, 513, 512, 513, 514]
    assert TRACE.select_trace_rows(positions, 512, [0, 2, 5], [127, 511], 511, 512) == (2, 5, 1)
    assert TRACE.select_trace_rows(positions, 512, [0, 2, 5], [127, 511]) is None
    assert TRACE.select_trace_rows(positions, 514, [0, 2, 5], [127, 511]) == (4, 5, 1)


def test_block_trace_rejects_wrong_block_or_ambiguous_request():
    assert TRACE.select_trace_rows([512, 513], 513, [0, 2], [511], 511, 513) is None
    assert TRACE.select_trace_rows([512, 512], 512, [0, 1, 2], [511, 511], 511, 512) is None
    assert TRACE.select_trace_rows([512], 512, [0, 0, 1], [511, 127], 511, 512) is None


@pytest.mark.parametrize("repeat", [False, True])
def test_attention_replay_is_opt_in_and_does_not_replace_output(repeat):
    class Operator:
        calls = 0

        def npu_sparse_attn_sharedkv(self, query):
            self.calls += 1
            return (query + self.calls,)

    operator = Operator()
    worker = TRACE.LayerTraceWorker()
    worker._layer_trace = dict(active=True, layer="layer", tokens=2, selector=slice(0, 2),
                               counts={}, tensors={}, shapes={}, wrapped=[], last_position=1,
                               repeat_attention=repeat)
    worker._wrap_trace(operator, "npu_sparse_attn_sharedkv", "op.npu_sparse_attn_sharedkv")
    query = torch.zeros(2, 3)
    output = operator.npu_sparse_attn_sharedkv(query)[0]
    assert torch.equal(output, query + 1)
    assert operator.calls == (2 if repeat else 1)
    snapshots = worker._layer_trace["tensors"]
    key = "layer.op.npu_sparse_attn_sharedkv.0."
    assert torch.equal(snapshots[key + "output/0"], query + 1)
    if repeat:
        assert torch.equal(snapshots[key + "repeat_output/0"], query + 2)
    else:
        assert key + "repeat_output/0" not in snapshots
