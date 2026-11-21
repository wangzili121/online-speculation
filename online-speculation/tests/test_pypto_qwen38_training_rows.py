"""缩小物理训练行数不能改变有效块的前向、监督或参数梯度。"""

from dataclasses import replace

import pytest
import torch

from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.training_programs import reduction_tile


@pytest.mark.parametrize("objective", ("tv", "kl", "log_overlap"))
@pytest.mark.parametrize("past,width,valid", ((0, 8, 7), (7, 8, 3), (32, 16, 15)))
def test_training_rows_preserve_active_forward_and_vjp(objective, past, width, valid):
    torch.set_num_threads(4)
    shape = SuffixShape(hidden=128, intermediate=256, vocabulary=128,
                        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32)
    old = LastLayerGraph(shape, objective=objective)
    compact = LastLayerGraph(replace(shape, block=16), objective=objective)
    generator = torch.Generator().manual_seed(257)
    inputs = {}
    for name, value in old.inputs.items():
        tensor = torch.randn(value.shape, generator=generator) * .03
        if "gamma" in name:
            tensor += 1
        inputs[name] = tensor if value.dtype == "fp32" else tensor.bfloat16()
    short = {name: inputs[name][:value.shape[0]].clone() for name, value in compact.inputs.items()}
    old_values = evaluate(old, inputs, past=past, width=width, valid=valid)
    new_values = evaluate(compact, short, past=past, width=width, valid=valid)
    torch.testing.assert_close(new_values[compact.logits.name][:width], old_values[old.logits.name][:width],
                               atol=0, rtol=0)
    torch.testing.assert_close(new_values[compact.losses.name].sum(0), old_values[old.losses.name].sum(0),
                               atol=0, rtol=0)
    for name in old.gradients:
        torch.testing.assert_close(new_values[compact.gradients[name].name], old_values[old.gradients[name].name],
                                   atol=1e-8, rtol=1e-5, msg=lambda msg, name=name: name + ": " + msg)


def test_token_reduction_preserves_old_tiles_and_accepts_sixteen():
    assert [reduction_tile(k, 64) for k in (16, 32, 64, 128)] == [16, 32, 64, 64]
    for block in (True, 8, 17, 64):
        with pytest.raises(ValueError):
            SuffixShape(block=block)


@pytest.mark.parametrize("rows", (16, 32))
def test_combined_training_registers_program_and_geometry(monkeypatch, rows):
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms
    from blockspec.pypto_qwen38 import training_pipeline

    graph = LastLayerGraph(SuffixShape(block=rows))
    compiled = object()

    class Entry:
        def compile(self, *arguments, config):
            assert arguments[-4:] == (0, rows, rows - 1, rows - 1)
            return compiled

    monkeypatch.setattr(training_pipeline, "make_pipeline",
                        lambda graph, **kwargs: (Entry(), tuple(graph.inputs.values())))
    programs = TrainingPrograms(graph, None, combined=True)
    assert programs.programs["suffix"] is compiled


def test_compact_online_graph_keeps_parameter_and_history_contract():
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    old = TP2Online(None, device=0, last_layers=1).build_training_graph()
    compact = TP2Online(None, device=0, last_layers=1, training_rows=16).build_training_graph()
    assert old.parameters == compact.parameters
    assert old.histories == compact.histories
    assert compact.inputs["hidden"].shape == (16, 5120)
    assert compact.inputs["teacher"].shape == (16, 248320)
    assert old.inputs["teacher"].shape == (32, 248320)


@pytest.mark.parametrize("rows,layers", ((True, 1), (8, 1), (17, 1), (16, 4)))
def test_compact_online_rejects_unsupported_geometry(rows, layers):
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    with pytest.raises(ValueError):
        TP2Online(None, device=0, last_layers=layers, training_rows=rows)


def test_compact_feedback_reuses_registered_storage_without_reading_tail():
    from collections import Counter
    from types import SimpleNamespace
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    learner = TP2Online(None, device=0, last_layers=1, training_rows=16)
    learner.feedback_hidden = torch.full((32, 8), -99., dtype=torch.bfloat16).share_memory_()
    learner.runner = SimpleNamespace(spec_stats=Counter())
    snapshots = []

    def read(rank, source, destination):
        assert rank == 0 and destination.shape == (16, 8)
        assert destination.data_ptr() == learner.feedback_hidden.data_ptr()
        destination.fill_(source)

    learner.read = read
    learner.upload = lambda name, value: snapshots.append((name, value.clone()))
    learner.capture_hidden(3.)
    learner.capture_hidden(7.)
    assert all(name == "hidden" for name, _ in snapshots)
    assert torch.all(snapshots[0][1] == 3) and torch.all(snapshots[1][1] == 7)
    assert torch.all(learner.feedback_hidden[16:] == -99)
