"""适配器更新与完整 head 发布之间的原子性和资源边界。"""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("serving.runtime")

from blockspec.pypto_qwen38.oneserving_tp2_lora import TP2LoRAOnline
from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer


@pytest.mark.parametrize("kwargs", [dict(adapter_rank=32), dict(adapter_rank=True),
    dict(adapter_seed=-1), dict(parameter_scope="norms")])
def test_incompatible_adapter_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TP2LoRAOnline(None, device=0, last_layers=1, **kwargs)


def test_optimizer_checkpoint_is_not_mistaken_for_adapter_state():
    with pytest.raises(ValueError, match="fresh Adam"):
        TP2LoRAOnline("dense-checkpoint", device=0, last_layers=4)
    with pytest.raises(ValueError, match="new learner"):
        TP2LoRAOnline(None, device=0, last_layers=1).load_head("other-head")


def test_publication_always_renders_the_inactive_full_weight_view():
    trainer = TP2LoRAOnline(None, device=0, last_layers=1, publish_every=4)
    layout, buffers = object(), (object(), object())
    trainer.optimizer = NS(index=0, layout=layout)
    called = []
    trainer.render_merged = lambda index, values: called.append((index, values)) or ("full-layout", "full-buffers")
    assert trainer.publication_inputs(layout, buffers) == ("full-layout", "full-buffers")
    assert called == [(1, buffers)]
    trainer.optimizer.index = 1
    trainer.publication_inputs(layout, buffers)
    assert called[-1] == (0, buffers)
    assert not trainer.can_reuse_publication_execution(True)


def test_nonfinite_merged_weight_cannot_commit_even_between_publications():
    trainer = TP2LoRAOnline(None, device=0, last_layers=1, publish_every=4)
    trainer.initial_step = 0
    optimizer = object.__new__(SuffixOptimizer)
    optimizer.steps = optimizer.index = 0
    optimizer.layout = object()
    optimizer.state = [object() for _ in range(3)]
    optimizer.candidate = [object() for _ in range(3)]
    optimizer.execution = [(object(), object()) for _ in range(2)]
    optimizer.programs = NS(adam=object())
    optimizer.statistics = lambda _: (1., 1.)
    optimizer.run = lambda *_: None
    optimizer.render = lambda *_: None
    trainer.optimizer = optimizer
    state, candidate = optimizer.state, optimizer.candidate
    def reject(*_):
        raise FloatingPointError("nonfinite merged LoRA head")
    trainer.render_merged = reject
    with pytest.raises(FloatingPointError, match="merged LoRA"):
        optimizer.step(trainer, 1e-4)
    assert optimizer.state is state and optimizer.candidate is candidate
    assert (optimizer.steps, optimizer.index, trainer.version, trainer.published_version) == (0, 0, 0, 0)


def test_learner_reads_full_merged_weights_and_current_adapters_from_one_version():
    trainer = TP2LoRAOnline(None, device=0, last_layers=1)
    trainer.optimizer = NS(index=1, execution=[("old", "old-special"), ("new", "new-special")],
                            tensors=lambda: {"W_lora_a": "new-A", "W": "wrong-adapter-view"})
    trainer.merged_execution = ["old-W", "new-W"]
    calls = []
    trainer.full_layout = NS(views=lambda *args: calls.append(args) or {"W": "new-full-W"})
    assert trainer.training_tensors() == {"W_lora_a": "new-A", "W": "new-full-W"}
    assert calls == [("new-W", "new-special")]


@pytest.mark.parametrize("layers", [1, 4])
def test_preflight_budgets_immutable_base_two_full_views_and_all_statistics(layers):
    trainer = TP2LoRAOnline(None, device=0, last_layers=layers)
    graph = trainer.build_training_graph()
    trainer.merged_counts = [123, 2, 1]
    assert trainer.training_view_bytes() == trainer.full_layout.elements * 6 + 126 * 64
    assert trainer.publication_layout().elements > 10 * graph.trainable_parameter_count
