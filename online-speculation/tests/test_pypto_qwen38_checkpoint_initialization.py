"""离线权重初始化保留 master，完整续训则保留 Adam 状态和步数。"""

import ctypes
import json
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from blockspec.pypto_qwen38 import online_checkpoint as single
from blockspec.pypto_qwen38 import online_suffix_checkpoint as suffix
from blockspec.pypto_qwen38.online import OnlineConfig


def handle(shape):
    value = torch.full(shape, 99., dtype=torch.float32)
    return NS(shape=tuple(shape), tensor=value, data_ptr=value.data_ptr())


class Worker:
    def copy_to(self, destination, source, count):
        ctypes.memmove(destination, source, count)


class Publisher:
    version = 0

    def publish(self, layer, values, *, expected_version):
        assert layer == 63 and expected_version == self.version == 0
        self.version += 1

    def publish_flat(self, layout, values, *, expected_version):
        assert expected_version == self.version == 0
        self.version += 1


@pytest.mark.parametrize("reset", (False, True))
@pytest.mark.parametrize("last_layers", (1, 4))
def test_checkpoint_initialization_and_resume_have_distinct_optimizer_semantics(tmp_path, monkeypatch, reset, last_layers):
    identity = dict(base="exact-model", head="same-source")
    layout = NS(shape=(64,), schema=lambda: dict(parameters=[dict(name="weight", shape=[64])]))
    names = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
    shapes = {name: (2, 2) for name in names}
    if last_layers == 1:
        values = {name+"."+label: torch.full(shape, float(i+1))
                  for name, shape in shapes.items() for i, label in enumerate(single.STATES)}
        state = {name: [handle(shape) for _ in single.STATES] for name, shape in shapes.items()}
        execution = {name: [object(), object()] for name in names}
    else:
        values = {label: torch.full(layout.shape, float(i+1)) for i, label in enumerate(single.STATES)}
        state = [handle(layout.shape) for _ in single.STATES]
        execution = [(object(), object()), (object(), object())]
    path = tmp_path/"optimizer.safetensors"
    save_file(values, str(path))
    manifest = dict(format=single.FORMAT if last_layers == 1 else suffix.FORMAT, head=identity, step=37,
                    beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0., payload_sha256=single.sha256(path))
    if last_layers != 1:
        manifest["layout"] = suffix.contract(layout)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    optimizer = NS(steps=0, index=1, worker=Worker(), state=state, execution=execution, layout=layout,
                   programs=NS(shapes=shapes, loader=object(), zero="zero"),
                   statistics=lambda *args: None, render=lambda *args: None, zero_grad=lambda: None,
                   run=lambda program, value: value.tensor.zero_())

    class Loader:
        def __init__(self, *args): pass
        def zero(self, value): value.tensor.zero_()
        def close(self): pass

    import blockspec.pypto_qwen38.loading as loading
    monkeypatch.setattr(loading, "DeviceLoader", Loader)
    publisher = Publisher()
    module = single if last_layers == 1 else suffix
    module.restore(optimizer, publisher, tmp_path, identity, reset_optimizer=reset)
    assert optimizer.steps == publisher.version == (0 if reset else 37)
    assert optimizer.index == 0
    rows = list(state.values()) if last_layers == 1 else [state]
    for row in rows:
        assert torch.equal(row[0].tensor, torch.ones_like(row[0].tensor))
        for i, value in enumerate(row[1:], 2):
            assert torch.equal(value.tensor, torch.full_like(value.tensor, 0. if reset else float(i)))


def test_optimizer_reset_requires_a_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        OnlineConfig(reset_optimizer=True)
    assert OnlineConfig(resume_dir="checkpoint", reset_optimizer=True).reset_optimizer


def test_existing_positional_online_config_keeps_its_device_and_objective():
    config = OnlineConfig(1, 16, 2, 1e-5, 1., None, None, None, 1, "tv")
    assert config.learner_device == 1 and config.objective == "tv" and not config.reset_optimizer
