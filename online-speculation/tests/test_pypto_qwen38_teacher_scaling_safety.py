"""设备教师缩放后仍须在 Adam/发布前拒绝非有限训练 loss。"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_never_advances_adam_or_publishes_a_head(invalid):
    learner = TP2Online(None, device=0, last_layers=1, teacher_temperature=.0625,
                        teacher_scaling="device")
    calls = []
    serving = object()
    learner.runner = SimpleNamespace(suffix=serving)
    learner.shared, learner.pack_stats = {}, {}
    learner.config = object()
    learner.graph = SimpleNamespace(losses=SimpleNamespace(name="losses"))
    learner.optimizer = SimpleNamespace(
        zero_grad=lambda: calls.append("zero_candidate_gradient"), tensors=lambda: {}, gradients={},
        accumulate=lambda: calls.append("accumulate"),
        step=lambda *_: pytest.fail("Adam/publication was reached after a nonfinite loss"))
    learner.workspace = SimpleNamespace(run=lambda *_a, **_k: calls.append("replay"),
                                        values={"losses": object()})
    learner.loss_program, learner.loss = object(), SimpleNamespace(data_ptr=2048)
    learner.loss_host = torch.full((1, 8), invalid)
    learner.worker = SimpleNamespace(run=lambda *_a, **_k: None, copy_from=lambda *_a, **_k: None)
    with pytest.raises(FloatingPointError, match="nonfinite TP2 online loss"):
        learner.train([dict(valid=1, inputs={}, position=0, width=32)], round_number=1)
    assert calls == ["zero_candidate_gradient", "replay", "accumulate"]
    assert learner.runner.suffix is serving and learner.version == learner.published_version == 0
