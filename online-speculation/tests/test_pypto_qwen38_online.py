"""在线采集与参数发布契约；设备计算由独立 NPU 对照入口验证。"""

from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.online import FeedbackBlock, OnlineConfig, OnlineLearner
from blockspec.pypto_qwen38.online_checkpoint import FORMAT, read, sha256
from blockspec.pypto_qwen38.weights import DraftWeights


def test_head_publication_is_complete_and_versioned():
    banks = tuple(torch.empty((1, 64)) for _ in range(6))
    head = SimpleNamespace(banks=lambda layer: banks)
    weights = DraftWeights(head)
    weights.worker = object()
    with pytest.raises(ValueError):
        weights.publish(63, banks[:5], expected_version=0)
    assert weights.version == 0 and not weights.overrides
    weights.publish(63, banks, expected_version=0)
    with pytest.raises(RuntimeError):
        weights.publish(63, banks, expected_version=0)
    assert weights.version == 1 and weights.overrides[63] == banks
    weights.close()
    assert weights.version == 0 and not weights.overrides


def test_teacher_capture_excludes_rollback_replay():
    calls = []
    owner = SimpleNamespace(teacher_program="teacher", run=lambda *args: calls.append(args))
    block = FeedbackBlock(owner, {"teacher": "owned_logits"})
    block.width = 32
    source = SimpleNamespace(logits="target_logits")
    block.teacher(source, 3, 16)
    block.teacher(source, 3, 16)
    block.teacher(source, 3, 1)
    assert block.rows == 32 and len(calls) == 2
    assert calls[0][-3:] == (48, 0, 16) and calls[1][-3:] == (48, 16, 16)


def test_feedback_supervises_only_reached_rows():
    learner = OnlineLearner(None, OnlineConfig(stride=16, replay_blocks=2))
    block = FeedbackBlock(learner, {})
    block.width = block.rows = 32
    learner.current = block
    learner.observe(SimpleNamespace(accepted=0, tokens=(3,)))
    assert block.valid == 1 and learner.rounds == 1
    learner.abort()
    assert not learner.replay and learner.current is None
    with pytest.raises(ValueError):
        OnlineConfig(last_layers=4)


def test_checkpoint_binds_head_and_tensor_schema(tmp_path):
    import json
    from safetensors.torch import save_file
    tensors = {"q_gamma."+name: torch.zeros((1, 64)) for name in ("master", "first", "second")}
    path = tmp_path/"optimizer.safetensors"
    save_file(tensors, str(path))
    manifest = dict(format=FORMAT, head={"hash": "original_head"}, step=3,
        beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0., payload_sha256=sha256(path))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    step, restored = read(tmp_path, manifest["head"], {"q_gamma": (1, 64)})
    assert step == 3 and set(restored) == set(tensors)
    with pytest.raises(ValueError, match="incompatible"):
        read(tmp_path, {"hash": "different_head"}, {"q_gamma": (1, 64)})
    with pytest.raises(ValueError, match="schema"):
        read(tmp_path, manifest["head"], {"q_gamma": (1, 128)})
    with path.open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        read(tmp_path, manifest["head"], {"q_gamma": (1, 64)})
