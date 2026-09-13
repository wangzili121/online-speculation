from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from inference_scaling.dllm.backends.loader import load_llada_backend


def _config(tmp_path: Path):
    return {
        "model": {"path": str(tmp_path / "base")},
        "proposal": {"kind": "shared_prefix_layers", "layers": 8},
        "alignment": {"adapter": str(tmp_path / "adapter")},
        "runtime": {"device": "cpu", "dtype": "float32", "max_batch_size": 3},
    }


def test_base_and_proposal_roles_share_the_loaded_model(monkeypatch, tmp_path):
    calls = []

    class FakeBackend:
        def with_prefix_layers(self, layers):
            calls.append(("prefix", layers))
            return "proposal"

    base = FakeBackend()

    def fake_loader(cls, path, **kwargs):
        calls.append((path, kwargs))
        return base

    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.LLaDATransformersBackend.from_pretrained",
        classmethod(fake_loader),
    )
    config = _config(tmp_path)

    loaded = load_llada_backend(config, "base")
    proposal = load_llada_backend(config, "proposal", base_backend=loaded)

    assert calls[0][1] == {
        "device": "cpu",
        "dtype": "float32",
        "mask_token_id": 156895,
        "max_batch_size": 3,
    }
    assert calls[1] == ("prefix", 8)
    assert proposal == "proposal"


def test_aligned_role_requires_a_completed_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the VRPO stage first"):
        load_llada_backend(_config(tmp_path), "aligned")


def test_aligned_role_retains_the_runtime_batch_cap(monkeypatch, tmp_path):
    config = _config(tmp_path)
    Path(config["alignment"]["adapter"]).mkdir(parents=True)
    constructed = []
    base = SimpleNamespace(model=object(), tokenizer=object(), mask_token_id=17)

    class FakeBackend:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return base

        def __init__(self, model, tokenizer, **kwargs):
            constructed.append((model, tokenizer, kwargs))

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter):
            return SimpleNamespace(eval=lambda: "aligned-model")

    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.LLaDATransformersBackend",
        FakeBackend,
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))

    load_llada_backend(config, "aligned")

    assert constructed[0][2]["max_batch_size"] == 3
    assert constructed[0][2]["mask_token_id"] == 17
