import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blockspec.vllm_ascend.dsv4_custom_proposer import (
    OrthrusDSV4CustomProposer,
    build_orthrus_runtime_config,
)


def _head(path: Path) -> Path:
    path.mkdir()
    (path / "orthrus_head_manifest.json").write_text(
        json.dumps(
            {
                "format": "orthrus-dsv4-head-v2",
                "format_version": 2,
                "history_backend": "shared_target_hybrid",
                "block_size": 32,
                "num_speculative_tokens": 31,
            }
        ),
        encoding="ascii",
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OrthrusDSV4DraftModel"],
                "orthrus_head_format": "orthrus-dsv4-head-v2",
            }
        ),
        encoding="ascii",
    )
    return path


def _config(head: Path):
    spec = SimpleNamespace(
        method="custom_class",
        model=("blockspec.vllm_ascend.dsv4_custom_proposer.OrthrusDSV4CustomProposer"),
        num_speculative_tokens=31,
        draft_sample_method="probabilistic",
        rejection_sample_method="standard",
        disable_padded_drafter_batch=False,
    )
    target = SimpleNamespace(
        tokenizer="target-tokenizer",
        tokenizer_mode="deepseek_v4",
        dtype="bfloat16",
        seed=7,
        max_model_len=131072,
        quantization="ascend",
    )
    return SimpleNamespace(
        speculative_config=spec,
        model_config=target,
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        load_config=object(),
        additional_config={
            "custom_proposer_config": {"orthrus": {"head_model": str(head)}}
        },
    )


def _install_fake_vllm(monkeypatch):
    class ModelConfig:
        def __init__(
            self,
            model,
            runner="auto",
            tokenizer=None,
            tokenizer_mode="auto",
            dtype="auto",
            seed=0,
            max_model_len=None,
            spec_target_max_model_len=None,
            quantization=None,
            enforce_eager=False,
            **kwargs,
        ):
            del kwargs
            config = json.loads((Path(model) / "config.json").read_text())
            self.model = model
            self.runner_type = runner
            self.tokenizer = tokenizer
            self.tokenizer_mode = tokenizer_mode
            self.dtype = dtype
            self.seed = seed
            self.max_model_len = max_model_len
            self.spec_target_max_model_len = spec_target_max_model_len
            self.quantization = quantization
            self.enforce_eager = enforce_eager
            self.architectures = config["architectures"]
            self.hf_config = SimpleNamespace(**config)

    vllm = ModuleType("vllm")
    config = ModuleType("vllm.config")
    config.ModelConfig = ModelConfig
    vllm.config = config
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config)


def test_build_runtime_config_keeps_public_custom_class(monkeypatch, tmp_path):
    _install_fake_vllm(monkeypatch)
    public = _config(_head(tmp_path / "head"))

    internal = build_orthrus_runtime_config(public)

    assert public.speculative_config.method == "custom_class"
    assert internal.speculative_config.method == "dflash"
    assert internal.speculative_config.parallel_drafting is True
    assert internal.speculative_config.enforce_eager is True
    assert internal.speculative_config.draft_tensor_parallel_size == 8
    assert internal.speculative_config.draft_model_config.architectures == [
        "OrthrusDSV4DraftModel"
    ]


def test_custom_proposer_requires_versioned_head(tmp_path):
    head = tmp_path / "head"
    head.mkdir()
    (head / "orthrus_head_manifest.json").write_text("{}", encoding="ascii")
    config = _config(head)

    with pytest.raises(ValueError, match="unsupported Orthrus head format"):
        OrthrusDSV4CustomProposer(config)


def test_custom_proposer_rejects_unknown_options(tmp_path):
    config = _config(_head(tmp_path / "head"))
    config.additional_config["custom_proposer_config"]["orthrus"]["typo"] = True

    with pytest.raises(ValueError, match="unsupported Orthrus proposer options"):
        OrthrusDSV4CustomProposer(config)


def test_custom_proposer_binds_and_delegates(monkeypatch, tmp_path):
    _install_fake_vllm(monkeypatch)
    config = _config(_head(tmp_path / "head"))
    calls = []

    class Backend:
        dynamic_spec = None

        def __init__(self, runtime_config, device, runner):
            calls.append(("init", runtime_config, device, runner))

        def load_model(self, target):
            calls.append(("load", target))

        def take_last_draft_probs(self):
            return "probabilities"

    module = ModuleType("blockspec.vllm_ascend.dsv4_proposer")
    module.AscendOrthrusProposer = Backend
    monkeypatch.setitem(sys.modules, "blockspec.vllm_ascend.dsv4_proposer", module)

    proposer = OrthrusDSV4CustomProposer(config)
    runtime = SimpleNamespace(
        api_version=1,
        vllm_config=config,
        device="npu:0",
        runner="runner",
    )
    proposer.bind_runtime(runtime)
    proposer.load_model("target")

    assert calls[0][0] == "init"
    assert calls[0][2:] == ("npu:0", "runner")
    assert calls[1] == ("load", "target")
    assert proposer.take_last_draft_probs() == "probabilities"
