from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_scaling.arllm.backends import loader


def _config(backend: str = "transformers"):
    return {
        "run": {"seed": 17},
        "models": {"base": "base-model", "proposal": "proposal-model"},
        "runtime": {
            "backend": backend,
            "device": "cuda:0",
            "dtype": "float32",
            "max_score_batch_size": 11,
        },
    }


def test_transformers_loader_preserves_existing_defaults(monkeypatch) -> None:
    captured = {}

    def fake(model, **kwargs):
        captured.update(model=model, **kwargs)
        return "transformers-backend"

    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained", fake)
    result = loader.load_backend_from_config("base-model", _config())

    assert result == "transformers-backend"
    assert captured == {
        "model": "base-model",
        "adapter_name_or_path": None,
        "device": "cuda:0",
        "dtype": "float32",
        "local_files_only": True,
        "trust_remote_code": False,
        "max_score_batch_size": 11,
        "score_chunk_size": 256,
    }


def test_loader_builds_one_active_batch_schedule_for_both_backends(monkeypatch) -> None:
    config = _config("transformers")
    config["acceleration"] = {
        "speculation": {
            "enabled": True,
            "tiers": [[2, 8], [16, 3], [64, 0]],
            "min_context_tokens": 1,
            "dynamic_vllm": False,
            "stochastic_tree": True,
        }
    }
    transformer_calls = []
    monkeypatch.setattr(
        loader.TransformersBackend,
        "from_pretrained",
        lambda model, **kwargs: transformer_calls.append((model, kwargs)) or "transformers",
    )
    assert loader.load_backend_from_config("base-model", config) == "transformers"
    schedule = transformer_calls[0][1]["speculation"]
    assert schedule.draft_tokens(2) == 8
    assert schedule.draft_tokens(10) == 3
    assert schedule.stochastic_tree is True

    config["runtime"]["backend"] = "vllm"
    vllm_calls = []
    monkeypatch.setattr(
        loader.AsyncVLLMBackend,
        "from_pretrained",
        lambda model, **kwargs: vllm_calls.append((model, kwargs)) or "vllm",
    )
    assert loader.load_backend_from_config("base-model", config) == "vllm"
    assert vllm_calls[0][1]["speculation"] == schedule
    assert vllm_calls[0][1]["dynamic_speculation"] is False


def test_vllm_dynamic_speculation_is_opt_in() -> None:
    config = _config("vllm")
    config["acceleration"] = {"speculation": {"enabled": True}}

    schedule, dynamic = loader._speculation_from_config(config)

    assert schedule is not None
    assert dynamic is False


def test_async_vllm_loader_merges_role_settings_and_exact_scorer(monkeypatch) -> None:
    config = _config("vllm")
    config["vllm"] = {
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.7,
        "max_num_seqs": 32,
        "engine_kwargs": {"enable_chunked_prefill": True},
        "proposal": {
            "gpu_memory_utilization": 0.2,
            "max_num_seqs": 8,
            "exact_scoring_backend": "transformers",
            "exact_scoring_device": "cpu",
            "engine_kwargs": {"cpu_offload_gb": 1},
        },
    }
    exact = SimpleNamespace(model_id="proposal-model")
    transformer_calls = []
    vllm_calls = []

    def fake_transformers(model, **kwargs):
        transformer_calls.append((model, kwargs))
        return exact

    def fake_vllm(model, **kwargs):
        vllm_calls.append((model, kwargs))
        return "async-vllm"

    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained", fake_transformers)
    monkeypatch.setattr(loader.AsyncVLLMBackend, "from_pretrained", fake_vllm)

    result = loader.load_backend_from_config("proposal-model", config)

    assert result == "async-vllm"
    assert transformer_calls[0][1]["device"] == "cpu"
    assert transformer_calls[0][1]["dtype"] == "float32"
    assert vllm_calls[0][1]["scoring_backend"] is exact
    assert vllm_calls[0][1]["gpu_memory_utilization"] == 0.2
    assert vllm_calls[0][1]["max_num_seqs"] == 8
    assert vllm_calls[0][1]["dtype"] == "bfloat16"
    assert vllm_calls[0][1]["seed"] == 17
    assert vllm_calls[0][1]["engine_kwargs"] == {
        "enable_chunked_prefill": True,
        "cpu_offload_gb": 1,
        "max_logprobs": 20,
    }


def test_vllm_sync_override_and_unknown_setting(monkeypatch) -> None:
    config = _config("vllm-sync")
    config["beam"] = {"num_beams": 16}
    config["vllm"] = {"engine_kwargs": {"max_logprobs": 4}}
    calls = []
    monkeypatch.setattr(
        loader.VLLMBackend,
        "from_pretrained",
        lambda model, **kwargs: calls.append((model, kwargs)) or "sync-vllm",
    )
    assert loader.load_backend_from_config("base-model", config) == "sync-vllm"
    assert calls[0][1]["enable_prefix_caching"] is True
    assert calls[0][1]["engine_kwargs"]["max_logprobs"] == 32

    config["vllm"] = {"gpu_memroy_utilization": 0.5}
    with pytest.raises(ValueError, match="gpu_memroy_utilization"):
        loader.load_backend_from_config("base-model", config)

    config["vllm"] = {"engine_kwargs": {"dtype": "float16"}}
    with pytest.raises(ValueError, match="duplicate explicit settings: dtype"):
        loader.load_backend_from_config("base-model", config)


def test_vllm_mh_fused_logprobs_require_sync_without_speculation(monkeypatch) -> None:
    config = _config("vllm-sync")
    config["vllm"] = {"proposal": {"mh_fused_logprobs": True}}
    calls = []
    monkeypatch.setattr(
        loader.VLLMBackend,
        "from_pretrained",
        lambda model, **kwargs: calls.append((model, kwargs)) or "sync-vllm",
    )

    assert loader.load_backend_from_config("proposal-model", config) == "sync-vllm"
    assert calls[0][1]["enable_mh_fused_logprobs"] is True

    config["runtime"]["backend"] = "vllm"
    with pytest.raises(ValueError, match="requires runtime.backend='vllm-sync'"):
        loader.load_backend_from_config("proposal-model", config)

    config["runtime"]["backend"] = "vllm-sync"
    config["acceleration"] = {"speculation": {"enabled": True}}
    with pytest.raises(ValueError, match="cannot be combined"):
        loader.load_backend_from_config("proposal-model", config)


def test_backend_override_is_fingerprinted_in_config() -> None:
    config = _config()
    loader.set_backend_override(config, "vllm")
    assert config["runtime"]["backend"] == "vllm"
    with pytest.raises(ValueError, match="unknown runtime backend"):
        loader.set_backend_override(config, "unknown")


def test_close_backend_closes_outer_and_exact_backend() -> None:
    calls = []
    exact = SimpleNamespace(close=lambda: calls.append("exact"))
    outer = SimpleNamespace(
        close=lambda: calls.append("outer"),
        scoring_backend=exact,
    )
    loader.close_backend(outer)
    assert calls == ["outer", "exact"]


def test_vllm_loader_closes_exact_scorer_when_engine_load_fails(monkeypatch) -> None:
    config = _config("vllm")
    config["vllm"] = {"exact_scoring_backend": "transformers"}
    closed = []
    exact = SimpleNamespace(model_id="base-model", close=lambda: closed.append(True))
    monkeypatch.setattr(
        loader.TransformersBackend,
        "from_pretrained",
        lambda *args, **kwargs: exact,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("engine allocation failed")

    monkeypatch.setattr(loader.AsyncVLLMBackend, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="engine allocation failed"):
        loader.load_backend_from_config("base-model", config)
    assert closed == [True]
