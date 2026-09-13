import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.arllm.runtime import model_metadata, validate_model_artifacts
from experiments.shared.artifacts import checkpoint_weight_hashes
from inference_scaling.arllm.backends import loader
from inference_scaling.shared.model_loading import model_loading_options


def test_shared_cli_retains_all_options_for_existing_entry_points():
    import argparse
    from experiments.shared.model_cli import add_model_output_arguments
    parser = argparse.ArgumentParser()
    add_model_output_arguments(parser)
    args = parser.parse_args(["--proposal-model", "org/proposal", "--mh-iterations", "7",
                              "--thinking-mode", "enabled"])
    assert args.proposal_model == "org/proposal" and args.mh_iterations == 7
    assert args.thinking_mode == "enabled"


def test_role_loading_overrides_are_merged_and_validated():
    config = {"model_loading": {"revision": "main", "local_files_only": True,
                               "tokenizer_kwargs": {"use_fast": True},
                               "proposal": {"revision": "fixed", "tokenizer_kwargs": {"legacy": False}}}}
    values = model_loading_options(config, "proposal")
    assert values["revision"] == "fixed"
    assert values["tokenizer_kwargs"] == {"use_fast": True, "legacy": False}
    config["model_loading"]["revison"] = "typo"
    with pytest.raises(ValueError, match="revison"):
        model_loading_options(config)


def test_explicit_role_distinguishes_two_revisions_of_one_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained", lambda model, **kw: captured.update(kw))
    config = {"models": {"base": "org/same", "proposal": "org/same"},
              "model_loading": {"base": {"revision": "base-ref"}, "proposal": {"revision": "proposal-ref"}}}
    loader.load_backend_from_config("org/same", config, role="proposal")
    assert captured["revision"] == "proposal-ref"


def test_plain_model_prompt_and_explicit_chat_requirement():
    from inference_scaling.shared.prompting import render_prompt
    messages = [{"role": "user", "content": "problem"}]
    assert render_prompt(SimpleNamespace(chat_template=None), messages, {}) == "problem"
    with pytest.raises(ValueError, match="requires"):
        render_prompt(SimpleNamespace(chat_template=None), messages, {"prompt": {"format": "chat"}})


def test_vllm_revision_is_visible_to_artifact_resolution():
    config = {"runtime": {"backend": "vllm"}, "vllm": {"revision": "base-ref", "proposal": {"revision": "proposal-ref"}}}
    assert model_loading_options(config, "base")["revision"] == "base-ref"
    assert model_loading_options(config, "proposal")["revision"] == "proposal-ref"


def test_generic_transformers_loader_forwards_independent_tokenizer_and_adapter(monkeypatch):
    captured = {}
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained", lambda model, **kw: captured.update(model=model, **kw))
    config = {"models": {"base": "org/model", "rl": "org/adapter", "rl_kind": "peft_adapter"},
              "model_loading": {"revision": "model-commit", "tokenizer_name_or_path": "org/tokenizer",
                                "tokenizer_revision": "tokenizer-commit", "adapter_revision": "adapter-commit",
                                "local_files_only": False, "device_map": "auto", "attn_implementation": "sdpa"},
              "runtime": {"dtype": "auto", "score_chunk_size": 128}}
    loader.load_backend_from_config("org/adapter", config, adapter_base="org/model")
    assert captured["model"] == "org/model"
    assert captured["adapter_name_or_path"] == "org/adapter"
    assert captured["tokenizer_name_or_path"] == "org/tokenizer"
    assert captured["revision"] == "model-commit"
    assert captured["adapter_revision"] == "adapter-commit"
    assert captured["score_chunk_size"] == 128
    assert captured["local_files_only"] is False


def test_vllm_async_uses_public_signature_and_same_revision_for_exact_scoring(monkeypatch):
    calls = []
    config = {"models": {"base": "local-model"}, "runtime": {"backend": "vllm"},
              "model_loading": {"revision": "pinned", "tokenizer_name_or_path": "tokenizer-path"},
              "vllm": {"exact_scoring_backend": "transformers"}}
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained", lambda model, **kw: calls.append(("score", kw)) or SimpleNamespace())

    def load(model, **kw):
        import inspect
        assert set(kw) <= set(inspect.signature(loader.AsyncVLLMBackend.from_pretrained_original).parameters)
        calls.append(("engine", kw))

    monkeypatch.setattr(loader.AsyncVLLMBackend, "from_pretrained_original", loader.AsyncVLLMBackend.from_pretrained, raising=False)
    monkeypatch.setattr(loader.AsyncVLLMBackend, "from_pretrained", load)
    loader.load_backend_from_config("local-model", config)
    assert all(kw["revision"] == "pinned" for _, kw in calls)
    assert "enable_mh_fused_logprobs" not in calls[-1][1]


def test_sharded_checkpoint_manifest_and_pinned_load(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}')
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    for number in (1, 2):
        (model / f"model-{number:05d}-of-00002.safetensors").write_bytes(bytes([number]))
    import experiments.arllm.runtime as runtime
    monkeypatch.setattr(runtime, "HASH_CACHE", tmp_path / "hashes")
    config = {"models": {"base": str(model)}}
    artifacts = validate_model_artifacts(config, ["base"])
    assert len(artifacts["shard_sha256"]["base"]) == 2
    assert Path(config["_resolved_models"]["base"]["model"]) == model
    assert model_metadata(config, "base")["source"] == str(model)
    previous = artifacts["weight_sha256"]["base"]
    (model / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    assert validate_model_artifacts(config, ["base"])["weight_sha256"]["base"] != previous


def test_checkpoint_shards_require_safe_existing_names(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"a":"../outside.safetensors"}}')
    with pytest.raises(ValueError, match="relative"):
        checkpoint_weight_hashes(tmp_path, cache_directory=tmp_path / "hashes")
    index.write_text('{"weight_map":{"a":"missing.safetensors"}}')
    with pytest.raises(FileNotFoundError):
        checkpoint_weight_hashes(tmp_path, cache_directory=tmp_path / "hashes")


def test_binary_adapter_and_full_rl_checkpoints(tmp_path, monkeypatch):
    import experiments.arllm.runtime as runtime
    monkeypatch.setattr(runtime, "HASH_CACHE", tmp_path / "hashes")
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "pytorch_model.bin").write_bytes(b"tiny fixture")
    (adapter / "adapter_config.json").write_text('{}')
    (adapter / "adapter_model.bin").write_bytes(b"adapter fixture")
    config = {"models": {"base": str(base), "rl": str(adapter), "rl_kind": "peft_adapter"}}
    manifest = validate_model_artifacts(config, ["rl"])
    assert "rl_adapter" in manifest["weight_sha256"]
    config["models"].update(rl=str(base), rl_kind="full_model")
    assert "rl" in validate_model_artifacts(config, ["rl"])["weight_sha256"]


def test_local_sharded_transformer_round_trip_uses_independent_tokenizer(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast
    from inference_scaling.arllm.backends.transformers_backend import TransformersBackend

    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=5, n_layer=1, n_head=1, n_embd=8, n_positions=32))
    model.save_pretrained(tmp_path / "weights", max_shard_size="1KB")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "a": 3, "b": 4}, unk_token="[UNK]")), bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    with pytest.warns(RuntimeWarning, match="precision"):
        backend = TransformersBackend.from_pretrained(str(tmp_path / "weights"),
            tokenizer_name_or_path=str(tmp_path / "tokenizer"), device="cpu", dtype="auto",
            local_files_only=True, score_chunk_size=4, device_map="cpu")
    assert next(backend.model.parameters()).device == torch.device("cpu")
    assert backend.tokenizer.get_vocab() == tokenizer.get_vocab()
    assert backend.tokenizer.pad_token_id == tokenizer.eos_token_id
    assert "tokenizer=" in backend.model_id
    backend.close()
    assert backend.model is None
    del model
