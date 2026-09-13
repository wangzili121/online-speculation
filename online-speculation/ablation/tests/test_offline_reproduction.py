"""Reference artifacts, two-arm accounting and fused low-rank inference."""

import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from blockspec_ablation.adapter_io import load_peft_adapter
from blockspec_ablation.benchmark import BenchmarkConfig, benchmark_offline
from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.model import Decoder, GatedLinear, ModelConfig
from blockspec_ablation.execution import FixedShapeExecutor


def tiny():
    torch.manual_seed(238)
    return Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                               num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=8, adapter_rank=2, adapter_alpha=32))


def artifact(path, model, *, bad_key=False, prefix="base_model.model.", duplicate=False):
    from safetensors.torch import save_file
    values = {prefix + name + ".weight": torch.randn_like(value)
              for name, value in adapter_state(model).items()}
    if duplicate:
        name = next(iter(values))
        values[name.removeprefix("base_model.model.")] = values[name].clone()
    if bad_key:
        values["unexpected.weight"] = torch.ones(1)
    file = path / "adapter_model.safetensors"
    save_file(values, str(file))
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": model.config.adapter_rank, "lora_alpha": model.config.adapter_alpha,
        "target_modules": model.config.adapter_targets, "lora_dropout": .05,
    }))
    return hashlib.sha256(file.read_bytes()).hexdigest(), values


@pytest.mark.parametrize("prefix", ["", "base_model.model."])
def test_reference_import_maps_every_tensor_and_preserves_base(tmp_path, prefix):
    model = tiny()
    frozen = base_fingerprint(model)
    digest, values = artifact(tmp_path, model, prefix=prefix)
    result = load_peft_adapter(tmp_path, model, expected_sha256=digest)
    assert result["tensors"] == 28 and result["kind"] == "published_peft_reference"
    assert base_fingerprint(model) == frozen
    for name, value in adapter_state(model).items():
        torch.testing.assert_close(value, values[prefix + name + ".weight"], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["hash", "key", "scale", "duplicate"])
def test_reference_import_failure_is_atomic(tmp_path, failure):
    model = tiny()
    initial = adapter_state(model)
    digest, _ = artifact(tmp_path, model, bad_key=failure == "key", duplicate=failure == "duplicate")
    if failure == "scale":
        config_file = tmp_path / "adapter_config.json"
        config = json.loads(config_file.read_text())
        config["lora_alpha"] = 16
        config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_peft_adapter(tmp_path, model, expected_sha256="0" * 64 if failure == "hash" else digest)
    torch.testing.assert_close(adapter_state(model), initial, atol=0, rtol=0)


@pytest.mark.parametrize("sampler", ["linear", "tree"])
def test_offline_benchmark_is_paired_and_keeps_weights_and_execution(sampler):
    model = tiny()
    original = adapter_state(model)
    events = []
    result = benchmark_offline(model, [torch.tensor([[0, 2, 4]]), torch.tensor([[1, 3, 2]])],
                               BenchmarkConfig(tokens=12, warmup_tokens=4, sampler=sampler,
                                               top_k=2, prefix_budget=5, attention_backend="grouped"),
                               progress=events.append)
    assert [e["arm"] for e in events] == ["ar", "static", "static", "ar", "static", "ar", "ar", "static"]
    assert result["greedy_identical"] and result["base_unchanged"] and result["adapter_unchanged"]
    assert result["arms"]["static"]["tokens"] == result["arms"]["ar"]["tokens"] == 48
    assert result["arms"]["static"]["updates"] == 0
    assert result["speedup"] == pytest.approx(result["arms"]["ar"]["seconds"] / result["arms"]["static"]["seconds"])
    assert model.attention_signature() == ("sdpa", "sdpa")
    torch.testing.assert_close(adapter_state(model), original, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1, 16, 64])
def test_rank_space_gate_and_addmm_match_full_residual(device, dtype, scale):
    torch.manual_seed(459)
    layer = GatedLinear(16, 24, 4, 4 * scale, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.lora_B.normal_(std=.1)
        x = torch.randn(2, 5, 16, device=device, dtype=dtype)
        mask = torch.rand(2, 5, device=device) > .5
        base = F.linear(x, layer.weight, layer.bias)
        expected = base + mask[..., None] * (F.linear(F.linear(x, layer.lora_A), layer.lora_B) * scale)
        actual = layer(x, mask)
        tol = 2e-5 if dtype == torch.float32 else 2e-13
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(actual[~mask], base[~mask], atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@torch.no_grad()
def test_forward_signatures_share_input_workspace_and_keep_owned_outputs(device):
    model = tiny().to(device).set_attention_backend("grouped")
    engine = FixedShapeExecutor(model, capacity=32, max_query=4, use_cuda_graph=device == "cuda")
    engine.prepare([(1, False, None), (4, True, None), (2, False, None)])
    assert len({slot.past.data_ptr() for slot in engine.slots.values()}) == 1
    prefix = torch.tensor([[0, 1, 2, 3, 4]], device=device)
    past = engine(prefix, return_cache=True)[1]
    saved = past.packed.clone() if hasattr(past, "packed") else tuple((k.clone(), v.clone()) for k, v in past)
    for ids, mask in (([[1, 2, 3, 4]], [[False, True, True, True]]), ([[2]], None), ([[4, 5]], None)):
        tokens = torch.tensor(ids, device=device)
        active = torch.tensor(mask, device=device) if mask is not None else None
        expected, expected_kv = model(tokens, cache=past, adapter_mask=active, return_cache=True)
        actual, actual_kv = engine(tokens, cache=past, adapter_mask=active, return_cache=True)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(tuple(actual_kv), tuple(expected_kv), atol=2e-5, rtol=2e-5)
    original = past.packed if hasattr(past, "packed") else tuple(past)
    torch.testing.assert_close(original, saved, atol=0, rtol=0)
