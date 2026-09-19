"""Tiny DSV4 checks for hybrid-head gradients, resume, and artifact exchange."""

import copy
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DeepseekV4ForCausalLM"):
    pytest.skip("Transformers has no DeepSeek-V4 model", allow_module_level=True)

from blockspec.dsv4_fitting import DSV4FitConfig, DSV4Trainer
from blockspec.dsv4_training import (
    OrthrusDSV4TrainingModel,
    export_bf16_head,
    load_bf16_head,
    validate_bf16_training_base,
)
from blockspec.parallel.fitting import TokenDataset


def tiny_target():
    config = transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        mlp_layer_types=["moe", "moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=2,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return transformers.DeepseekV4ForCausalLM(config)


def token_data(tmp_path, length=16):
    path = tmp_path / "tokens.jsonl"
    rows = [
        {"input_ids": [2 + ((offset + index) % 48) for index in range(length + 4)]}
        for offset in range(6)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return TokenDataset(path, 64, length)


def fit_config(steps=2):
    return DSV4FitConfig(
        steps=steps,
        batch_size=1,
        sequence_length=16,
        anchors_per_sequence=2,
        accumulate=1,
        chunk_rows=4,
        learning_rate=1e-3,
        warmup_steps=0,
        precision="fp32",
        seed=71,
    )


def test_hybrid_head_backward_freezes_target():
    torch.manual_seed(7)
    model = OrthrusDSV4TrainingModel(tiny_target(), block_size=4, mask_token_id=63)
    tokens = torch.arange(16).remainder(48).add(2).unsqueeze(0)
    anchors = torch.tensor([[0, 8]])
    before = {name: value.detach().clone() for name, value in model.target.state_dict().items()}
    loss, metrics = model.loss(tokens, anchors, chunk_rows=4)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["indexer_loss"])
    assert model.compress_ratios == [1, 4, 128]
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())
    assert all(torch.equal(value, model.target.state_dict()[name]) for name, value in before.items())


def test_exact_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(11)
    original = tiny_target()
    data = token_data(tmp_path)
    full = DSV4Trainer(
        OrthrusDSV4TrainingModel(copy.deepcopy(original), block_size=4, mask_token_id=63),
        data,
        fit_config(),
    )
    full.run()
    split = DSV4Trainer(
        OrthrusDSV4TrainingModel(copy.deepcopy(original), block_size=4, mask_token_id=63),
        data,
        fit_config(),
    )
    split.run(1)
    checkpoint = tmp_path / "step.pt"
    split.save(checkpoint)
    resumed = DSV4Trainer.resume(
        checkpoint,
        OrthrusDSV4TrainingModel(copy.deepcopy(original), block_size=4, mask_token_id=63),
        data,
    )
    resumed.run()
    assert resumed.step == full.step == 2
    assert all(
        torch.equal(value, resumed.model.head.state_dict()[name])
        for name, value in full.model.head.state_dict().items()
    )


def test_resume_rejects_a_different_bf16_base(tmp_path):
    torch.manual_seed(13)
    data = token_data(tmp_path)
    identity = {"config_sha256": "a", "weight_index_sha256": "b"}
    trainer = DSV4Trainer(
        OrthrusDSV4TrainingModel(tiny_target(), block_size=4, mask_token_id=63),
        data,
        fit_config(steps=1),
        base_identity=identity,
    )
    trainer.run()
    checkpoint = tmp_path / "identity.pt"
    trainer.save(checkpoint)
    with pytest.raises(ValueError, match="BF16 base identity"):
        DSV4Trainer.resume(
            checkpoint,
            OrthrusDSV4TrainingModel(tiny_target(), block_size=4, mask_token_id=63),
            data,
            base_identity={"config_sha256": "a", "weight_index_sha256": "c"},
        )


def test_bf16_exchange_round_trip(tmp_path):
    torch.manual_seed(19)
    model = OrthrusDSV4TrainingModel(tiny_target(), block_size=32, mask_token_id=63)
    serving = tmp_path / "serving"
    training = tmp_path / "training"
    serving.mkdir()
    training.mkdir()
    serving_config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 3,
        "compress_ratios": [0, 4, 128],
    }
    training_config = dict(serving_config)
    for directory, config in ((serving, serving_config), (training, training_config)):
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {}}), encoding="utf-8"
        )
    output = tmp_path / "head"
    expected = {
        name: value.detach().clone() for name, value in model.head.state_dict().items()
    }
    export_bf16_head(
        model,
        output,
        serving_target=serving,
        training_base=training,
        mask_token_id=63,
        base_model_id="test/dsv4",
        training_metadata={"step": 1},
    )
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    load_bf16_head(model, output)
    assert all(
        torch.equal(value.to(torch.bfloat16), model.head.state_dict()[name].to(torch.bfloat16))
        for name, value in expected.items()
    )
    manifest = json.loads((output / "orthrus_head_manifest.json").read_text())
    assert manifest["artifact_role"] == "trained"
    assert manifest["precision"] == "bf16"
    assert manifest["quantization"]["head_calibration_required"] is True


def test_training_base_uses_shard_dtypes_not_stale_config_metadata(tmp_path):
    from safetensors.torch import save_file

    directory = tmp_path / "bf16"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    save_file(
        {"model.layers.0.attn.wq_a.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
        directory / "model-00001-of-00001.safetensors",
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.attn.wq_a.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    result = validate_bf16_training_base(directory)
    assert result["weight_tensors"] == 1
    assert result["weight_dtypes"] == {"BF16": 1}
    assert result["stale_quantization_metadata"] is True


def test_training_base_rejects_quantized_weight_scale_pairs(tmp_path):
    from safetensors.torch import save_file

    directory = tmp_path / "int8"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    tensors = {
        "model.layers.0.attn.wq_a.weight": torch.ones(2, 2, dtype=torch.int8),
        "model.layers.0.attn.wq_a.weight_scale": torch.ones(2, 1),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, directory / shard)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quantized weights"):
        validate_bf16_training_base(directory)
