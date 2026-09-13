import torch

from blockspec_ablation.checkpoint import adapter_state, load_checkpoint, save_checkpoint
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.validation import FixedValidation


def tiny():
    torch.manual_seed(97)
    return Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, adapter_rank=2))


def test_fixed_validation_is_repeatable_and_read_only():
    model = tiny().train_adapters_only()
    validation = FixedValidation([torch.tensor([0, 1, 2, 3, 4, 5, 6])],
                                  vocab_size=7, blocks=(2, 4), batches=2, length=6)
    before = adapter_state(model)
    first = validation.evaluate(model)
    second = validation.evaluate(model)
    assert first == second
    for row in first["blocks"].values():
        assert row["positions"] == 10
        assert 0 <= row["teacher_forced_overlap"] <= 1
        assert 0 <= row["argmax_agreement"] <= 1
    assert all(torch.equal(before[k], v) for k, v in adapter_state(model).items())
    assert all(p.grad is None for p in model.parameters())


def test_fp32_master_adapters_survive_bf16_base_checkpoint_loading(tmp_path):
    model = tiny().bfloat16().train_adapters_only()
    with torch.no_grad():
        for parameter in model.adapter_parameters():
            parameter.uniform_(.001231, .002733)
    path = tmp_path / "master.pt"
    save_checkpoint(path, model, adapter_only=True)
    loaded = tiny().bfloat16()
    load_checkpoint(path, model=loaded)
    assert all(p.dtype == torch.float32 for p in loaded.adapter_parameters())
    assert all(torch.equal(v, adapter_state(loaded)[k]) for k, v in adapter_state(model).items())
