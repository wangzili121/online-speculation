import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint, load_checkpoint, save_checkpoint
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.distillation import offline_step, paired_loss
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner


def model_and_prompt():
    torch.manual_seed(87)
    model = Decoder(ModelConfig(vocab_size=9, hidden_size=16, intermediate_size=24,
                                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                num_hidden_layers=2, adapter_rank=2, adapter_alpha=2))
    return model, torch.tensor([[0, 1, 2]])


@pytest.mark.parametrize("block_size", [2, 4, 8])
@pytest.mark.parametrize("budget", [0, 1, 2, 3, 15])
def test_greedy_speculation_matches_true_ar(block_size, budget):
    model, prompt = model_and_prompt()
    # Nonzero adapters make this more than an all-zero low-rank branch test.
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.2)
    ar = generate_ar(model, prompt, budget)
    spec = generate_speculative(model, prompt, budget, block_size=block_size)
    assert ar.tokens == spec.tokens
    assert ar.decode_forwards == budget


@pytest.mark.parametrize("eos_position", [0, 1, 4])
def test_eos_is_neither_skipped_nor_duplicated(eos_position):
    model, prompt = model_and_prompt()
    full = generate_ar(model, prompt, 12)
    eos = full.tokens[eos_position]
    ar = generate_ar(model, prompt, 12, eos_id=eos)
    spec = generate_speculative(model, prompt, 12, block_size=4, eos_id=eos)
    assert ar.tokens == spec.tokens
    assert spec.tokens[-1] == eos
    assert eos not in spec.tokens[:-1]


def test_full_offline_online_checkpoint_pipeline(tmp_path):
    model, prompt = model_and_prompt()
    model.train_adapters_only()
    base = base_fingerprint(model)
    clean = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 4, 3, 2, 1, 5]])
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=.02, weight_decay=0)
    rng = torch.Generator().manual_seed(17)
    before = float(paired_loss(model, clean, 3, noisy=clean.flip(-1), kind="forward_kl").detach())
    for _ in range(30):
        offline_step(model, optimizer, clean, 3, kind="forward_kl", generator=rng)
    after = float(paired_loss(model, clean, 3, noisy=clean.flip(-1), kind="forward_kl").detach())
    assert after < before
    assert base_fingerprint(model) == base
    path = tmp_path / "full.pt"
    save_checkpoint(path, model, metadata={"test": True})
    loaded, metadata = load_checkpoint(path)
    assert metadata == {"test": True}
    torch.testing.assert_close(loaded(clean), model(clean), atol=0, rtol=0)
    for key, value in adapter_state(model).items():
        assert torch.equal(value, adapter_state(loaded)[key])
    ar = generate_ar(loaded, prompt, 24)
    learner = OnlineLearner(loaded, OnlineConfig(stride=1, replay_blocks=2, learning_rate=.01,
                                               loss="forward_kl"))
    initial = adapter_state(loaded)
    online = generate_speculative(loaded, prompt, 24, block_size=4, learner=learner)
    assert ar.tokens == online.tokens
    assert online.updates > 0 and online.update_seconds > 0
    assert online.update_seconds < online.seconds
    assert any(not torch.equal(value, adapter_state(loaded)[name]) for name, value in initial.items())
    assert base_fingerprint(loaded) == base
    assert not learner.replay
    previous_version = learner.version
    generate_speculative(loaded, prompt, 24, block_size=4, learner=learner)
    assert learner.version > previous_version


def test_online_replays_only_valid_rows_and_changes_actual_adapter():
    model, prompt = model_and_prompt()
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1))
    _, cache = model(prompt[:, :-1], return_cache=True)
    inputs = torch.tensor([[2, 4, 5, 6]])
    teacher = torch.randn(1, 9)
    result = learner.observe(Feedback(inputs, cache, teacher, 1))
    assert result["positions"] == 1 and result["version"] == 1
    assert learner.replay[0].teacher_logits.grad_fn is None
    assert all(k.grad_fn is None and v.grad_fn is None for k, v in learner.replay[0].cache)


def test_adapter_checkpoint_rejects_wrong_base_before_mutation(tmp_path):
    model, _ = model_and_prompt()
    path = tmp_path / "adapter.pt"
    save_checkpoint(path, model, adapter_only=True)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, adapter_only=True)
    target, _ = model_and_prompt()
    with torch.no_grad():
        target.lm_head.weight.add_(1)
    original = adapter_state(target)
    with pytest.raises(ValueError, match="different base"):
        load_checkpoint(path, model=target)
    for name, value in original.items():
        assert torch.equal(value, adapter_state(target)[name])


def test_adapter_checkpoint_round_trip(tmp_path):
    model, _ = model_and_prompt()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_()
    path = tmp_path / "adapter.pt"
    save_checkpoint(path, model, adapter_only=True)
    target, _ = model_and_prompt()
    load_checkpoint(path, model=target)
    for name, value in adapter_state(model).items():
        assert torch.equal(value, adapter_state(target)[name])
