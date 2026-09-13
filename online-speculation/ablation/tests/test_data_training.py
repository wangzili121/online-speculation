import json
import sys

import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint, load_checkpoint
from blockspec.data import load_sequences, sample_batch
from blockspec_ablation.distillation import divergence
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.training import TrainingConfig, train_adapter


def test_jsonl_records_are_never_packed_across_boundaries(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"input_ids": [1] * 8}) + "\n" +
                    json.dumps({"input_ids": [2] * 8}), encoding="utf-8")
    sequences = load_sequences(path, 3)
    batch = sample_batch(sequences, batch_size=16, length=6, bos_id=0, device="cpu",
                         generator=torch.Generator().manual_seed(2))
    assert torch.equal(batch[:, 0], torch.zeros(16, dtype=torch.long))
    assert all(row[1:].unique().numel() == 1 for row in batch)


@pytest.mark.parametrize("ids", [[0, 7], [0, -1], [0, 1.0], [True, 1], [1]])
def test_bad_dataset_fails(ids, tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"input_ids": ids}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sequences(path, 7)


def test_curriculum_and_loss_schedule_execute_with_frozen_teacher():
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, adapter_rank=2))
    frozen = base_fingerprint(model)
    events = []
    result = train_adapter(model, [torch.tensor([1, 2, 3, 4, 5, 6, 7])],
                           TrainingConfig(steps=4, batch_size=1, sequence_length=6,
                                          blocks=(2, 4), warmup_steps=2), progress=events.append)
    assert [e["block"] for e in events] == [2, 2, 4, 4]
    assert [e["loss_kind"] for e in events] == ["reverse_kl_l1", "reverse_kl_l1", "l1", "l1"]
    assert base_fingerprint(model) == frozen
    assert result["training_tokens"] == 24


def test_joint_warmup_value_and_gradient_are_reverse_kl_plus_unhalved_l1():
    student = torch.tensor([[.4, 1.2, -.8], [-.4, .2, .9]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.8, -.2, .3], [.6, .4, -.7]], dtype=torch.float64, requires_grad=True)
    combined = divergence(student, teacher, "reverse_kl_l1").sum()
    expected = (divergence(student, teacher, "reverse_kl") + divergence(student, teacher, "l1")).sum()
    torch.testing.assert_close(combined, expected, atol=0, rtol=0)
    g = torch.autograd.grad(combined, (student, teacher), allow_unused=True)
    reference = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(g[0], reference, atol=1e-15, rtol=1e-15)
    assert g[1] is None
    assert not torch.isclose(combined, (divergence(student, teacher, "reverse_kl") +
                                        divergence(student, teacher, "tv")).sum())


@pytest.mark.parametrize("alpha", [None, 32.0])
def test_training_cli_preserves_adapter_scale_through_checkpoint(alpha, tmp_path, monkeypatch, capsys):
    from blockspec_ablation import cli

    loaded = []

    def tiny_base(directory, *, rank, alpha=None, device, dtype):
        torch.manual_seed(713)
        model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=8, adapter_rank=rank,
                                     adapter_alpha=float(rank) if alpha is None else alpha)).to(device, dtype)
        loaded.append(model)
        return model

    monkeypatch.setattr(cli, "load_hf_base", tiny_base)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"input_ids": [0, 1, 2, 3, 4, 5, 6, 7]}) + "\n", encoding="utf-8")
    output = tmp_path / "adapter.pt"
    arguments = ["blockspec", "train", "--base", str(tmp_path), "--data", str(data),
                 "--output", str(output), "--device", "cpu", "--dtype", "float32",
                 "--rank", "2", "--steps", "4", "--warmup-steps", "1", "--threads", "1",
                 "--sequence-length", "8", "--blocks", "2,4,6,8"]
    if alpha is not None:
        arguments.extend(["--alpha", str(alpha)])
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["training_tokens"] == 32
    trained = loaded[0]
    payload = torch.load(output, weights_only=True)
    assert trained.config.adapter_alpha == (2.0 if alpha is None else alpha)
    assert payload["config"]["adapter_alpha"] == trained.config.adapter_alpha
    target = tiny_base(tmp_path, rank=2, alpha=trained.config.adapter_alpha, device="cpu", dtype=torch.float32)
    restored, metadata = load_checkpoint(output, model=target)
    assert metadata["training_config"]["blocks"] == (2, 4, 6, 8)
    assert base_fingerprint(trained) == base_fingerprint(restored)
    for name, value in adapter_state(trained).items():
        torch.testing.assert_close(adapter_state(restored)[name], value, atol=0, rtol=0)
    clean = torch.tensor([[0, 1, 2, 3]])
    active = torch.ones_like(clean, dtype=torch.bool)
    torch.testing.assert_close(restored(clean, adapter_mask=active), trained(clean, adapter_mask=active),
                               atol=0, rtol=0)
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"input_ids": [0, 3, 2, 1, 6, 7, 4, 5]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["blockspec", "benchmark", "--base", str(tmp_path),
                                     "--adapter", str(output), "--data", str(validation), "--device", "cpu",
                                     "--prompts", "1", "--prompt-length", "3", "--tokens", "8",
                                     "--warmup-tokens", "4", "--threads", "1", "--update-stride", "1"])
    cli.main()
    benchmark = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert loaded[-1].config.adapter_alpha == trained.config.adapter_alpha
    assert benchmark["greedy_identical"] and benchmark["base_unchanged"] and benchmark["adapter_restored"]
    assert all(benchmark["online_adapter_changed_per_stream"])

    continuation = tmp_path / "continued.pt"
    resumed_args = arguments.copy()
    resumed_args[resumed_args.index(str(output))] = str(continuation)
    resumed_args.extend(["--initial-adapter", str(output)])
    monkeypatch.setattr(sys, "argv", resumed_args)
    cli.main()
    resumed_payload = torch.load(continuation, weights_only=True)
    import hashlib
    assert resumed_payload["metadata"]["initial_adapter_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert resumed_payload["base_fingerprint"] == payload["base_fingerprint"]
    assert any(not torch.equal(value, resumed_payload["state"][name]) for name, value in payload["state"].items())
