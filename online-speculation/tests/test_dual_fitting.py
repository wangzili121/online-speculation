from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors.torch import save_file
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder
from blockspec.parallel.fitting import BatchStream, FitConfig, TokenDataset, Trainer, frozen_fingerprint
from blockspec.parallel.weights import load_ar_base, load_checkpoint, public_key_map


def test_adamw_execution_choices_follow_the_same_fp32_update():
    reference = torch.nn.Parameter(torch.linspace(-1, 1, 37))
    fused = torch.nn.Parameter(reference.detach().clone())
    optimizers = [FitConfig(optimizer_impl=mode).make_optimizer([p])
                  for mode, p in (("single", reference), ("fused", fused))]
    for step in range(4):
        for parameter, optimizer in zip((reference, fused), optimizers, strict=True):
            parameter.grad = torch.sin(torch.arange(37.) + step)
            optimizer.step()
        torch.testing.assert_close(reference, fused, rtol=2e-6, atol=2e-7)
    with pytest.raises(ValueError, match="AdamW"):
        FitConfig(optimizer_impl="unknown")


def model():
    torch.manual_seed(112)
    config = replace(DualViewConfig(), vocab_size=13, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    return DualViewDecoder(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_fingerprint_buffer_matches_the_serialized_byte_definition(dtype):
    network = model().to(dtype=dtype)
    digest = hashlib.sha256()
    for name, value in network.named_parameters():
        if ".attention.draft." not in name:
            digest.update((name + str(tuple(value.shape)) + str(value.dtype)).encode())
            digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    assert frozen_fingerprint(network) == digest.hexdigest()
    with torch.no_grad():
        network.embedding.weight[0, 0].add_(1)
    assert frozen_fingerprint(network) != digest.hexdigest()


def ar_fixture(folder, source, *, sharded=False):
    config = source.config.to_dict() | {"model_type": "qwen3"}
    for key in ("block_size", "mask_token_id"):
        config.pop(key)
    (folder / "config.json").write_text(json.dumps(config))
    state = source.state_dict()
    tensors = {public: state[own].clone().contiguous()
               for own, public in public_key_map(source.config, include_draft=False).items()}
    if sharded:
        index = {}
        for shard in range(2):
            name = f"model-{shard}.safetensors"
            selected = {key: value for i, (key, value) in enumerate(tensors.items()) if i % 2 == shard}
            index.update({key: name for key in selected})
            save_file(selected, folder / name)
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(tensors, folder / "model.safetensors")


@pytest.mark.parametrize("sharded", [False, True])
def test_ar_checkpoint_initializes_independent_attention_storage(tmp_path, sharded):
    original = model()
    ar_fixture(tmp_path, original, sharded=sharded)
    loaded = load_ar_base(tmp_path, block_size=4, mask_token_id=1)
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, dict(original.named_parameters())[name])
        assert not parameter.requires_grad
    for layer in loaded.layers:
        for name, parameter in layer.attention.draft.named_parameters():
            ar = dict(layer.attention.ar.named_parameters())[name]
            assert parameter.data_ptr() != ar.data_ptr()
            assert torch.equal(parameter, ar)
    tokens = torch.tensor([[4, 7, 2]])
    torch.testing.assert_close(original(tokens).logits, loaded(tokens).logits, rtol=0, atol=0)


def test_sharded_import_checks_index_and_local_paths(tmp_path):
    ar_fixture(tmp_path, model(), sharded=True)
    path = tmp_path / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    key = next(iter(index["weight_map"]))
    index["weight_map"][key] = "../outside.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="local safetensors"):
        load_ar_base(tmp_path, block_size=4, mask_token_id=1)


def dataset(path, *, altered=False):
    records = [{"input_ids": [(i + j) % 11 + 2 for j in range(9 + i)]} for i in range(4)]
    if altered:
        records[0]["input_ids"][0] = 4
    path.write_text("\n".join(json.dumps(record) for record in records))
    return TokenDataset(path, 13, 8)


def test_data_cursor_crop_and_epoch_restore(tmp_path):
    data = dataset(tmp_path / "data.jsonl")
    stream = BatchStream(data, 99)
    stream.batch(3)
    state = stream.state_dict()
    expected = stream.batch(9)
    loaded = BatchStream(data, 1)
    loaded.load_state_dict(state)
    assert torch.equal(loaded.batch(9), expected)
    assert loaded.epoch == stream.epoch
    state["order"][0] = state["order"][1]
    with pytest.raises(ValueError, match="permutation"):
        loaded.load_state_dict(state)


CASES = [("cpu", "fp32")] + ([("cuda", "fp32"), ("cuda", "bf16")] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device,precision", CASES)
@pytest.mark.parametrize("optimizer_impl", ["single", "fused"])
def test_complete_training_resume_matches_every_update(tmp_path, device, precision, optimizer_impl):
    data = dataset(tmp_path / "data.jsonl")
    config = FitConfig(steps=5, batch_size=2, sequence_length=8, anchors_per_sequence=2,
                       accumulate=2, chunk_rows=3, warmup_steps=1, precision=precision, optimizer_impl=optimizer_impl)
    original = model().to(device)
    base = frozen_fingerprint(original)
    full = Trainer(original, data, config)
    full_records = full.run()
    split = Trainer(model().to(device), data, config)
    first_records = split.run(2)
    path = tmp_path / "resume.pt"
    split.save(path)
    resumed = Trainer.resume(path, data, device=device)
    rest_records = resumed.run()
    for full_row, split_row in zip(full_records, first_records + rest_records, strict=True):
        for key in ("loss", "gradient_norm", "learning_rate", "step"):
            assert full_row[key] == split_row[key], (device, precision, key)
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert frozen_fingerprint(resumed.model) == base
    assert full.evaluate(data) == resumed.evaluate(data)
    assert torch.equal(full.stream.order, resumed.stream.order)
    assert torch.equal(full.anchors_rng.get_state(), resumed.anchors_rng.get_state())
    changed = dataset(tmp_path / "changed.jsonl", altered=True)
    with pytest.raises(ValueError, match="data SHA256"):
        Trainer.resume(path, changed, device=device)
    with pytest.raises(FileExistsError):
        resumed.save(path)


def test_validation_preserves_training_random_streams(tmp_path):
    data = dataset(tmp_path / "data.jsonl")
    trainer = Trainer(model(), data, FitConfig(steps=3, warmup_steps=0, sequence_length=8))
    before = trainer.stream.state_dict()
    anchors = trainer.anchors_rng.get_state().clone()
    trainer.evaluate(data)
    after = trainer.stream.state_dict()
    assert torch.equal(before["rng"], after["rng"])
    assert torch.equal(anchors, trainer.anchors_rng.get_state())
    assert trainer.model.training


def test_schedule_warmup_and_decay_boundaries():
    config = FitConfig(steps=6, warmup_steps=2, learning_rate=.01, minimum_lr_ratio=.2)
    assert [config.rate(i) for i in (0, 1, 2, 5)] == pytest.approx([.005, .01, .01, .002])
    with pytest.raises(ValueError):
        FitConfig(steps=2, warmup_steps=3)


def test_cli_train_interruption_and_resume_match_uninterrupted(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    ar_fixture(base, model(), sharded=True)
    data = dataset(tmp_path / "data.jsonl")
    script = Path(__file__).resolve().parents[1] / "scripts/train_dual_view.py"
    train = [sys.executable, str(script), "train", "--base", str(base), "--mask-token-id", "1",
             "--block-size", "4", "--data", str(data.path), "--steps", "4", "--warmup-steps", "1",
             "--sequence-length", "8", "--learning-rate", ".001"]
    partial, complete, continued = [tmp_path / name for name in ("part.pt", "full.pt", "resumed.pt")]
    for command in (train + ["--stop-after", "2", "--output", str(partial)],
                    [sys.executable, str(script), "resume", "--checkpoint", str(partial),
                     "--data", str(data.path), "--output", str(continued)],
                    train + ["--output", str(complete)]):
        subprocess.run(command, check=True, capture_output=True, text=True)
    full, full_state = load_checkpoint(complete)
    resumed, resumed_state = load_checkpoint(continued)
    assert full_state["step"] == resumed_state["step"] == 4
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())
