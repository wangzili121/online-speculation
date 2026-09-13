from __future__ import annotations

import ast
import hashlib
import sys

import pytest

from experiments.dllm import download_llada


def test_checkpoint_validation_uses_size_and_official_digest(monkeypatch, tmp_path):
    content = b"pinned-llada-shard"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        download_llada,
        "WEIGHTS",
        (("model-00001-of-00001.safetensors", len(content), digest),),
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(content)

    result = download_llada.validate_checkpoint(tmp_path)

    assert result["weights"]["model-00001-of-00001.safetensors"] == {
        "bytes": len(content),
        "sha256": digest,
    }


def test_checkpoint_validation_rejects_corruption(monkeypatch, tmp_path):
    content = b"corrupted"
    monkeypatch.setattr(
        download_llada,
        "WEIGHTS",
        (("model.safetensors", len(content), "0" * 64),),
    )
    (tmp_path / "model.safetensors").write_bytes(content)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_llada.validate_checkpoint(tmp_path)


def test_cli_reads_output_directory_from_paired_config(monkeypatch, tmp_path, capsys):
    model_dir = tmp_path / "model"
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\npath = "{model_dir.as_posix()}"\n', encoding="utf-8")
    observed = []

    def fake_validate(path):
        observed.append(path)
        return {"path": str(path)}

    monkeypatch.setattr(download_llada, "validate_checkpoint", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_llada.py", "--config", str(config), "--validate-only"],
    )

    download_llada.main()

    assert observed == [model_dir]
    assert ast.literal_eval(capsys.readouterr().out)["path"] == str(model_dir)


def test_cli_reuses_a_valid_local_checkpoint_without_network(
    monkeypatch, tmp_path, capsys
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    expected = {"path": str(model_dir), "status": "valid"}
    monkeypatch.setattr(download_llada, "validate_checkpoint", lambda path: expected)

    def fail_download(*args, **kwargs):
        raise AssertionError("a valid local checkpoint must not access the network")

    monkeypatch.setattr(download_llada, "huggingface_snapshot_download", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_llada.py", "--output", str(model_dir)],
    )

    download_llada.main()

    assert ast.literal_eval(capsys.readouterr().out) == expected
