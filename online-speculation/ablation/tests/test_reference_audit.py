"""Oracle source gates are tested without importing any external model code."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest
import torch


def module(filename="audit_hf_reference.py"):
    spec = importlib.util.spec_from_file_location("base_audit", Path(__file__).resolve().parents[1] /
                                                 "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_decode_path_summary_exposes_a_near_tie_argmax_change():
    audit = module("audit_decode_path.py")
    result = audit.compare_target_rows(torch.tensor([1., 1.00001, -1.]),
                                       torch.tensor([1.00002, 1., -1.]))
    assert result["ar_top_ids"][0] == 1 and result["spec_top_ids"][0] == 0
    assert result["ar_argmax"] == 1 and result["spec_argmax"] == 0
    assert 0 < result["ar_margin"] < 2e-5 and 0 < result["tv"] < 2e-5


@torch.no_grad()
def test_decode_path_trace_keeps_base_rows_and_exact_tree_ancestry():
    from blockspec_ablation.execution import FixedShapeExecutor
    from blockspec_ablation.model import Decoder, ModelConfig

    audit = module("audit_decode_path.py")
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, adapter_rank=2))
    engine = FixedShapeExecutor(model, capacity=16, max_query=3, use_cuda_graph=False)
    engine.prepare([(2, True, None), (3, False, None)])
    past = model(torch.tensor([[0, 1, 2, 3, 4]]), return_cache=True)[1]
    trace = audit.TargetTrace(engine, 6)
    trace._forward(torch.tensor([[5, 6]]), cache=past, return_cache=True,
                   adapter_mask=torch.tensor([[False, True]]))
    assert trace.rows == []
    allowed = torch.ones(1, 1, 3, 8, dtype=torch.bool)
    allowed[..., 5:] = torch.tensor([[True, False, False], [True, True, False], [True, False, True]])
    trace._forward(torch.tensor([[5, 7, 8]]), positions=torch.tensor([[5, 6, 6]]),
                   allowed=allowed, cache=past, return_cache=True)
    assert [row["path"] for row in trace.rows] == [[5, 7], [5, 8]]
    assert all(row["prefix"] == 5 and row["kind"] == "base" for row in trace.rows)


def test_local_online_audit_replays_prior_requests_and_benchmark_seed(tmp_path, monkeypatch, capsys):
    from blockspec_ablation.checkpoint import save_checkpoint
    from blockspec_ablation.model import Decoder, ModelConfig

    audit = module("audit_decode_path.py")
    torch.manual_seed(419)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2, adapter_alpha=2))
    checkpoint = tmp_path / "adapter.pt"
    save_checkpoint(checkpoint, model, adapter_only=True)
    data = tmp_path / "validation.jsonl"
    data.write_text('\n'.join(json.dumps({"input_ids": ids}) for ids in ([0, 1, 2, 3], [0, 2, 1, 4])))
    monkeypatch.setattr(audit, "load_hf_base", lambda *args, **kwargs: model)
    monkeypatch.setattr(sys, "argv", ["audit", "--base", str(tmp_path), "--adapter", str(checkpoint),
                                     "--data", str(data), "--request", "1", "--token-index", "0",
                                     "--tokens", "8", "--prompt-length", "3", "--block-size", "3",
                                     "--top-k", "2", "--prefix-budget", "5", "--device", "cpu",
                                     "--execution", "eager", "--online-stream", "--stream-prompts", "2",
                                     "--repeat-index", "1", "--stream-seed", "71", "--online-last-layers", "1",
                                     "--update-stride", "1", "--update-policy", "periodic", "--online-execution", "eager"])
    threads = torch.get_num_threads()
    try:
        audit.main()
    finally:
        torch.set_num_threads(threads)
    result = json.loads(capsys.readouterr().out)
    assert result["request_seed"] == 74
    assert result["adapter_version_after_request"] > result["adapter_version_before_request"] > 0
    assert result["adapter"]["kind"] == "local_training"
    assert result["comparison"]["identical"]
    assert result["target_rows"] and all(row["tv"] < 1e-6 for row in result["target_rows"])


def test_error_summary_and_position_weighted_aggregation():
    audit = module()
    zeros = torch.zeros(2, 7)
    a = audit.error_summary(zeros, zeros)
    b = audit.error_summary(torch.ones(1, 7), torch.zeros(1, 7))
    result = audit.summarize([a, b])
    assert result["positions"] == 3 and result["mean_abs"] == pytest.approx(1 / 3)
    assert result["max_abs"] == 1 and result["argmax_mismatches"] == result["mean_tv"] == 0
    with pytest.raises(ValueError, match="invalid"):
        audit.error_summary(zeros, torch.full_like(zeros, float("nan")))


def test_reference_gate_rejects_modified_code_weights_and_index(tmp_path):
    audit = module()
    source = b"raise RuntimeError('NEVER IMPORT THIS TEST SOURCE')\n"
    (tmp_path / "modeling.py").write_bytes(source.replace(b"\n", b"\r\n"))
    (tmp_path / "weights.safetensors").write_bytes(b"placeholder")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"models": {"base": {
        "reference_lf_sha256": {"modeling.py": hashlib.sha256(source).hexdigest()},
        "weight_filename": "weights.safetensors", "weight_sha256": hashlib.sha256(b"placeholder").hexdigest(),
    }}}))
    assert audit.checked_reference(tmp_path, lock)["weight_filename"] == "weights.safetensors"
    index.write_text(json.dumps({"weight_map": {"x": "unchecked.safetensors"}}))
    with pytest.raises(ValueError, match="index"):
        audit.checked_reference(tmp_path, lock)
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights differ"):
        audit.checked_reference(tmp_path, lock)
    (tmp_path / "modeling.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source/config differs"):
        audit.checked_reference(tmp_path, lock)


@pytest.mark.parametrize("change", ["crlf", "source", "extra", "commit"])
def test_external_engine_gate_compares_canonical_git_blobs(tmp_path, change):
    audit = module("benchmark_reference.py")
    source = tmp_path / "engine.py"
    original = b"VALUE = 1\n"
    source.write_bytes(original)
    git = ["git", "-C", str(tmp_path), "-c", "core.autocrlf=false",
           "-c", "user.name=Reference Test", "-c", "user.email=reference-test@example.invalid"]
    subprocess.run(git + ["init", "-q"], check=True, capture_output=True)
    subprocess.run(git + ["add", "engine.py"], check=True, capture_output=True)
    subprocess.run(git + ["commit", "-qm", "test source"], check=True, capture_output=True)
    commit = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert audit.check_source(tmp_path, commit) == tmp_path.resolve()
    if change == "crlf":
        source.write_bytes(original.replace(b"\n", b"\r\n"))
        assert audit.check_source(tmp_path, commit) == tmp_path.resolve()
    else:
        if change == "source":
            source.write_bytes(b"VALUE = 2\n")
        elif change == "extra":
            (tmp_path / "extra.py").write_bytes(original)
        else:
            commit = "0" * 40
        with pytest.raises(ValueError, match="reference"):
            audit.check_source(tmp_path, commit)


def test_external_engine_weights_and_token_weighted_summary(tmp_path):
    audit = module("benchmark_reference.py")
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    lock = {"models": {}}
    for key, path in (("base", base), ("adapter", adapter)):
        payload = key.encode()
        (path / "weights").write_bytes(payload)
        lock["models"][key] = {"weight_filename": "weights", "weight_sha256": hashlib.sha256(payload).hexdigest()}
    assert set(audit.check_artifacts(base, adapter, lock)) == set(lock["models"])
    (adapter / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.check_artifacts(base, adapter, lock)
    rows = [{"arm": arm, "tokens": tokens, "seconds": seconds, "initialization_seconds": 2,
             "warmup_seconds": 1, "stats": {"forwards": 10, "accepts": 20}}
            for arm, tokens, seconds in (("ar", 10, 1), ("ar", 30, 3), ("static", 10, .5), ("static", 30, 1.5))]
    result = audit.summarize(rows)
    assert result["arms"]["ar"]["tps"] == 10
    assert result["arms"]["static"]["tps"] == 20 and result["speedup"] == 2
    assert result["arms"]["static"]["tokens_per_sequence_forward"] == 2
    assert result["arms"]["static"]["initialization_seconds"] == 4


def test_reference_layer_trace_removes_hooks():
    from blockspec_ablation.model import Decoder, ModelConfig

    audit = module()
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2))
    result = audit.trace_layers(model, model, torch.tensor([[0, 1, 2]]))
    assert result and all(row["max_abs"] == row["mean_abs"] == 0 for row in result)
    assert all(not child._forward_hooks for child in model.modules())


@pytest.mark.parametrize("fails", [False, True])
def test_reference_restores_accumulation_policy(monkeypatch, fails):
    audit = module()
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

    def run(args):
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if fails:
            raise RuntimeError("test audit failure")
        return "complete"

    monkeypatch.setattr(audit, "run", run)
    monkeypatch.setattr(sys, "argv", ["audit", "--base", ".", "--reference-manifest", "local/reference.local.json", "--bf16-full-accumulation"])
    if fails:
        with pytest.raises(RuntimeError, match="test audit failure"):
            audit.main()
    else:
        assert audit.main() == "complete"
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
