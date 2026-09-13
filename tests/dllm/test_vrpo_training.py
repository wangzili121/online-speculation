from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from experiments.dllm.train_gsm8k_vrpo import _save_checkpoint


class SaveableAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def save_pretrained(self, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_checkpoint_persists_resume_metrics_and_cumulative_cost(tmp_path):
    model = SaveableAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = {"current_policy": 11, "reference_policy": 11, "total": 22}

    _save_checkpoint(
        model,
        optimizer,
        tmp_path,
        update=3,
        fingerprint="fingerprint",
        metrics=({"update": 3, "loss": 0.5},),
        cost_slots=slots,
        elapsed_seconds=2.5,
    )

    state = json.loads((tmp_path / "training_state.json").read_text(encoding="utf-8"))
    assert state["completed_updates"] == 3
    assert state["forward_token_slots"] == slots
    assert state["elapsed_seconds"] == 2.5
    assert (tmp_path / "optimizer.pt").is_file()
    assert (tmp_path / "adapter_config.json").is_file()
