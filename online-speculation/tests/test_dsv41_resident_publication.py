import torch

from blockspec.pypto_dsv41_runtime import MutableDraftHeadBank


class Resident:
    def __init__(self):
        self.pins = []
        self.syncs = []

    def pin(self, value, ranks, *, readonly, name):
        assert not readonly
        self.pins.append((name, tuple(ranks), value))

    def handle(self, value, rank):
        return value

    def tensor(self, value, rank):
        self.syncs.append((rank, value.detach().clone()))
        return value


def test_mutable_bank_stages_suffix_then_switches_all_layers_atomically():
    resident = Resident()
    ranks, layers = {0: 0, 1: 1}, (38, 39)
    initial = {
        (rank, layer): {"weight": torch.full((2, 2), rank + layer, dtype=torch.bfloat16)}
        for rank in ranks for layer in layers
    }
    bank = MutableDraftHeadBank(resident, ranks, initial, layers)
    assert len(resident.pins) == 2 * len(ranks) * len(layers)
    before = bank.operands(0, 38)["weight"].clone()
    values = {
        f"layers.{layer}.attn.weight": torch.full((2, 2), 100 + layer, dtype=torch.bfloat16)
        for layer in layers
    }
    prepared = bank.stage(values, 1, lambda full, rank: {"weight": full["weight"] + rank})
    assert prepared.phase == "prepared" and bank.active_version == 0
    torch.testing.assert_close(bank.operands(0, 38)["weight"], before)
    activated = bank.activate(1)
    assert activated.phase == "activated" and activated.digest == prepared.digest
    torch.testing.assert_close(
        bank.operands(1, 39)["weight"], torch.full((2, 2), 140, dtype=torch.bfloat16),
    )
    bank.commit(1)
    bank.stage(values, 2, lambda full, rank: {"weight": full["weight"] + rank})
    bank.rollback(2)
    assert bank.active_version == 1 and bank.staged is None


def test_mutable_bank_can_reverse_an_activation_before_cluster_commit():
    resident = Resident()
    ranks, layers = {0: 0}, (39,)
    initial = {(0, 39): {"weight": torch.zeros((1,), dtype=torch.bfloat16)}}
    bank = MutableDraftHeadBank(resident, ranks, initial, layers)
    values = {"layers.39.attn.weight": torch.ones((1,), dtype=torch.bfloat16)}
    old = bank.operands(0, 39)["weight"]
    bank.stage(values, 1, lambda full, rank: {"weight": full["weight"]})
    bank.activate(1)
    assert bank.active_version == 1 and bank.operands(0, 39)["weight"].item() == 1
    bank.rollback(1)
    assert bank.active_version == 0 and bank.active_slot == 0
    assert bank.operands(0, 39)["weight"] is old
