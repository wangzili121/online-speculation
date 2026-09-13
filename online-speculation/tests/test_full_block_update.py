"""File-backed and separated streaming parameters share the same update."""

from contextlib import nullcontext
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder
from blockspec.parallel.training import distillation_update, sample_anchors


def tiny(device):
    torch.manual_seed(371)
    return DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2,
                                          num_key_value_heads=1, head_dim=8)).to(device)


CASES = [("cpu", "fp32"), pytest.param("cuda", "fp32", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA execution check")), pytest.param("cuda", "bf16",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 execution check"))]


@pytest.mark.parametrize("device,precision", CASES)
def test_offline_and_streaming_masters_match_every_update(device, precision):
    offline = tiny(device).train_draft_only()
    online = copy.deepcopy(offline).requires_grad_(False).eval()
    initial = {name: p.detach().clone() for name, p in online.named_parameters()}
    resident = {name: p for name, p in offline.named_parameters() if ".attention.draft." in name}
    master = {name: torch.nn.Parameter(p.detach().clone()) for name, p in resident.items()}
    optimizers = [torch.optim.AdamW(list(params.values()), lr=.001, foreach=False)
                  for params in (resident, master)]
    rng = torch.Generator().manual_seed(732)
    context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if precision == "bf16" else nullcontext
    for step in range(4):
        batches = []
        for _ in range(2):
            tokens = torch.randint(2, 17, (1, 12), generator=rng)
            anchors = sample_anchors(tokens, 4, 3, generator=rng)
            batches.append((tokens.to(device), anchors.to(device)))
        settings = dict(learning_rate=.001 / (step + 1), chunk_rows=3, autocast=context)
        first = distillation_update(offline, batches, optimizers[0], **settings)
        second = distillation_update(online, batches, optimizers[1], master=master, **settings)
        assert first == second
        for name, parameter in resident.items():
            assert torch.equal(parameter, master[name]), (device, precision, step, name)
            a, b = optimizers[0].state[parameter], optimizers[1].state[master[name]]
            assert all(torch.equal(value, b[key]) for key, value in a.items())
        assert all(torch.equal(p, initial[name]) for name, p in online.named_parameters())
    assert any(not torch.equal(p, initial[name]) for name, p in master.items())
    assert all(torch.equal(p, initial[name]) for name, p in offline.named_parameters() if name not in resident)


def test_optimizer_parameter_contract():
    model = tiny("cpu").train_draft_only()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters[:-1], lr=.001)
    with pytest.raises(ValueError, match="exactly the full"):
        distillation_update(model, [(torch.tensor([[2, 3, 4, 5]]), torch.tensor([[0]]))],
                             optimizer, learning_rate=.001)


def deterministic_cuda_worker():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(744)
    model = DualViewDecoder(DualViewConfig(vocab_size=257, hidden_size=256, intermediate_size=512,
                                           num_hidden_layers=4, num_attention_heads=4,
                                           num_key_value_heads=1, head_dim=64, block_size=32))
    model = model.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False).eval()
    weights = {name: p for name, p in model.named_parameters() if ".attention.draft." in name}
    masters = [{name: torch.nn.Parameter(p.detach().float().clone()) for name, p in weights.items()} for _ in range(2)]
    optimizers = [torch.optim.AdamW(list(master.values()), lr=.0002, foreach=False) for master in masters]
    rng = torch.Generator().manual_seed(744)
    for _ in range(3):
        tokens = torch.randint(2, 257, (1, 256), generator=rng)
        anchors = sample_anchors(tokens, 32, 4, generator=rng)
        batches = [(tokens.cuda(), anchors.cuda())]
        records = [distillation_update(model, batches, optimizer, learning_rate=.0002, master=master,
                                       autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16))
                   for master, optimizer in zip(masters, optimizers, strict=True)]
        assert records[0] == records[1]
        assert all(torch.equal(p, masters[1][name]) for name, p in masters[0].items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA deterministic backward check")
def test_deterministic_block_backward_with_fused_attention_shapes():
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True, capture_output=True,
                   text=True, env=os.environ | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


if __name__ == "__main__":
    deterministic_cuda_worker()
