"""有效块长同时控制双向范围和训练反馈，物理存储仍为两组 K32。"""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner, draft_inputs


def config():
    return NS(speculative_tokens=0, prefix_cache=False, max_model_len=512,
              long_prefill_token_threshold=4, max_num_batched_tokens=128, max_num_seqs=2)


@pytest.mark.parametrize("width", [0, 1, 33, True])
def test_invalid_width_is_rejected_before_weights_or_device_are_opened(width):
    with pytest.raises(ValueError, match="draft width"):
        OrthrusTP2Runner(config(), "unused", draft_width=width)


def test_candidate_or_recurrence_policy_is_rejected_before_device_start():
    with pytest.raises(ValueError, match="candidate budget"):
        OrthrusTP2Runner(config(), "unused", draft_width=8, output_candidates=8)
    with pytest.raises(ValueError, match="boolean"):
        OrthrusTP2Runner(config(), "unused", gdn_valid_rows="valid")


@pytest.mark.parametrize("width", [8, 16, 32])
def test_two_requests_keep_separate_valid_boundaries_in_attention_and_feedback(width):
    runner = object.__new__(OrthrusTP2Runner)
    runner.draft_width, runner.settings = width, NS(max_model_len=512)
    runner.head = {"source_head_contract": {"mask_token_id": 999}}
    runner._host_slabs = {f"orthrus_2_{name}": torch.empty((2, *shape), dtype=dtype)
                          for name, shape, dtype in draft_inputs(2)}
    runner.cos = torch.full((512, 64), .5, dtype=torch.bfloat16)
    runner.sin = torch.full((512, 64), .25, dtype=torch.bfloat16)
    runner.tables = torch.arange(16, dtype=torch.int32).reshape(2, 2, 4)
    runner.requests = {"a": {"slot": 0, "pages": NS(length=11)},
                       "b": {"slot": 1, "pages": NS(length=506)}}
    runner.orthrus_pair = object()
    runner.draft_buffers = {2: (object(), object(), object())}
    runner.draft_banks, runner.suffix, runner.resident = (), (), (object(),) * 4
    runner.group_layers, runner.runconfig, runner.feedback_hidden = 64, object(), object()
    captured, calls = [], []
    runner.online = NS(capture_history=lambda *args: captured.append(args), capture_request=lambda slot: None)
    class BeforeDevice(Exception):
        pass
    def stop(method, *args, **kwargs):
        assert method == "run"
        calls.append(args)
        raise BeforeDevice
    runner._worker_call = stop
    a = NS(request=NS(request_id="a"), position=11, token_ids=[123])
    b = NS(request=NS(request_id="b"), position=506, token_ids=[456])
    with pytest.raises(BeforeDevice):
        runner.propose_many([(a, 7), (b, 5)], feedback_index=0)
    assert calls[0][-7:-1] == (0, 11, width, 1, 506, 6)
    assert captured[0][:4] == (0, 11, width, 7)
    rows = runner._host_slabs["orthrus_2_rows"]
    assert (rows[:, :width] == 0).all() and (rows[:, width:32] == -1).all()
    assert (rows[:, 32:38] == 1).all() and (rows[:, 38:] == -1).all()
    mask = runner._host_slabs["orthrus_2_bias"]
    assert (mask[:, 0, :11 + width] == 0).all() and (mask[:, 0, 11 + width:] == -10000).all()
    assert (mask[:, 1] == 0).all()
    valid = runner._host_slabs["orthrus_2_valid"]
    assert valid.sum() == 24 and (valid[:, 7:32] == 0).all() and (valid[:, 37:] == 0).all()
    assert [runner.requests[k]["pages"].length for k in ["a", "b"]] == [11, 506]
