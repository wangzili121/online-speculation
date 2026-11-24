from types import SimpleNamespace
import inspect
from pathlib import Path

import torch

from blockspec.dsv41_serving_online import DSV41SequenceSpan
from blockspec.dsv41_suffix import DSV41SuffixPlan
from blockspec.pypto_dsv41_runtime import DraftRound


class ReplayPool:
    def __init__(self):
        self.retained = set()

    @staticmethod
    def workspace(shape, dtype, rank, *, fill=0):
        return torch.empty(shape, dtype=dtype)

    @staticmethod
    def copy_tensor(destination, source, rank):
        destination.copy_(source)

    def retain_replay_values(self, values):
        self.retained.update((rank, tensor.untyped_storage()._cdata) for rank, tensor in values)

    def release_replay_values(self):
        self.retained.clear()


def test_last_n_replay_starts_before_the_first_selected_layer():
    owner = (0, 3, 9)
    plan = DSV41SuffixPlan(2)
    attention = SimpleNamespace(
        rank_ids={0: 0}, topology={"tp": 4}, config={"num_hidden_layers": 40},
        resident=ReplayPool(),
        native=lambda name: SimpleNamespace(
            resolve_layer_role=lambda layer: {"index_source": None},
        ),
    )
    histories = {(0, layer, owner): {} for layer in plan.layers}
    round_ = DraftRound(attention, histories, capture=(plan, {owner: ("r", (11, 12))}))

    for layer in plan.layers:
        normalized = object()
        round_.captured_attention[0, layer, owner] = {
            "rank": 0, "cos": object(), "sin": object(), "history": object(),
            "selection": object(), "qkv": {"hidden": normalized},
            "contribution": object(), "lse": object(),
        }
        round_.captured_tail[0, layer, owner] = {
            "readonly_hidden": f"layer-{layer}-input",
            "incoming": f"layer-{layer}-premix",
            "hidden": object(), "pre": object(), "post": object(), "comb": object(),
            "attn_collapsed": object(), "changed": object(),
            "attention_scattered": object(),
            "ffn_pre": object(), "ffn_post": object(), "ffn_comb": object(),
            "ffn_collapsed": object(), "ffn_normed": object(), "router_input": object(),
            "router_logits": object(), "route_ids": object(), "weights": object(),
            "shared": object(), "expert_outputs": object(), "combined": object(),
            "output_hidden": object(),
        }

    replay = round_.replays()[owner]
    assert replay.boundary == {
        0: {"hidden": "layer-38-input", "pre_mix": "layer-38-premix"},
    }
    first_attention = replay.layers[0].attention[0]
    assert first_attention["normed"] is first_attention["qkv"]["hidden"]
    assert first_attention["normed"] is not replay.boundary[0]["hidden"]
    assert replay.layers[-1].ffn[0]["output_pre_mix"] is round_.captured_tail[
        0, 39, owner
    ]["ffn_pre"]


def test_real_capture_preserves_the_exact_k32_token_span():
    owner = (0, 3, 9)
    plan = DSV41SuffixPlan(1)
    attention = SimpleNamespace(
        rank_ids={1: 0}, topology={"tp": 4}, config={"num_hidden_layers": 40},
        resident=ReplayPool(),
        native=lambda name: SimpleNamespace(
            resolve_layer_role=lambda layer: {"index_source": None},
        ),
    )
    round_ = DraftRound(attention, {(1, 39, owner): {}}, capture=(plan, {owner: ("r", (11, 12))}))
    normalized = torch.ones(32, 64)
    round_.captured_attention[1, 39, owner] = {
        "rank": 1, "cos": object(), "sin": object(), "history": object(),
        "selection": object(), "qkv": {"hidden": normalized},
        "contribution": object(), "lse": object(),
    }
    hidden = torch.ones(32, 4, 64)
    premix = torch.ones(32, 4)
    state = {
        "readonly_hidden": hidden, "incoming": premix, "hidden": hidden,
        "pre": object(), "post": object(), "comb": object(), "attn_collapsed": object(),
        "changed": object(), "ffn_pre": premix,
        "attention_scattered": object(),
        "ffn_post": object(), "ffn_comb": object(), "ffn_collapsed": object(),
        "ffn_normed": object(), "router_input": object(), "router_logits": object(),
        "route_ids": object(), "weights": object(), "shared": object(),
        "expert_outputs": object(), "combined": object(), "output_hidden": hidden,
        "original": {"request_token_map": [{
            "request_slot": 3, "token_begin": 16, "token_end": 48,
        }], "cp_token_range": (16, 32, 16, 64), "SP_valid_rows": 16},
    }
    # The native draft branch leaves full_norm out of the FFN state dictionary.
    round_.observe_layer(39, {"states": [{}, {"fresh": state}]})
    round_.finish_layer(39)
    hidden.zero_()
    premix.zero_()
    replay = round_.replays()[owner]
    captured = replay.layers[0]
    for row in (captured.attention[1], captured.ffn[1]):
        assert (row["token_begin"], row["token_end"]) == (16, 48)
        assert row["sp_span"] == DSV41SequenceSpan(0, 0, 16)
    assert captured.ffn[1]["output_hidden"].all()
    assert captured.ffn[1]["output_pre_mix"].all()
    assert (0, normalized.untyped_storage()._cdata) in attention.resident.retained
    assert (0, hidden.untyped_storage()._cdata) not in attention.resident.retained
    assert replay.suffix().owner == owner
    replay.suffix().release()
    assert not attention.resident.retained


def test_replay_uses_the_real_serving_ffn_state_names():
    source = (Path(__file__).parents[1] / "runtime" / "pypto_dsv41_a3" /
              "lib-online-state-capture.patch").read_text()
    for name in ("ffn_collapsed", "ffn_normed", "router_input", "router_logits", "output_hidden"):
        assert f"{name}=" in source
    assert 'state["padded_hidden"]' not in inspect.getsource(DraftRound.replays)
