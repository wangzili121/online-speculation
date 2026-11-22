from types import SimpleNamespace

from blockspec.dsv41_suffix import DSV41SuffixPlan
from blockspec.pypto_dsv41_runtime import DraftRound


def test_last_n_replay_starts_before_the_first_selected_layer():
    owner = (0, 3, 9)
    plan = DSV41SuffixPlan(2)
    attention = SimpleNamespace(
        rank_ids={0: 0}, topology={"tp": 4}, config={"num_hidden_layers": 40},
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
            "selection": object(), "qkv": object(), "contribution": object(),
        }
        round_.captured_tail[0, layer, owner] = {
            "readonly_hidden": f"layer-{layer}-input",
            "incoming": f"layer-{layer}-premix",
            "hidden": object(), "pre": object(), "post": object(), "comb": object(),
            "attn_collapsed": object(), "full_norm": normalized, "changed": object(),
            "ffn_pre": object(), "ffn_post": object(), "ffn_comb": object(),
            "ffn_collapsed": object(), "ffn_normed": object(), "router_input": object(),
            "router_logits": object(), "route_ids": object(), "weights": object(),
            "shared": object(), "expert_outputs": object(), "combined": object(),
            "padded_hidden": object(),
        }

    replay = round_.replays()[owner]
    assert replay.boundary == {
        0: {"hidden": "layer-38-input", "pre_mix": "layer-38-premix"},
    }
    assert replay.layers[0].attention[0]["normed"] is not replay.boundary[0]["hidden"]
    assert replay.layers[-1].ffn[0]["output_pre_mix"] is round_.captured_tail[
        0, 39, owner
    ]["ffn_pre"]
