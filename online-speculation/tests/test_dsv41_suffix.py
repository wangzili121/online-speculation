import torch

from blockspec.dsv41_suffix import DSV41SuffixLearner, DSV41SuffixPlan, DSV41SuffixReplay


def _parameters(plan, hidden=8):
    shapes = {
        "wq_a.weight": (hidden, hidden), "q_norm.weight": (hidden,),
        "wq_b.weight": (hidden, hidden), "wkv.weight": (hidden, hidden),
        "kv_norm.weight": (hidden,), "attn_sink": (1,),
        "wo_a.weight": (hidden, hidden), "wo_b.weight": (hidden, hidden),
    }
    return {f"layers.{layer}.attn.{name}": torch.nn.Parameter(torch.randn(shape) * .02)
            for layer in plan.layers for name, shape in shapes.items()}


def test_suffix_plan_selects_last_n_and_freezes_reindexer():
    plan = DSV41SuffixPlan(4)
    assert plan.layers == (36, 37, 38, 39)
    assert plan.reindex_layers == (36,)
    assert len(plan.trainable_names()) == 32
    assert not any("indexer" in name for name in plan.trainable_names())


def test_last_n_recomputes_downstream_layers_and_updates_every_attention():
    torch.manual_seed(9)
    plan = DSV41SuffixPlan(4)
    parameters = _parameters(plan)
    calls = []

    def layer(layer, hidden, state, weights):
        calls.append(layer)
        # A small differentiable stand-in: trainable Attention followed by a
        # frozen nonlinear tail.  It verifies suffix orchestration and gradient
        # ownership without pretending to be the production DSV4.1 kernel.
        update = hidden @ weights["wq_a.weight"].T
        update = update @ weights["wq_b.weight"].T
        update = update @ weights["wkv.weight"].T
        update = update @ weights["wo_a.weight"].T @ weights["wo_b.weight"].T
        scale = weights["q_norm.weight"].mean() * weights["kv_norm.weight"].mean()
        return torch.tanh(hidden + update * scale + weights["attn_sink"].mean() + state)

    learner = DSV41SuffixLearner(
        parameters, last_layers=4, replay_layer=layer,
        project_logits=lambda x: x,
        loss=lambda student, teacher, kind: (student - teacher).square().mean(),
        learning_rate=1e-2,
    )
    replay = DSV41SuffixReplay(plan, torch.randn(32, 8), tuple(torch.randn(()) for _ in plan.layers))
    before = {name: value.detach().clone() for name, value in parameters.items()}
    loss, norm, version = learner.update_rows(replay, torch.randn(3, 8))
    assert calls == [36, 37, 38, 39]
    assert loss > 0 and norm > 0 and version == 1
    assert all(value.grad is not None and value.grad.isfinite().all() for value in parameters.values())
    assert all(not torch.equal(value, before[name]) for name, value in parameters.items())
    assert set(learner.publication_state()) == set(parameters)
