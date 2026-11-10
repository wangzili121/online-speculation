"""多层参数布局与发布边界。"""

import pytest
import torch

from test_pypto_qwen38_suffix_training import small_suffix

def test_flat_layout_preserves_gdn_and_attention_norm_semantics():
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    layout = FlatLayout(small_suffix(4).parameters)
    assert layout.elements%4096 == 0
    assert len(layout.entries) == 33
    by_name = {e.parameter.name:e for e in layout.entries}
    assert by_name["l60_a_log"].special is not None
    assert by_name["l60_norm_weight"].special is None
    assert by_name["l63_q_gamma"].parameter.offset_norm
    assert not by_name["l60_norm_weight"].parameter.offset_norm
    assert all(e.offset%64 == 0 for e in layout.entries)


def test_multi_layer_publication_is_atomic():
    from types import SimpleNamespace
    from blockspec.pypto_qwen38.weights import DraftWeights
    gdn = tuple(torch.zeros((1,64)) for _ in range(9))
    full = tuple(torch.zeros((1,64)) for _ in range(6))
    weights = DraftWeights(SimpleNamespace(banks=lambda layer:full if layer == 63 else gdn))
    weights.worker = object()
    with pytest.raises(ValueError):
        weights.publish_many({62:gdn,63:full[:5]},expected_version=0)
    assert weights.version == 0 and not weights.overrides
    weights.publish_many({62:gdn,63:full},expected_version=0)
    assert weights.version == 1 and set(weights.overrides) == {62,63}
    with pytest.raises(RuntimeError):
        weights.publish_many({62:gdn},expected_version=0)
    assert weights.version == 1


def test_multi_layer_config_requires_a_separate_worker():
    from blockspec.pypto_qwen38.online import OnlineConfig
    for n in (2,3,4,5,8,64):
        with pytest.raises(ValueError,match="separate"):
            OnlineConfig(last_layers=n)
        assert OnlineConfig(last_layers=n,learner_device=1).last_layers == n


def test_suffix_checkpoint_rejects_another_layout(tmp_path):
    import json
    from safetensors.torch import save_file
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    from blockspec.pypto_qwen38.online_checkpoint import sha256,STATES
    from blockspec.pypto_qwen38.online_suffix_checkpoint import FORMAT,contract,read
    layout = FlatLayout(small_suffix(2).parameters)
    values = {name:torch.zeros(layout.shape) for name in STATES}
    path = tmp_path/"optimizer.safetensors"
    save_file(values,str(path))
    head = {"base":"a","head":"b"}
    manifest = dict(format=FORMAT,head=head,step=2,layout=contract(layout),beta1=.9,beta2=.999,
        epsilon=1e-8,weight_decay=0.,payload_sha256=sha256(path))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    step,actual = read(tmp_path,head,layout)
    assert step == 2 and set(actual) == set(STATES)
    with pytest.raises(ValueError,match="incompatible"):
        read(tmp_path,head,FlatLayout(small_suffix(4).parameters))
    with pytest.raises(ValueError,match="incompatible"):
        read(tmp_path,{"base":"a","head":"other"},layout)
    with path.open("ab") as file:
        file.write(b"bad")
    with pytest.raises(ValueError,match="SHA256"):
        read(tmp_path,head,layout)


def test_cross_group_full_attention_has_its_own_history_and_trainable_norms():
    graph = small_suffix(5)
    assert set(graph.histories) == {59,60,61,62,63}
    for layer in (59,63):
        assert graph.histories[layer]["history_key"].name == f"l{layer}_history_key"
        assert {p.bank for p in graph.parameters if p.layer == layer} == set(range(6))
        assert all(p.offset_norm for p in graph.parameters if p.layer == layer and p.bank in (3,4))
    assert sum(node.kind == "tv" for node in graph.nodes) == 1


def test_suffix_memory_rejects_insufficient_space_before_allocation():
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    from blockspec.pypto_qwen38.training_memory import suffix_memory,require_suffix_memory
    graph = small_suffix(5)
    layout = FlatLayout(graph.parameters)
    memory = suffix_memory(graph,layout,2)
    assert memory["learner"] > memory["serving"] > 0
    assert require_suffix_memory(graph,layout,2,memory["serving"],memory["learner"],reserve=0) == memory
    for role in ("serving","learner"):
        available = dict(memory);available[role] -= 1
        with pytest.raises(RuntimeError,match=f"last-5 {role}"):
            require_suffix_memory(graph,layout,2,available["serving"],available["learner"],reserve=0)


@pytest.mark.parametrize("n", (2,4,5,6,8,9,64))
def test_feedback_capture_follows_the_selected_layer_across_groups(n):
    from types import SimpleNamespace as NS
    from blockspec.pypto_qwen38.draft import NeuralDraft
    draft = object.__new__(NeuralDraft)
    feedback = NS(inputs={"hidden":"private_input"})
    serving = NS(prepare=lambda *args:None,select=lambda logits,start,count:list(range(count)))
    resident = NS(pages=NS(length=16,pages=(0,)),slot=2)
    draft.native = NS(requests={"r":resident},banks=tuple(range(8)),resident=tuple(range(4)),device_serving=serving)
    draft.learner = NS(options=NS(last_layers=n),graph=NS(capture_layer=64-n),begin=lambda *args,**kw:feedback)
    draft.weights = NS(for_group=lambda base:tuple(tuple(range(6 if (base+i)%4 == 3 else 9)) for i in range(4)))
    draft.group_size,draft.block_size,draft.rounds = 4,32,0
    draft.hidden,draft.group_scratch = ("hidden0","hidden1"),("scratch0","scratch1")
    for name in ("ids","cos","sin","query","key","value","gate","table","mask","logits"):
        setattr(draft,name,name)
    calls = []
    draft._run = lambda name,args:calls.append((name,args))
    draft.propose(NS(request_id="r",prompt_ids=[42]*16,generated=[7]),7,32)
    captures = [i for i,(name,_) in enumerate(calls) if name == "hidden_capture"]
    assert len(captures) == 1
    name,args = calls[captures[0]+1]
    assert args[0] == "private_input"
    assert (args[-4] if name == "group" else args[-3] if name == "full_layer" else args[-2]) == 64-n
    # 截断点之后仍走整组快路径，无需把整个末段拆成逐层执行。
    grouped = [args[-4] for name,args in calls if name == "group"]
    assert grouped == [base for base in range(0,64,4) if (64-n)%4 == 0 or base != (64-n)//4*4]
