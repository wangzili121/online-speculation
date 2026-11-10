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
    for n in (2,3,4):
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
