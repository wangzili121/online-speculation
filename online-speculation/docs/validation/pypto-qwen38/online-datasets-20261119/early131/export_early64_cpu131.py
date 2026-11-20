"""Export an existing last-1 checkpoint as TP2 initialization; never restore Adam."""
from pathlib import Path
import hashlib
import json
import sys
import shutil

BASE = Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105')
ROOT = BASE / 'oneserving-v109-b99e118'
sys.path.insert(0, str(ROOT / 'online-tp2-test/src-dense-v114'))
import torch
from safetensors.torch import load_file, save_file
from blockspec.pypto_qwen38.online_checkpoint import read, sha256

torch.set_num_threads(2)
source = BASE / 'online-study-20261111/tp1-colocated-step0-last1-64-gsm8k-r3.checkpoint'
initial = ROOT / 'step0-suffix-tp2'
output = ROOT / 'step64-last1-suffix-tp2-cpu131'
proof = ROOT / 'online-tp2-test/early64-export131.json'
meta = json.loads((source / 'manifest.json').read_text())
base = json.loads((initial / 'manifest.json').read_text())
assert meta['step'] == 64 and base['step'] == 0
assert base['initialization'] == 'AR-copy; no draft optimization'
assert meta['head']['contract'] == base['source_head_contract']
assert meta['head']['payload_sha256'] == '159646bde490fd63636557838b05440e676be6ab60420f4e3a753382a8c00def'
mapping = {'q_weight': ('q_proj.weight', 0), 'k_weight': ('k_proj.weight', 0),
           'v_weight': ('v_proj.weight', 0), 'o_weight': ('o_proj.weight', 1),
           'q_gamma': ('q_norm.weight', None), 'k_gamma': ('k_norm.weight', None)}
shapes = {'q_weight': (12288, 5120), 'k_weight': (1024, 5120),
          'v_weight': (1024, 5120), 'o_weight': (5120, 6144),
          'q_gamma': (1, 256), 'k_gamma': (1, 256)}
step, values = read(source, meta['head'], shapes)
expected = {k: values[k + '.master'].bfloat16() for k in mapping}
assert all(torch.isfinite(v).all() for v in values.values())
del values
assert not output.exists() and not proof.exists()
output.mkdir()
try:
    shards = []
    for rank, original in enumerate(base['rank_files']):
        path = initial / original['name']
        assert sha256(path) == original['sha256']
        tensors = load_file(str(path), device='cpu')
        changed = []
        for name, (raw_name, axis) in mapping.items():
            key = 'layers.63.attention.draft.' + raw_name
            value = expected[name] if axis is None else expected[name].chunk(2, dim=axis)[rank]
            value = value.reshape_as(tensors[key]).contiguous()
            assert value.dtype == tensors[key].dtype
            tensors[key] = value
            changed.append(key)
        target = output / original['name']
        save_file(tensors, str(target))
        del tensors
        actual = load_file(str(target), device='cpu')
        before = load_file(str(path), device='cpu')
        for key in actual:
            if key not in changed:
                assert torch.equal(actual[key], before[key]), key
        for name, (raw_name, axis) in mapping.items():
            raw = actual['layers.63.attention.draft.' + raw_name]
            reference = expected[name] if axis is None else expected[name].chunk(2, dim=axis)[rank]
            assert torch.equal(raw.reshape_as(reference), reference), name
            if axis is None:
                # Match make_publish(gamma=True): round raw norm, then add one in FP32.
                assert torch.equal(1 + raw.float().reshape_as(reference), 1 + reference.float())
        del actual, before
        shards.append(dict(name=target.name, sha256=sha256(target), tensors=33))
    manifest = dict(base)
    manifest.update(step=step, rank_files=shards, parent_payload_sha256=base['source_payload_sha256'],
                    source_payload_sha256=hashlib.sha256(''.join(s['sha256'] for s in shards).encode()).hexdigest(),
                    initialization='AR-copy plus 64 prior last-1 updates; fresh Adam required',
                    source_optimizer_checkpoint_sha256=meta['payload_sha256'],
                    source_initial_payload_sha256=meta['head']['payload_sha256'],
                    export_kind='last1_optimizer_weights_only', optimizer_reset_required=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = dict(completed=True, cpu_only=True, source_checkpoint=str(source),
                  output=str(output), source_script_sha256=sha256(Path(__file__)),
                  checkpoint_sha256=meta['payload_sha256'], manifest=manifest,
                  schema_and_payload_validated=True, frozen_layers_exact=True,
                  all_six_rendered_parameters_exact=True, optimizer_moments_imported=False,
                  device_replay_gate_still_required=True)
    proof.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
except BaseException:
    shutil.rmtree(output)
    raise
