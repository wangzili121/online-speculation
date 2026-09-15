import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_SUFFIXES = (
    "attn_sink",
    "wq_a.weight",
    "wq_a.weight_scale",
    "wq_a.weight_offset",
    "q_norm.weight",
    "wq_b.weight",
    "wq_b.weight_scale",
    "wq_b.weight_offset",
    "wkv.weight",
    "wkv.weight_scale",
    "wkv.weight_offset",
    "kv_norm.weight",
    "wo_a.weight",
    "wo_a.weight_scale",
    "wo_a.weight_offset",
    "wo_b.weight",
    "wo_b.weight_scale",
    "wo_b.weight_offset",
)
C4_SUFFIXES = (
    "indexer.wq_b.weight",
    "indexer.wq_b.weight_scale",
    "indexer.wq_b.weight_offset",
    "indexer.weights_proj.weight",
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "ACK_EXCLUSIVE_NODE=1"),
        ({"ACK_EXCLUSIVE_NODE": "1", "NPU_DEVICES": "0,1,2,3"}, "full DSV4-Flash deployment"),
        ({"ACK_EXCLUSIVE_NODE": "1", "IMAGE_REF": "wrong:image"}, "pinned v0.26.0rc"),
        ({"ACK_EXCLUSIVE_NODE": "1", "ORTHRUS_DRAFT_SAMPLE_METHOD": "greedy"},
         "trained Chang-Bo deployment requires probabilistic"),
    ],
)
def test_offline_deployment_refuses_unsafe_configuration(overrides, message):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/deploy_dsv4_offline.sh")],
        env={**os.environ, "ORTHRUS_PROBABILISTIC_VALIDATED": "1", **overrides},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_container_creation_rejects_unapproved_image_digest(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = container ]; then exit 1; fi\n"
        "if [ \"$1\" = image ] && [ \"$3\" = --format ]; then\n"
        "  echo quay.io/ascend/vllm-ascend@sha256:bad\n"
        "fi\n",
        encoding="ascii",
    )
    docker.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/create_dsv4_container.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CONTAINER": "digest-test",
            "REPO_HOST": str(ROOT),
            "MODELS_HOST": str(models),
            "IMAGE_PULL": "never",
            "EXPECTED_IMAGE_DIGEST": "sha256:approved",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "image digest mismatch" in result.stderr


def _fake_target(path: Path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    path.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 16,
        "num_attention_heads": 64,
        "vocab_size": 128,
        "compress_ratios": [0, 4, 128] * 14 + [0],
        "sliding_window": 128,
        "dspark_noise_token_id": 127,
    }
    _write_json(path / "config.json", config)

    tensors = {}
    quant = {}
    for layer, ratio in enumerate(config["compress_ratios"]):
        suffixes = BASE_SUFFIXES + (C4_SUFFIXES if ratio == 4 else ())
        for suffix in suffixes:
            name = f"model.layers.{layer}.self_attn.{suffix}"
            is_int8 = suffix.endswith(".weight") and not suffix.endswith(
                ("q_norm.weight", "kv_norm.weight", "weights_proj.weight")
            )
            tensors[name] = torch.ones(
                2, dtype=torch.int8 if is_int8 else torch.float32
            )
            if is_int8:
                quant[name] = "W8A8_DYNAMIC"
        compressor = f"model.layers.{layer}.self_attn.compressor.wkv.weight"
        tensors[compressor] = torch.ones(2)
        quant[compressor] = "FLOAT"
    shard = "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    _write_json(
        path / "quant_model_weights.safetensors.index.json",
        {"metadata": {}, "weight_map": {name: shard for name in tensors}},
    )
    _write_json(path / "quant_model_description.json", quant)


def test_export_and_preflight_dsv4_head(tmp_path):
    model = tmp_path / "model"
    head = tmp_path / "head"
    _fake_target(model)

    exported = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_dsv4_orthrus_head.py"),
            str(model),
            str(head),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated 43 layers" in exported.stdout

    config = json.loads((head / "config.json").read_text())
    assert config["architectures"] == ["OrthrusDSV4DraftModel"]
    assert config["compress_ratios"] == [0, 4, 128] * 14 + [0]
    assert config["orthrus_block_size"] == 32
    assert config["pard_token"] == config["mask_token_id"]
    assert config["sample_from_anchor"] is False

    index = json.loads(
        (head / "quant_model_weights.safetensors.index.json").read_text()
    )
    assert not any("compressor" in name for name in index["weight_map"])
    assert "layers.1.attn.indexer.wq_b.weight" in index["weight_map"]
    quant = json.loads((head / "quant_model_description.json").read_text())
    assert "orthrus.layers.0.self_attn.wq_a.weight" in quant
    assert not any("compressor" in name for name in quant)

    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight_dsv4_orthrus.py"),
            "--model",
            str(model),
            "--head",
            str(head),
            "--tensor-parallel-size",
            "8",
            "--production",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(checked.stdout)
    assert result["status"] == "ok"
    assert result["compress_ratios"] == {"c128": 14, "c4": 14, "swa": 15}

    trained_gate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight_dsv4_orthrus.py"),
            "--model",
            str(model),
            "--head",
            str(head),
            "--require-trained",
        ],
        capture_output=True,
        text=True,
    )
    assert trained_gate.returncode != 0
    assert "trained diffusion head" in trained_gate.stderr

    shard = head / next(iter(index["weight_map"].values()))
    shard.write_bytes(shard.read_bytes()[:-1])
    corrupted = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight_dsv4_orthrus.py"),
            "--model",
            str(model),
            "--head",
            str(head),
        ],
        capture_output=True,
        text=True,
    )
    assert corrupted.returncode != 0
    assert "truncated or oversized safetensors shard" in corrupted.stderr


def test_dsv4_plugin_registers_dedicated_architecture(monkeypatch):
    calls = []
    fake_vllm = SimpleNamespace(
        ModelRegistry=SimpleNamespace(
            register_model=lambda architecture, implementation: calls.append(
                (architecture, implementation)
            )
        )
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    from blockspec.vllm_ascend import plugin

    plugin.register()
    assert [architecture for architecture, _ in calls] == [
        "OrthrusLM",
        "OrthrusDSV4DraftModel",
    ]


def _load_runtime_patcher():
    path = ROOT / "runtime/vllm_ascend_v0_26/patch_runtime.py"
    spec = importlib.util.spec_from_file_location("orthrus_patch_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def test_v026_overlay_contains_dedicated_method_and_guards():
    module, path = _load_runtime_patcher()
    source = path.read_text()
    assert 'OrthrusModelTypes = Literal["orthrus"]' in source
    assert "AscendOrthrusProposer" in source
    assert "bind_shared_backbone(target_language_model)" in source
    assert 'self.speculative_config.method != \\"orthrus\\"' in source
    assert '"architectures": ["OrthrusDSV4DraftModel"]' in source
    assert 'self.method == \\"orthrus\\" and sampling_metadata is not None' in source
    assert 'self._get_spec_decode_draft_probs(' in source
    assert "validate_vllm_runtime(roots[0])" in source
    assert callable(module.validate_vllm_runtime)


def test_v026_vllm_overlay_is_guarded_and_idempotent(tmp_path):
    module, _ = _load_runtime_patcher()
    root = tmp_path / "vllm-root"
    path = root / "vllm/config/speculative.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        'DSparkModelTypes = Literal["dspark"]\n'
        "SpeculativeMethod = Union[\n"
        "    DSparkModelTypes,\n"
        "]\nRejectionSampleMethod = Literal['standard']\n"
        '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):\n'
        '                if self.method in ("dflash", "dspark"):\n'
        '            self.method == "draft_model"\n'
        '        return self.method == "draft_model"\n'
        '                if self.method == "medusa":\n'
        '                    draft_hf_overrides = {"model_type": "medusa"}\n'
        '                else:\n',
        encoding="utf-8",
    )
    assert module.patch_vllm(root, dry_run=True) == ["would apply"] * 7
    assert module.patch_vllm(root, dry_run=False) == ["applied"] * 7
    assert module.patch_vllm(root, dry_run=False) == ["already applied"] * 7
    patched = path.read_text(encoding="utf-8")
    assert 'OrthrusModelTypes = Literal["orthrus"]' in patched
    assert '"architectures": ["OrthrusDSV4DraftModel"]' in patched
    assert 'self.method in ("draft_model", "orthrus")' in patched


def test_dsv4_target_compressors_are_not_registered_in_draft():
    source = (ROOT / "src/blockspec/vllm_ascend/dsv4.py").read_text()
    assert 'object.__setattr__(draft_impl, "compressor"' not in source
    assert "draft_impl.compressor = target_attention.compressor" not in source
    assert '"compressor_wkv"' in source
    assert "_query_readonly(" in source
    assert 'if ".compressor." in name' in source


def test_dsv4_orthrus_rejects_unimplemented_context_parallel():
    source = (ROOT / "src/blockspec/vllm_ascend/dsv4.py").read_text()
    assert 'additional_config.get("enable_dsa_cp")' in source
    assert 'additional_config.get("enable_flashcomm1")' in source
    for script in (
        "scripts/serve_dsv4_target.sh",
        "scripts/serve_dsv4_orthrus.sh",
        "scripts/serve_dsv4_dspark.sh",
    ):
        launch = (ROOT / script).read_text()
        default = launch.split("DEFAULT_ADDITIONAL_CONFIG=", 1)[1].splitlines()[0]
        assert "enable_dsa_cp" not in default
        assert "enable_flashcomm1" not in default


def test_dsv4_orthrus_launcher_uses_lossless_sampling_defaults():
    launch = (ROOT / "scripts/serve_dsv4_orthrus.sh").read_text()
    assert 'ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy' in launch
    assert '"rejection_sample_method": "standard"' in launch
    assert '"parallel_drafting": True' in launch
    smoke = (ROOT / "scripts/smoke_construct_dsv4_orthrus.py").read_text()
    assert '"draft_sample_method": "greedy"' in smoke
    target = (ROOT / "scripts/serve_dsv4_target.sh").read_text()
    dspark = (ROOT / "scripts/serve_dsv4_dspark.sh").read_text()
    assert 'TARGET_EXECUTION_MODE:-eager' in target
    assert 'DSPARK_EXECUTION_MODE:-eager' in dspark
    assert "ORTHRUS_TORCH_PROFILE_DIR" in launch
    assert "--profiler-config" in launch


def test_dsv4_container_supports_disconnected_image_startup():
    create = (ROOT / "scripts/create_dsv4_container.sh").read_text()
    assert 'IMAGE_PULL="${IMAGE_PULL:-auto}"' in create
    assert 'docker image inspect "$IMAGE_REF"' in create
    assert "image is not available locally and IMAGE_PULL=never" in create


def test_dsv4_healthcheck_covers_model_and_decode():
    health = (ROOT / "scripts/healthcheck_dsv4.py").read_text()
    assert 'base + "/health"' in health
    assert 'base + "/v1/models"' in health
    assert 'base + "/v1/completions"' in health


def test_dsv4_benchmark_parses_and_attaches_spec_metrics(tmp_path):
    path = ROOT / "scripts/benchmark_dsv4_matrix.py"
    spec = importlib.util.spec_from_file_location("dsv4_benchmark_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parsed = module._parse_prometheus(
        '# HELP ignored\n'
        'vllm:spec_decode_num_drafts_total{engine="0"} 3\n'
        'vllm:spec_decode_num_drafts_total{engine="1"} 2\n'
        'vllm:spec_decode_num_draft_tokens_total 31\n'
    )
    assert parsed["vllm:spec_decode_num_drafts_total"] == 5
    result = tmp_path / "result.json"
    _write_json(result, {"request_goodput": 1.0})
    before = {name: 10.0 for name in module.SPEC_COUNTERS}
    after = {
        "spec_decode_num_drafts": 12.0,
        "spec_decode_draft_tokens": 72.0,
        "spec_decode_accepted_tokens": 18.0,
    }
    module._attach_spec_delta(result, before, after)
    payload = json.loads(result.read_text())
    assert payload["spec_decode_num_drafts"] == 2
    assert payload["spec_decode_draft_tokens"] == 62
    assert payload["spec_decode_accepted_tokens"] == 8
    assert payload["spec_decode_acceptance_length"] == 5


def test_quantization_wrapper_preflights_bf16_exchange():
    wrapper = (ROOT / "scripts/quantize_dsv4_orthrus_head.sh").read_text()
    preflight = wrapper.index("preflight_dsv4_orthrus.py")
    adapter = wrapper.index('\n"$ORTHRUS_MODELSLIM_ADAPTER" ', preflight)
    assert preflight < adapter
    assert "--allow-bf16" in wrapper
    finalizer = (ROOT / "scripts/finalize_dsv4_orthrus_w8a8.py").read_text()
    assert 'mask_token_id = int(head_config.get("mask_token_id", -1))' in finalizer
    assert '"pard_token": mask_token_id' in finalizer


def test_runtime_lock_prefers_stable_driver_version_file():
    capture = (ROOT / "scripts/capture_dsv4_runtime_lock.py").read_text()
    apply = (ROOT / "runtime/vllm_ascend_v0_26/apply.sh").read_text()
    for source in (capture, apply):
        driver = source.index('"driver": first_existing(')
        npu_smi = source.index('command("npu-smi", "info", "-t", "board"', driver)
        assert driver < npu_smi
