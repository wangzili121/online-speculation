"""Check public W8A8 detection independently of the optional PyPTO environment."""

import json
from pathlib import Path
import runpy
import struct


SCRIPT = Path(__file__).parents[1] / "scripts/preflight_dsv41_pypto.py"
inspect_checkpoint = runpy.run_path(str(SCRIPT))["inspect_checkpoint"]


def checkpoint_fixture(root, *, public):
    config = dict(quantization_config={"quant_method": "ascend"}, text_config=dict(
        num_hidden_layers=1, hidden_size=4, q_lora_rank=2, num_attention_heads=2,
        head_dim=2, engram_layer_ids=[0]))
    if not public:
        config["quantization_config"]["model_quant_type"] = "W8A8_DYNAMIC"
    (root / "config.json").write_text(json.dumps(config))
    (root / "quant_model_description.json").write_text(json.dumps({"model_quant_type": "W8A8_DYNAMIC"}))
    specs = {}
    for name, rows, cols in (("wq_a", 2, 4), ("wq_b", 4, 2), ("wkv", 2, 4)):
        prefix = f"layers.0.attn.{name}"
        floating = public and name in ("wq_a", "wkv")
        specs[prefix + ".weight"] = dict(shape=[rows, cols], dtype="BF16" if floating else "I8")
        if not floating:
            for suffix in ("weight_scale", "weight_offset"):
                specs[prefix + "." + suffix] = dict(shape=[rows, 1], dtype="F32")
    specs["layers.0.engram.embed.weight"] = dict(shape=[2, 256], dtype="I8")
    specs["layers.0.engram.embed.scale"] = dict(shape=[2, 8], dtype="F8_E8M0" if public else "F32")
    for info in specs.values():
        info["data_offsets"] = [0, 1]  # The inspector reads headers, never tensor arithmetic.
    header = json.dumps(specs).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0")
    (root / "quant_model_weights.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(specs, "weights.safetensors")}))
    return root / "weights.safetensors"


def test_public_w8a8_reports_actual_formats_and_missing_affine_metadata(tmp_path):
    checkpoint_fixture(tmp_path, public=True)
    result = inspect_checkpoint(tmp_path)
    assert result["weight_format"] == "W8A8_DYNAMIC"
    assert not result["projection_and_engram_layout_matches"]
    problems = {item.get("tensor", item.get("field")): item for item in result["layout_issues"]}
    assert problems["layers.0.attn.wq_a.weight"]["actual"]["dtype"] == "BF16"
    assert problems["layers.0.attn.wq_a.weight_scale"]["actual"] == {}
    assert problems["layers.0.engram.embed.scale"]["actual"]["dtype"] == "F8_E8M0"


def test_expected_layout_passes_and_truncated_header_fails(tmp_path):
    path = checkpoint_fixture(tmp_path, public=False)
    assert inspect_checkpoint(tmp_path)["projection_and_engram_layout_matches"]
    path.write_bytes(b"short")
    try:
        inspect_checkpoint(tmp_path)
    except ValueError as error:
        assert "Truncated safetensors" in str(error)
    else:
        raise AssertionError("Truncated header accepted")
