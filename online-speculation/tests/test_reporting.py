"""Reports retain measurements while resource identities stay local."""

import ast
import hashlib
from io import StringIO
import json
from pathlib import Path, PureWindowsPath
import re

from blockspec.reporting import dump, dumps, public_report


def test_public_summary_filters_nested_identities_without_mutation(tmp_path):
    digest = hashlib.sha256(b"private artifact").hexdigest()
    path = PureWindowsPath("C:") / "Users" / "example" / "weights"
    payload = {
        "model_revision": digest, "implementation": digest,
        "config": {"model": str(path), "data": tmp_path, "output": "local/report.json",
                   "block_size": 32, "temperature": 1, "reference_manifest": "private.json"},
        "adapter": {"sha256": digest, "rank": 128, "base_model_name_or_path": "owner/model"},
        "records": [{"tps": 187.7, "tokens": 512, "committed_tokens": 510,
                     "token_ids": [1, 2], "source": {"directory": str(tmp_path)}}],
        "prompts": ["private question"], "device": "private hardware label",
        "numeric_gate_passed": True, "adapter_version_after_request": 4,
    }
    expected = {"config": {"block_size": 32, "temperature": 1}, "adapter": {"rank": 128},
                "records": [{"tps": 187.7, "tokens": 512, "committed_tokens": 510, "source": {}}],
                "numeric_gate_passed": True, "adapter_version_after_request": 4}
    assert public_report(payload) == expected
    assert json.loads(dumps(payload)) == expected
    stream = StringIO()
    dump(payload, stream)
    assert json.loads(stream.getvalue()) == expected
    assert payload["model_revision"] == digest and payload["records"][0]["token_ids"] == [1, 2]


def test_resource_values_are_filtered_even_under_generic_keys():
    path = str(Path("/") / "srv" / "private" / "weights")
    digest = hashlib.sha256(b"content").hexdigest()
    value = {"details": [path, "local/checkpoint.pt", digest, {path: 1, "tps": 20}],
             "other_file": "private-name", "reference": {"max_abs": 0.0},
             "sampling": {"top_k": 20, "top_p": .8}, "output_lengths": [256, 256],
             "checkpoint_base_dtype": "float32"}
    assert public_report(value) == {"details": [{"tps": 20}], "reference": {"max_abs": 0.0},
                                    "sampling": {"top_k": 20, "top_p": .8}, "output_lengths": [256, 256],
                                    "checkpoint_base_dtype": "float32"}


def test_report_serialization_has_one_publication_boundary():
    root = Path(__file__).resolve().parents[1]
    for path in list((root / "scripts").glob("*.py")) + [root / "src/blockspec/cli.py", *root.glob("src/blockspec/commands/*.py"),
                                                        *root.glob("ablation/scripts/*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "json"
                    and node.func.attr in ("dump", "dumps")):
                # These two writes construct synthetic training fixtures, not reports.
                assert path.name == "fit.py"
                assert isinstance(node.args[0], ast.Name) and node.args[0].id in ("public_config", "row")


def test_numeric_audit_exports_agreement_instead_of_token_sequences():
    assert public_report({"own_tokens": [1, 2], "reference_tokens": [1, 2], "ar_top_ids": [1],
                          "greedy_tokens": 2, "token_ids_identical": True, "pass": True}) == {
        "greedy_tokens": 2, "token_ids_identical": True, "pass": True}


def test_reference_manifest_details_stay_out_of_public_results():
    assert public_report({"reference": {"id": "owner/model", "entrypoint": {"class": "LocalModel"},
                                        "architectures": ["LocalModel"], "reference_transformers": "test-version"},
                          "errors": {"max_abs": 0}}) == {
        "reference": {"reference_transformers": "test-version"}, "errors": {"max_abs": 0}}


def test_versioned_sources_contain_no_literal_personal_paths_or_artifact_pins():
    root = Path(__file__).resolve().parents[1]
    files = [root / "README.md", *root.glob("docs/*.md"), *root.glob("scripts/*.py"),
             *root.glob("src/blockspec/**/*.py"), *root.glob("config/*.json"),
             *root.glob("ablation/**/*.py"), *root.glob("ablation/**/*.md")]
    personal_path = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/][^\s/\\]+|/(?:home|mnt/c/Users)/[^\s/]+)")
    literal_pin = re.compile(r"\b[0-9a-f]{32,64}\b", re.I)
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not personal_path.search(text), path.name
        assert not literal_pin.search(text), path.name
    assert not list((root / "references").glob("*.lock.json"))
