import hashlib
import json

from blockspec.vllm_ascend.artifacts import sha256_file, write_manifest


def test_sha256_file_and_exclusive_manifest(tmp_path):
    source = tmp_path / "workload.jsonl"
    source.write_bytes(b"one\ntwo\n")
    assert sha256_file(source) == hashlib.sha256(b"one\ntwo\n").hexdigest()

    output = tmp_path / "manifest.json"
    write_manifest(output, {"source": str(source), "value": 3})
    assert json.loads(output.read_text()) == {
        "source": str(source), "value": 3}

    try:
        write_manifest(output, {})
    except FileExistsError:
        pass
    else:
        raise AssertionError("manifest writes must not overwrite a trial")
