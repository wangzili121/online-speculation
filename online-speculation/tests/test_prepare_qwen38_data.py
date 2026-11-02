import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_qwen38_data", Path(__file__).parents[1]
                                             / "scripts" / "prepare_qwen38_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and not add_generation_prompt
        return self.encode(messages[0]["content"], False)


def test_real_rows_are_not_padded_or_repeated(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "tokens.jsonl"
    source.write_text('\n'.join(json.dumps(row) for row in (
        {"text": "a"}, {"text": "abcdefgh"}, {"messages": [{"role": "user", "content": "ijklmnop"}]})))
    result = module.prepare(source, output, Tokenizer(), min_tokens=4, max_tokens=6, max_records=0)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["input_ids"] for row in rows] == [list(b"abcdef"), list(b"ijklmn")]
    assert [row["source_row"] for row in rows] == [1, 2]
    assert result["skipped_short"] == 1
    assert result["tokens_sha256"] == module.sha256(output)
    with pytest.raises(FileExistsError):
        module.prepare(source, output, Tokenizer(), min_tokens=4, max_tokens=6, max_records=1)


def test_no_valid_data_leaves_no_training_file(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "tokens.jsonl"
    source.write_text('{"text": "a"}\n')
    with pytest.raises(ValueError, match="no source rows"):
        module.prepare(source, output, Tokenizer(), min_tokens=4, max_tokens=6, max_records=0)
    assert not output.exists()


def test_record_limit_and_invalid_rows(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "tokens.jsonl"
    source.write_text('{"text": "abcd"}\n{"missing": "bad"}\n')
    report = module.prepare(source, output, Tokenizer(), min_tokens=4, max_tokens=8, max_records=1)
    assert len(report["records"]) == 1
    with pytest.raises(ValueError, match="text or messages"):
        module.tokenize({"missing": "bad"}, Tokenizer())
