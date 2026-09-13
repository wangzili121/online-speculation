from __future__ import annotations

import pytest

from inference_scaling.shared.output import ThinkingFormat, find_token_sequence
from inference_scaling.shared.structured_output import UnifiedOutputParser
from inference_scaling.shared.consilience import confidence_windows
from inference_scaling.arllm.output import thinking_format_from_backend
from inference_scaling.arllm.reward_factory import model_reward_from_config, reward_temperature_from_config
from inference_scaling.arllm.config import SamplingConfig


@pytest.mark.parametrize(
    ("prompt", "completion", "status", "thinking", "content"),
    [
        ((), (90, 1, 2, 91, 92, 3), "complete", (1, 2), (3,)),
        ((8, 90, 7), (1, 2, 91, 92, 3), "complete", (1, 2), (3,)),
        ((90, 1, 91, 92), (2, 3), "absent", (), (2, 3)),
        ((), (1, 2), "absent", (), (1, 2)),
        ((), (90, 91, 92, 3), "empty", (), (3,)),
        ((), (90, 1, 91), "incomplete", (1, 91), ()),
        ((90,), (1, 2), "incomplete", (1, 2), ()),
        ((90,), (1, 99, 91, 92, 3), "incomplete", (1,), ()),
    ],
)
def test_thinking_content_spans(prompt, completion, status, thinking, content):
    result = ThinkingFormat((91, 92), (90,)).split(prompt, completion, eos_token_id=99)
    assert result.status == status
    assert result.thinking_token_ids == thinking
    assert result.content_token_ids == content


def test_end_delimited_format_and_explicit_prompt_phase():
    result = ThinkingFormat((91, 92)).split((), (1, 2, 91, 92, 3, 99, 99), eos_token_id=99)
    assert result.thinking_token_ids == (1, 2)
    assert result.content_token_ids == (3,)
    assert result.boundary_end == 4
    assert result.ended_by_eos
    format_ = ThinkingFormat((91,), (90,), starts_in_thinking=False)
    assert format_.split((90,), (1, 2)).status == "absent"


def test_marker_search_uses_whole_multitoken_sequence():
    assert find_token_sequence((1, 2, 1, 3), (1, 3)) == 2
    assert find_token_sequence((1, 2, 1), (1, 3)) is None
    with pytest.raises(ValueError):
        find_token_sequence((1,), ())


def test_confidence_windows_short_sequences_and_large_fixed_windows():
    one = confidence_windows((2.0,))
    assert (one.skipped, one.window, one.score) == (0, 1, -4.0)
    larger = confidence_windows((1.0, 2.0, 4.0), window_tokens=2048, skip_fraction=0.5)
    assert (larger.skipped, larger.window, larger.initial, larger.final) == (1, 2, 3.0, 3.0)


class _Tokenizer:
    eos_token_id = 99

    def get_vocab(self):
        return {"<think>": 90, "</think>": 91}


class _Backend:
    tokenizer = _Tokenizer()
    model_id = "any-user-model"

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return (self.tokenizer.get_vocab()[text],)


def test_format_resolution_uses_tokenizer_not_model_name():
    format_ = thinking_format_from_backend(_Backend(), required=True)
    assert format_.formats == (ThinkingFormat((91,), (90,), name="think"),)


def test_factory_separates_reward_probability_and_proposal_temperature():
    reward = model_reward_from_config(
        _Backend(), {"conditional_is": {"reward_temperature": 0.1}},
        source="consilience", sampling=SamplingConfig(temperature=0.6),
    )
    assert reward.scope == "thinking"
    assert reward.sampling.temperature == 1
    assert reward_temperature_from_config(
        {"conditional_is": {"reward_temperature": 0.1}}, source="consilience"
    ) == 2
    assert reward_temperature_from_config({"reward": {"temperature": 4}}, source="consilience") == 4


class _TextTokenizer:
    eos_token_id = 0

    def __init__(self, template):
        self.chat_template = template

    def get_vocab(self):
        return {}

    def encode(self, text, **kwargs):
        return tuple(ord(char) for char in text)

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def _text_parser(template="<think></think>", options=None):
    from types import SimpleNamespace

    tokenizer = _TextTokenizer(template)
    backend = SimpleNamespace(tokenizer=tokenizer, encode=tokenizer.encode, decode=tokenizer.decode)
    return thinking_format_from_backend(backend, options), tokenizer


@pytest.mark.parametrize("opening,closing", (
    ("<think>", "</think>"), ("<thinking>", "</thinking>"),
    ("[THINK]", "[/THINK]"), ("<reasoning>", "</reasoning>"),
))
def test_multiple_token_formats_and_prefilled_opening(opening, closing):
    parser, tokenizer = _text_parser(opening + closing)
    for prompt, output in (("", opening + "work" + closing + "result"), (opening + "\n", "work" + closing + "result")):
        segments = parser.split(tokenizer.encode(prompt), tokenizer.encode(output))
        assert segments.status == "complete"
        assert tokenizer.decode(segments.thinking_token_ids) == "work"
        assert tokenizer.decode(segments.content_token_ids) == "result"


@pytest.mark.parametrize("text,status", (
    ("result", "absent"), ("<think>unfinished", "incomplete"),
    ("<think></think>result", "empty"), ("<think>\n \t</think>result", "empty"),
    ("<think>a<think>b</think>result", "malformed"),
    ("ordinary text <think>work</think>result", "leading_content"),
))
def test_failed_segmentation_keeps_full_output(text, status):
    parser, tokenizer = _text_parser()
    result = parser.split((), tokenizer.encode(text))
    assert result.status == status
    assert result.thinking_token_ids == ()
    assert tokenizer.decode(result.content_token_ids) == text
    assert result.describe()["segmentation_fallback_reason"] == status


def test_unknown_and_disabled_modes_and_empty_prompt_prefill():
    parser, tokenizer = _text_parser("plain template")
    assert parser.split((), tokenizer.encode("result")).status == "unrecognized_format"
    parser, tokenizer = _text_parser(options={"thinking_mode": "disabled"})
    assert parser.split((), tokenizer.encode("<think>work</think>result")).status == "disabled"
    parser, tokenizer = _text_parser()
    result = parser.split(tokenizer.encode("<think>\n\n</think>\n"), tokenizer.encode("result"))
    assert result.status == "disabled"
    assert result.describe()["detected_output_mode"] == "content"
    # A quoted tag in an earlier message does not prefill the assistant's thinking.
    quoted_prompt = tokenizer.encode("user: '<think>'\nassistant:")
    assert parser.split(quoted_prompt, tokenizer.encode("result")).status == "absent"


def test_configured_formats_and_template_thinking_switch():
    from inference_scaling.arllm.output import output_settings_from_config

    settings = {"thinking_formats": [
        {"name": "custom_phase", "start_text": "BEGIN:", "end_text": "FINAL:"},
        {"name": "bracket", "start_text": "[THINK]", "end_text": "[/THINK]"},
    ]}
    parser, tokenizer = _text_parser("", settings)
    assert isinstance(parser, UnifiedOutputParser)
    item = parser.split((), tokenizer.encode("BEGIN:workFINAL:result"))
    assert item.format_name == "custom_phase"
    assert tokenizer.decode(item.content_token_ids) == "result"
    assert output_settings_from_config({"prompt": {"chat_template_kwargs": {"enable_thinking": False}}})[
        "thinking_mode"
    ] == "disabled"


def test_boundary_decision_is_stable_after_final_content_arrives():
    parser, tokenizer = _text_parser("<think></think>[THINK][/THINK]")
    prefix = tokenizer.encode("<think>work</think>")
    before = parser.split((), prefix)
    for tail in ("result", "[THINK]quoted[/THINK]", "<think>literal"):
        after = parser.split((), prefix + tokenizer.encode(tail))
        assert after.status == "complete"
        assert after.boundary_end == before.boundary_end
        assert after.thinking_token_ids == before.thinking_token_ids


@pytest.mark.parametrize("text,kind,thinking,content", (
    ('{"thinking":"work","answer":"7"}', "json", "work", "7"),
    ('{"analysis":"路径\\n\\\"检查\\\"","final":7}', "json", '路径\n"检查"', "7"),
    ('```json\n{"reasoning":"work","content":"7"}\n```', "json", "work", "7"),
    ('<response><thinking>work</thinking><answer>7</answer></response>', "xml", "work", "7"),
    ('<response><thinking>a &amp; b</thinking><answer>7</answer></response>', "xml", "a & b", "7"),
    ('<response><thinking><step>a</step><step>b</step></thinking><answer>7</answer></response>', "xml", "ab", "7"),
    ('```xml\n<response><analysis>work</analysis><content>7</content></response>\n```', "xml", "work", "7"),
))
def test_structured_output_fields_preserve_text_and_original_token_positions(text, kind, thinking, content):
    parser, tokenizer = _text_parser()
    tokens = tokenizer.encode(text)
    item = parser.split((), tokens)
    assert item.status == "complete"
    assert item.format_name == kind
    assert item.thinking_text == thinking
    assert item.content_text == content
    assert item.thinking_token_ids == tokens[item.thinking_start:item.thinking_end]
    assert item.boundary_end is None  # full-object parsing has no early-stop boundary
    assert parser.stop_boundary((), tokens) is None


def test_explicit_nested_json_and_xml_paths():
    parser, tokenizer = _text_parser("", {
        "thinking_format": "json", "thinking_path": ["result", "trace"], "content_path": "result.value",
    })
    assert parser.split((), tokenizer.encode('{"result":{"trace":"work","value":"7"}}')).thinking_text == "work"
    parser, tokenizer = _text_parser("", {
        "thinking_format": "xml", "thinking_path": "response.trace", "content_path": ["response", "value"],
    })
    assert parser.split((), tokenizer.encode('<response><trace>work</trace><value>7</value></response>')).content_text == "7"
    parser, tokenizer = _text_parser("", {"thinking_format": "xml"})
    fragment = parser.split((), tokenizer.encode('<thinking>work</thinking><answer>7</answer>'))
    assert fragment.thinking_text == "work"
    assert fragment.content_text == "7"


@pytest.mark.parametrize("text,reason", (
    ('{"thinking":"cut', "malformed_json"),
    ('{"thinking":"a","thinking":"b"}', "malformed_json"),
    ('{"thinking":"a","analysis":"b"}', "malformed_json"),
    ('{"thinking":"a","answer":NaN}', "malformed_json"),
    ('{"thinking":false,"answer":"7"}', "non_text_thinking_field"),
    ('<response><thinking>a</response>', "malformed_xml"),
    ('<response><thinking>a</thinking><thinking>b</thinking></response>', "malformed_xml"),
    ('<!DOCTYPE response [<!ENTITY secret SYSTEM "file:///example">]><response><thinking>&secret;</thinking></response>', "malformed_xml"),
))
def test_malformed_or_ambiguous_structured_output_falls_back_to_full(text, reason):
    parser, tokenizer = _text_parser()
    item = parser.split((), tokenizer.encode(text))
    assert item.status == reason
    assert item.content_token_ids == tokenizer.encode(text)
    assert item.thinking_token_ids == ()


def test_structured_reward_never_retokenizes_a_field_or_splits_a_token():
    from types import SimpleNamespace
    from inference_scaling.arllm.rewards import ConsilienceReward

    chunks = {1: '{"thinking":"wo', 2: 'rk","answer":"7"}'}
    tokenizer = SimpleNamespace(eos_token_id=99, get_vocab=lambda: {},
        decode=lambda tokens, **kwargs: "".join(chunks[token] for token in tokens))
    scored = []

    def statistics(requests, **kwargs):
        scored.extend(requests)
        return [SimpleNamespace(token_topk_confidences=(1.0, 2.0))]

    backend = SimpleNamespace(tokenizer=tokenizer, score_statistics_batch=statistics)
    reward = ConsilienceReward(backend)
    assert reward((), (1, 2)) == -1
    assert scored[0].continuations == ((1, 2),)
    assert reward.describe_completion((), (1, 2))["reward_fallback_reason"] == "unaligned_thinking_tokens"
