"""JSON/XML output fields aligned to the original generated token sequence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from xml.parsers import expat

from inference_scaling.shared.output import OutputSegments, ThinkingParser, full_sequence_segments
from inference_scaling.shared.types import TokenSequence

FieldSpan = tuple[Any, int, int]
_DUPLICATE = object()


def _json_fields(text: str) -> dict[tuple[str, ...], FieldSpan]:
    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    decoder = json.JSONDecoder(parse_constant=invalid_constant)
    fields: dict[tuple[str, ...], FieldSpan] = {}

    def whitespace(position: int) -> int:
        while position < len(text) and text[position] in " \r\n\t":
            position += 1
        return position

    def value(position: int, path: tuple[str, ...]) -> tuple[Any, int]:
        if len(path) > 64:
            raise ValueError("JSON nesting exceeds 64 levels")
        start = whitespace(position)
        if start >= len(text):
            raise ValueError("incomplete JSON value")
        if text[start] == "{":
            result: dict[str, Any] = {}
            position = whitespace(start + 1)
            if position < len(text) and text[position] == "}":
                return result, position + 1
            while True:
                key, end = decoder.raw_decode(text, position)
                if not isinstance(key, str) or key in result:
                    raise ValueError("JSON keys must be distinct strings")
                position = whitespace(end)
                if position >= len(text) or text[position] != ":":
                    raise ValueError("missing JSON colon")
                result[key], position = value(position + 1, path + (key,))
                position = whitespace(position)
                if position < len(text) and text[position] == "}":
                    position += 1
                    break
                if position >= len(text) or text[position] != ",":
                    raise ValueError("missing JSON object delimiter")
                position = whitespace(position + 1)
            parsed: Any = result
        else:
            parsed, position = decoder.raw_decode(text, start)
        begin, end = (start + 1, position - 1) if isinstance(parsed, str) else (start, position)
        fields[path] = (parsed, begin, end)
        return parsed, position

    parsed, end = value(0, ())
    if not isinstance(parsed, dict) or whitespace(end) != len(text):
        raise ValueError("output must contain exactly one JSON object")
    return fields


def _xml_fields(text: str, *, fragment: bool = False) -> dict[tuple[str, ...], FieldSpan]:
    raw = text.encode("utf-8")
    parser = expat.ParserCreate()
    stack: list[tuple[str, int, list[str]]] = []
    fields: dict[tuple[str, ...], FieldSpan] = {}

    def start(name, attributes):
        if len(stack) >= 64:
            raise ValueError("XML nesting exceeds 64 levels")
        index = parser.CurrentByteIndex
        quoted = None
        while index < len(raw):
            char = raw[index]
            if quoted is not None:
                if char == quoted:
                    quoted = None
            elif char in (34, 39):
                quoted = char
            elif char == 62:
                break
            index += 1
        stack.append((name, index + 1, []))

    def data(value):
        for _, _, pieces in stack:
            pieces.append(value)

    def end(name):
        path = tuple(item[0] for item in stack)
        _, begin, pieces = stack.pop()
        stop = parser.CurrentByteIndex
        # An empty self-closing element ends at its opening-tag boundary.
        stop = max(begin, stop)
        fields[path] = (
            _DUPLICATE if path in fields else "".join(pieces),
            len(raw[:begin].decode("utf-8")), len(raw[:stop].decode("utf-8")),
        )

    def forbid_doctype(*args):
        raise ValueError("XML document types and external entities are unsupported")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = data
    parser.StartDoctypeDeclHandler = forbid_doctype
    parser.ExternalEntityRefHandler = forbid_doctype
    try:
        parser.Parse(raw, True)
    except expat.ExpatError as error:
        if not fragment and error.code == expat.errors.codes[expat.errors.XML_ERROR_JUNK_AFTER_DOC_ELEMENT]:
            prefix = "<_output_fragment>"
            parsed = _xml_fields(prefix + text + "</_output_fragment>", fragment=True)
            return {
                path[1:]: (value, start - len(prefix), end - len(prefix))
                for path, (value, start, end) in parsed.items() if len(path) > 1
            }
        raise
    return fields


def _field(
    fields: Mapping[tuple[str, ...], FieldSpan], path: tuple[str, ...] | None,
    names: tuple[str, ...], *, required: bool,
) -> FieldSpan | None:
    matches = [span for key, span in fields.items() if key and (key == path if path is not None else key[-1] in names)]
    if len(matches) > 1:
        raise ValueError("ambiguous structured output fields; specify a path")
    if matches and matches[0][0] is _DUPLICATE:
        raise ValueError("ambiguous repeated structured output field")
    if not matches and required:
        raise ValueError("missing thinking field")
    return matches[0] if matches else None


def _token_boundary(tokens: TokenSequence, text: str, position: int, decode: Callable[[TokenSequence], str]) -> int | None:
    """Validate an exact original-token boundary, without re-tokenizing the text."""
    wanted = text[:position]
    low, high = 0, len(tokens)
    while low <= high:
        middle = (low + high) // 2
        prefix = decode(tokens[:middle])
        if prefix == wanted:
            return middle
        if len(prefix) < position:
            low = middle + 1
        else:
            high = middle - 1
    # Byte-fallback Unicode tokens can temporarily decode to replacement chars.
    # Accept only equality; uncertain or intra-token boundaries use full scoring.
    for index in range(max(0, high - 8), min(len(tokens), low + 8) + 1):
        if decode(tokens[:index]) == wanted:
            return index
    return None


@dataclass(frozen=True, slots=True)
class StructuredThinkingParser:
    kind: str
    decode: Callable[[TokenSequence], str]
    thinking_path: tuple[str, ...] | None = None
    content_path: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.kind not in {"json", "xml"}:
            raise ValueError("structured output kind must be json or xml")

    def split(self, prompt, completion, *, eos_token_id=None) -> OutputSegments:
        tokens = tuple(completion)
        ended = eos_token_id is not None and eos_token_id in tokens
        if ended:
            tokens = tokens[:tokens.index(eos_token_id)]
        text = self.decode(tokens)

        def fallback(reason):
            return full_sequence_segments(completion, eos_token_id=eos_token_id, reason=reason)

        try:
            body, offset = text, 0
            if text.lstrip().startswith("```"):
                fenced = re.fullmatch(r"(\s*```(?:json|xml)?[ \t]*\r?\n)(.*?)(\r?\n```\s*)", text, re.DOTALL)
                if fenced is None:
                    raise ValueError("malformed structured output fence")
                body, offset = fenced.group(2), len(fenced.group(1))
            fields = _json_fields(body) if self.kind == "json" else _xml_fields(body)
            fields = {path: (value, start + offset, end + offset) for path, (value, start, end) in fields.items()}
            thinking = _field(fields, self.thinking_path, ("thinking", "reasoning", "analysis", "think"), required=True)
            content = _field(fields, self.content_path, ("answer", "content", "final"), required=self.content_path is not None)
        except (ValueError, expat.ExpatError, RecursionError):
            return fallback("malformed_" + self.kind)
        assert thinking is not None
        if not isinstance(thinking[0], str):
            return fallback("non_text_thinking_field")
        try:
            thinking[0].encode("utf-8")
            if content is not None and isinstance(content[0], str):
                content[0].encode("utf-8")
        except UnicodeError:
            return fallback("malformed_" + self.kind)
        if not thinking[0].strip():
            return fallback("empty")
        start = _token_boundary(tokens, text, thinking[1], self.decode)
        end = _token_boundary(tokens, text, thinking[2], self.decode)
        if start is None or end is None or end <= start:
            return fallback("unaligned_thinking_tokens")
        content_tokens: TokenSequence = ()
        content_text = ""
        if content is not None:
            content_start = _token_boundary(tokens, text, content[1], self.decode)
            content_end = _token_boundary(tokens, text, content[2], self.decode)
            if content_start is None or content_end is None:
                return fallback("unaligned_content_tokens")
            content_tokens = tokens[content_start:content_end]
            content_text = content[0] if isinstance(content[0], str) else json.dumps(content[0], ensure_ascii=False)
        return OutputSegments(
            tokens[start:end], content_tokens, start, end, None, "complete", ended,
            self.kind, thinking[0], content_text,
        )

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "thinking_path": self.thinking_path, "content_path": self.content_path,
                "requires_complete_output": True, "token_alignment": "original_token_boundaries"}


@dataclass(frozen=True, slots=True)
class UnifiedOutputParser:
    markers: ThinkingParser
    decode: Callable[[TokenSequence], str] | None = None
    kind: str = "auto"
    thinking_path: tuple[str, ...] | None = None
    content_path: tuple[str, ...] | None = None

    @property
    def formats(self):
        return self.markers.formats

    @property
    def mode(self):
        return self.markers.mode

    @property
    def supports_early_stop(self):
        return bool(self.formats) and self.kind in {"auto", "tags"}

    def prompt_mode(self, prompt):
        return self.markers.prompt_mode(prompt)

    def split(self, prompt, completion, *, eos_token_id=None):
        marker = self.markers.split(prompt, completion, eos_token_id=eos_token_id)
        if marker.status == "disabled" or self.decode is None or self.kind == "tags":
            return marker
        tokens = completion[:completion.index(eos_token_id)] if eos_token_id is not None and eos_token_id in completion else completion
        text = self.decode(tokens).lstrip()
        kind = self.kind
        if kind == "auto":
            if text.startswith(("{", "```json")):
                kind = "json"
            elif text.startswith("```xml") or (
                text.startswith("<") and marker.status in {"leading_content", "absent", "unrecognized_format"}
            ):
                kind = "xml"
            else:
                return marker
        return StructuredThinkingParser(kind, self.decode, self.thinking_path, self.content_path).split(
            prompt, completion, eos_token_id=eos_token_id,
        )

    def stop_boundary(self, prompt, completion, *, eos_token_id=None):
        if not self.supports_early_stop:
            return None
        if self.decode is not None and self.decode(completion).lstrip().startswith("{"):
            return None
        marker = self.markers.split(prompt, completion, eos_token_id=eos_token_id)
        return marker.boundary_end if marker.has_complete_thinking else None

    def describe(self) -> dict[str, object]:
        return {**self.markers.describe(), "format": self.kind, "thinking_path": self.thinking_path,
                "content_path": self.content_path, "structured_sampling": "full"}


__all__ = ["StructuredThinkingParser", "UnifiedOutputParser"]
