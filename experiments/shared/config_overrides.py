"""Validated dotted-key overrides shared by experiment entry points."""

from __future__ import annotations

from copy import deepcopy
import tomllib
from typing import Any, Mapping, Sequence


def parse_override_value(text: str) -> Any:
    """Parse one CLI value with TOML scalar/list syntax."""

    if text.strip().lower() in {"none", "null"}:
        return None
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def apply_config_overrides(
    config: Mapping[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """Apply ``section.key=value`` entries without permitting misspelled keys."""

    result = deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"config override must contain '=': {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        parts = tuple(part.strip() for part in dotted_key.split("."))
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid config override key: {dotted_key!r}")
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            value = cursor.get(part)
            if not isinstance(value, dict):
                raise KeyError(f"unknown config override path: {dotted_key!r}")
            cursor = value
        leaf = parts[-1]
        if leaf not in cursor:
            raise KeyError(f"unknown config override key: {dotted_key!r}")
        cursor[leaf] = parse_override_value(raw_value.strip())
    return result


__all__ = ["apply_config_overrides", "parse_override_value"]

