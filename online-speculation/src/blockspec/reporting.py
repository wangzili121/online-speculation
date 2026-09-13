"""Publication summaries, separate from private checkpoint and input metadata."""

import json
from pathlib import PurePath
import re


_OMIT = object()
_IDENTIFIER = re.compile(r"(?:^|_)(?:sha\d*|hash|fingerprints?|digest|revision|commit|hostname|username)(?:_|$)", re.I)
_HEX = re.compile(r"(?<![0-9a-z])[0-9a-f]{32,64}(?![0-9a-z])", re.I)
_LOCATION = re.compile(r"(?:[a-z]:[\\/]|\\\\|(?<!\w)/(?:[\w.-]+/)+|https?://|file://)", re.I)
_FILE = re.compile(r"\.(?:safetensors|pt|pth|bin|jsonl?|ya?ml|toml)(?:\b|$)", re.I)
_RESOURCE_KEY = re.compile(r"(?:^|_)(?:model|path|directory|file|filename|location|url|repo|checkpoint)(?:_|$)", re.I)
_PRIVATE_FIELDS = {
    "model_id", "model_name", "model_name_or_path", "base_model_name_or_path",
    "dataset", "device_name", "gpu", "host", "user", "email", "repository", "id", "entrypoint", "architectures",
    "prompt_text", "prompt_texts", "learning_prompts", "input_ids", "token_ids",
    "output_ids", "generated_text", "base_sample", "text", "messages", "conversation",
}
_LOCATION_FIELDS = {
    "model", "base", "adapter", "data", "train_data", "validation", "validation_data",
    "output", "summary", "head", "compare_head", "audit_reference", "checkpoint",
    "initial_adapter", "source", "checkout", "cache", "file", "filename", "directory",
    "path", "reference_manifest", "reference_source", "reference_class",
}


def public_report(value):
    """Retain measurements and method settings; omit resource identities and text.

    Apply only at report boundaries. Local datasets and resumable checkpoints
    keep the integrity information used by validation and experiment isolation.
    """
    def clean(item, key=""):
        if _IDENTIFIER.search(key) or key in _PRIVATE_FIELDS:
            return _OMIT
        if isinstance(item, PurePath):
            return _OMIT
        if isinstance(item, dict):
            result = {}
            for name, child in item.items():
                name = str(name)
                if _LOCATION.search(name) or _HEX.search(name):
                    continue
                filtered = clean(child, name.lower())
                if filtered is not _OMIT:
                    result[name] = filtered
            return result
        if isinstance(item, (list, tuple)):
            if key in ("prompts", "tokens", "outputs", "path") or key.endswith(("_tokens", "_ids")):
                return _OMIT
            return [filtered for child in item if (filtered := clean(child)) is not _OMIT]
        if isinstance(item, str):
            resource_key = _RESOURCE_KEY.search(key) and not key.endswith(("_dtype", "_precision"))
            if (key in _LOCATION_FIELDS or key == "device" or resource_key
                    or _LOCATION.search(item) or _FILE.search(item) or _HEX.search(item)):
                return _OMIT
        return item

    result = clean(value)
    return None if result is _OMIT else result


def dumps(value, **kwargs):
    return json.dumps(public_report(value), **kwargs)


def dump(value, stream, **kwargs):
    return json.dump(public_report(value), stream, **kwargs)
