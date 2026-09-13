"""Configurable, model-independent verifier rewards.

Inference algorithms consume token-level reward callables.  This module keeps
the source of that reward separate: a verifier scores decoded prompt/completion
pairs and a small adapter exposes the token-level interface used by AR and dLLM
algorithms.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class VerifierInput:
    """One decoded input to a verifier."""

    prompt: str
    completion: str


@dataclass(frozen=True, slots=True)
class VerifierContext:
    """Information available when one verifier instance is constructed.

    ``reference`` is optional.  A deployed verifier normally needs only the
    prompt and its own configuration; reference-based benchmark diagnostics
    must opt in explicitly through ``requires_reference``.
    """

    prompt: str
    reference: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class TextVerifier(Protocol):
    """Pointwise reward function over decoded prompt/completion text."""

    def score(self, prompt: str, completion: str) -> float:
        """Return ``r(prompt, completion)``."""


class BatchTextVerifier(TextVerifier, Protocol):
    """Optional batched extension implemented by remote or model verifiers."""

    def score_batch(self, inputs: Sequence[VerifierInput]) -> Sequence[float]:
        """Score a batch without changing pointwise reward semantics."""


VerifierCallable = Callable[[str, str], float]
VerifierBatchCallable = Callable[[Sequence[VerifierInput]], Sequence[float]]
TokenReward = Callable[[TokenSequence, TokenSequence], float]
TokenBatchReward = Callable[
    [TokenSequence, Sequence[TokenSequence]], Sequence[float]
]


@dataclass(frozen=True, slots=True)
class VerifierSpec:
    """Validated configuration needed to construct one verifier.

    Providers:

    - ``python`` loads a trusted factory named by ``factory``.  The factory is
      called as ``factory(context=context, **options)`` and returns either a
      callable ``(prompt, completion) -> reward`` or an object with ``score``.
    - ``constant`` returns the configured scalar ``value``.  It is useful for
      integration tests and reward-free controls.
    """

    provider: str
    name: str
    factory: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    requires_reference: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "VerifierSpec":
        allowed = {
            "provider",
            "name",
            "factory",
            "options",
            "requires_reference",
            "value",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown verifier configuration fields: {unknown}")
        provider_value = values.get("provider", "python")
        if not isinstance(provider_value, str) or not provider_value:
            raise TypeError("verifier.provider must be a non-empty string")
        provider = provider_value
        if provider not in {"python", "constant"}:
            raise ValueError(f"unsupported verifier provider {provider!r}")
        factory_value = values.get("factory")
        if factory_value is not None and (
            not isinstance(factory_value, str) or not factory_value
        ):
            raise TypeError("verifier.factory must be a non-empty string")
        factory: str | None = factory_value
        if provider == "python" and not factory:
            raise ValueError("a Python verifier requires verifier.factory")
        if provider == "constant" and factory is not None:
            raise ValueError("a constant verifier cannot define verifier.factory")
        raw_options = values.get("options", {})
        if not isinstance(raw_options, Mapping):
            raise TypeError("verifier.options must be a table")
        options = dict(cast(Mapping[str, object], raw_options))
        if "context" in options:
            raise ValueError("verifier.options.context is reserved by the runtime")
        if "value" in values:
            if provider != "constant":
                raise ValueError("verifier.value is valid only for provider='constant'")
            options["value"] = values["value"]
        if provider == "constant" and "value" not in options:
            raise ValueError("a constant verifier requires verifier.value")
        name_value = values.get("name", factory or provider)
        if not isinstance(name_value, str) or not name_value:
            raise TypeError("verifier.name must be a non-empty string")
        requires_reference = values.get("requires_reference", False)
        if not isinstance(requires_reference, bool):
            raise TypeError("verifier.requires_reference must be a boolean")
        return cls(
            provider=provider,
            name=name_value,
            factory=factory,
            options=options,
            requires_reference=requires_reference,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable description for manifests."""

        return {
            "provider": self.provider,
            "name": self.name,
            "factory": self.factory,
            "options": dict(self.options),
            "requires_reference": self.requires_reference,
        }


def verifier_spec_from_config(config: Mapping[str, object]) -> VerifierSpec:
    """Read the required top-level ``[verifier]`` configuration table."""

    values = config.get("verifier")
    if not isinstance(values, Mapping):
        raise ValueError("a verifier-assisted method requires a [verifier] table")
    return VerifierSpec.from_mapping(cast(Mapping[str, object], values))


def load_verifier_table(path: str | Path) -> dict[str, object]:
    """Load a standalone TOML file containing exactly one ``[verifier]`` table."""

    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) != {"verifier"}:
        raise ValueError(
            f"{source} must contain one top-level [verifier] table and no other sections"
        )
    values = document["verifier"]
    if not isinstance(values, dict):
        raise TypeError("[verifier] must be a TOML table")
    # Parse eagerly so configuration errors occur before loading a model.
    VerifierSpec.from_mapping(values)
    return dict(values)


def replace_verifier_from_file(
    config: dict[str, Any], path: str | Path | None
) -> None:
    """Replace the active verifier table with a standalone configuration."""

    if path is not None:
        config["verifier"] = load_verifier_table(path)


def _load_factory(reference: str) -> Callable[..., object]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "verifier.factory must use the form 'package.module:factory_name'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"verifier factory {reference!r} is not callable")
    return cast(Callable[..., object], factory)


def _finite_reward(value: object, *, verifier_name: str) -> float:
    try:
        reward = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"verifier {verifier_name!r} returned a non-numeric reward {value!r}"
        ) from error
    if not isfinite(reward):
        raise ValueError(
            f"verifier {verifier_name!r} returned a non-finite reward {reward!r}"
        )
    return reward


class ConfiguredVerifier:
    """Validated verifier instance plus reproducibility metadata."""

    def __init__(
        self,
        spec: VerifierSpec,
        score: VerifierCallable,
        score_batch: VerifierBatchCallable | None = None,
    ) -> None:
        self.spec = spec
        self._score = score
        self._score_batch = score_batch

    def score(self, prompt: str, completion: str) -> float:
        return _finite_reward(
            self._score(prompt, completion), verifier_name=self.spec.name
        )

    def score_batch(self, inputs: Sequence[VerifierInput]) -> tuple[float, ...]:
        if self._score_batch is None:
            return tuple(self.score(item.prompt, item.completion) for item in inputs)
        values = tuple(self._score_batch(inputs))
        if len(values) != len(inputs):
            raise ValueError(
                f"verifier {self.spec.name!r} returned {len(values)} rewards for "
                f"{len(inputs)} inputs"
            )
        return tuple(
            _finite_reward(value, verifier_name=self.spec.name) for value in values
        )

    def describe(self) -> dict[str, object]:
        return self.spec.as_dict()

    @property
    def version(self) -> str:
        """Stable replay key for the selected verifier configuration."""

        payload = json.dumps(
            self.spec.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.spec.name}:{digest}"


def build_verifier(
    spec: VerifierSpec,
    *,
    context: VerifierContext,
) -> ConfiguredVerifier:
    """Construct a configured verifier without importing model-family code."""

    if spec.requires_reference and context.reference is None:
        raise ValueError(f"verifier {spec.name!r} requires a reference value")
    if spec.provider == "constant":
        value = _finite_reward(spec.options["value"], verifier_name=spec.name)
        return ConfiguredVerifier(spec, lambda _prompt, _completion: value)

    assert spec.factory is not None
    factory = _load_factory(spec.factory)
    constructed = factory(context=context, **dict(spec.options))
    batch: VerifierBatchCallable | None = None
    if isinstance(constructed, TextVerifier):
        point = cast(VerifierCallable, constructed.score)
        possible_batch = getattr(constructed, "score_batch", None)
        if callable(possible_batch):
            batch = cast(VerifierBatchCallable, possible_batch)
    elif callable(constructed):
        point = cast(VerifierCallable, constructed)
    else:
        raise TypeError(
            f"verifier factory {spec.factory!r} returned unsupported value "
            f"{type(constructed).__name__}"
        )
    return ConfiguredVerifier(spec, point, batch)


class TokenVerifierReward:
    """Adapt one text verifier to the token-level algorithm interfaces."""

    def __init__(
        self,
        verifier: ConfiguredVerifier,
        decoder: Callable[[TokenSequence], str],
    ) -> None:
        self.verifier = verifier
        self._decoder = decoder

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.verifier.score(
            self._decoder(prompt), self._decoder(completion)
        )

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        prompt_text = self._decoder(prompt)
        return self.verifier.score_batch(
            tuple(
                VerifierInput(prompt_text, self._decoder(completion))
                for completion in completions
            )
        )

    def describe(self) -> dict[str, object]:
        return self.verifier.describe()

    @property
    def version(self) -> str:
        return self.verifier.version


def _training_text(value: object) -> str:
    """Convert TRL plain or conversational values to verifier text."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if "content" not in value:
            raise TypeError("a training message must contain a content field")
        return str(value["content"])
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(_training_text(item) for item in value)
    raise TypeError(f"unsupported training text value: {type(value).__name__}")


class ConfiguredTrainingVerifierReward:
    """Adapt the same verifier specification to a TRL batched reward callback."""

    def __init__(self, spec: VerifierSpec, *, reference_field: str = "reference") -> None:
        if not reference_field:
            raise ValueError("reference_field must be non-empty")
        self.spec = spec
        self.reference_field = reference_field
        self._verifiers: dict[tuple[str, str | None], ConfiguredVerifier] = {}
        self.calls = 0
        self.completions = 0
        self.completion_tokens = 0
        self.reward_sum = 0.0
        self.reward_minimum: float | None = None
        self.reward_maximum: float | None = None

    def __call__(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **columns: object,
    ) -> list[float]:
        if len(prompts) != len(completions):
            raise ValueError("training prompts and completions have different lengths")
        references: Sequence[object | None]
        if self.spec.requires_reference:
            raw_references = columns.get(self.reference_field)
            if not isinstance(raw_references, Sequence) or isinstance(
                raw_references, (str, bytes, bytearray)
            ):
                raise ValueError(
                    f"training verifier requires column {self.reference_field!r}"
                )
            if len(raw_references) != len(completions):
                raise ValueError("training references and completions have different lengths")
            references = raw_references
        else:
            references = (None,) * len(completions)

        prompt_texts = tuple(_training_text(value) for value in prompts)
        completion_texts = tuple(_training_text(value) for value in completions)
        grouped: dict[tuple[str, str | None], list[int]] = {}
        for index, (prompt, reference) in enumerate(
            zip(prompt_texts, references, strict=True)
        ):
            reference_text = None if reference is None else str(reference)
            grouped.setdefault((prompt, reference_text), []).append(index)

        rewards = [0.0] * len(completions)
        for key, indices in grouped.items():
            prompt, reference = key
            verifier = self._verifiers.get(key)
            if verifier is None:
                verifier = build_verifier(
                    self.spec,
                    context=VerifierContext(prompt=prompt, reference=reference),
                )
                self._verifiers[key] = verifier
            values = verifier.score_batch(
                tuple(
                    VerifierInput(prompt, completion_texts[index]) for index in indices
                )
            )
            for index, value in zip(indices, values, strict=True):
                rewards[index] = value

        self.calls += 1
        self.completions += len(completions)
        if completion_ids is not None:
            if len(completion_ids) != len(completions):
                raise ValueError(
                    "training completion token ids and completions have different lengths"
                )
            self.completion_tokens += sum(len(tokens) for tokens in completion_ids)
        if rewards:
            self.reward_sum += sum(rewards)
            minimum = min(rewards)
            maximum = max(rewards)
            self.reward_minimum = (
                minimum
                if self.reward_minimum is None
                else min(self.reward_minimum, minimum)
            )
            self.reward_maximum = (
                maximum
                if self.reward_maximum is None
                else max(self.reward_maximum, maximum)
            )
        return rewards

    def snapshot(self, *, num_generations: int) -> dict[str, int | float | None]:
        if num_generations <= 0:
            raise ValueError("num_generations must be positive")
        return {
            "reward_calls": self.calls,
            "generated_completions": self.completions,
            "generated_prompt_groups": self.completions // num_generations,
            "generated_completion_tokens": self.completion_tokens,
            "reward_sum": self.reward_sum,
            "observed_mean_reward": (
                self.reward_sum / self.completions if self.completions else None
            ),
            "observed_minimum_reward": self.reward_minimum,
            "observed_maximum_reward": self.reward_maximum,
        }


def build_token_verifier_reward(
    config: Mapping[str, object],
    *,
    context: VerifierContext,
    decoder: Callable[[TokenSequence], str],
) -> TokenVerifierReward:
    """Build the configured text verifier and its token adapter."""

    spec = verifier_spec_from_config(config)
    return TokenVerifierReward(build_verifier(spec, context=context), decoder)


__all__ = [
    "BatchTextVerifier",
    "ConfiguredVerifier",
    "ConfiguredTrainingVerifierReward",
    "TextVerifier",
    "TokenBatchReward",
    "TokenReward",
    "TokenVerifierReward",
    "VerifierContext",
    "VerifierInput",
    "VerifierSpec",
    "build_token_verifier_reward",
    "build_verifier",
    "load_verifier_table",
    "replace_verifier_from_file",
    "verifier_spec_from_config",
]
