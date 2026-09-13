"""Fixed-budget SIR/MH comparisons; rewards never receive benchmark answers."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from math import isfinite
import time
from typing import Any, Callable

from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
from inference_scaling.arllm.backends.absorbing import AbsorbingEOSBackend
from inference_scaling.arllm.backends.reference import ReferencePolicyBackend
from inference_scaling.arllm.config import RewardMHConfig, SamplingConfig
from inference_scaling.arllm.reward_factory import model_reward_from_config
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.shared.stepwise import normalize_log_weights, categorical_index_from_uniform
from inference_scaling.shared.rng import SeedStream

REWARDS = ("self_consistency", "sequence_log_probability", "consilience")


def sampling_policy(config: dict[str, Any], *, eos_token_id: int | None = None,
                    require_full_support: bool = False) -> SamplingConfig:
    options = config.get("sampling", {})
    unknown = set(options) - {"temperature", "top_p", "top_k"}
    if unknown:
        raise ValueError(f"unsupported comparison sampling options: {sorted(unknown)}")
    policy = SamplingConfig(**options, eos_token_id=eos_token_id)
    if require_full_support and (policy.top_p != 1.0 or policy.top_k is not None):
        raise ValueError("the shared IS/MH reference policy requires top_p=1 and no top_k truncation")
    return policy


def budget_plan(slots: int, candidates: int, prompt_tokens: int, maximum: int) -> dict[str, int]:
    """Reserve generation+rescoring for each state, including repeated prefixes.

    Two independent consensus pilots cost <= two generation calls, covered by
    the scoring reserve for N >= 2. MH uses N states (initialization + N-1
    proposals). EOS can leave budget unused; allocation is not actual spending.
    """
    if slots <= 0 or candidates < 2 or prompt_tokens <= 0 or maximum <= 0:
        raise ValueError("invalid inference budget")
    length = min(maximum, slots // (2 * candidates) - prompt_tokens)
    if length <= 0:
        raise ValueError("budget is insufficient for model prefixes")
    return {"budget_forward_tokens": slots, "candidates": candidates,
            "max_new_tokens": length, "mh_updates": candidates - 1,
            "reserved_forward_tokens": 2 * candidates * (prompt_tokens + length)}


def generation_cost(prompt_tokens: int, generated_tokens: int, parameter_count: int) -> dict[str, int]:
    # Single-request decoding: prefill + all decode inputs except the final token.
    slots = prompt_tokens + generated_tokens - 1
    return {"generated_tokens": generated_tokens, "generation_forward_token_slots": slots,
            "score_forward_token_slots": 0,
            "estimated_dense_forward_flops": dense_forward_flops(parameter_count, slots)}


def single_sample_budget_length(slots: int, prompt_tokens: int, maximum: int) -> int:
    """Allow a single sample to use the budget without reserving other draws."""
    if slots < prompt_tokens or prompt_tokens <= 0 or maximum <= 0:
        raise ValueError("single-sample budget is insufficient for its prompt")
    return min(maximum, slots - prompt_tokens + 1)


def add_costs(*costs: dict[str, Any]) -> dict[str, Any]:
    keys = {key for cost in costs for key in cost}
    return {key: sum(cost.get(key, 0) for cost in costs) for key in keys}


def check_budget(cost: dict[str, Any], limit: int) -> int:
    used = int(cost.get("generation_forward_token_slots", 0) + cost.get("score_forward_token_slots", 0))
    if used > limit:
        raise RuntimeError(f"inference cost {used} exceeds the declared budget {limit}")
    return used


def crop_sample(sample: dict[str, Any], length: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return tuple(sample["token_ids"][:length]), tuple(sample["token_logprobs"][:length])


def majority_index(contents: list[str], judge) -> int:
    """Group mathematically equivalent final answers; first draw breaks ties."""
    groups: list[list[int]] = []
    for index, content in enumerate(contents):
        if not content or judge.answer_key(content) is None:
            continue
        for group in groups:
            if judge.equivalent(content, contents[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])
    return max(groups, key=len)[0] if groups else 0


class FrozenAnswerReward:
    """Agreement with independent, frozen model outputs, without gold answers."""

    def __init__(self, contents: list[str], judge, decode_content):
        self.contents = tuple(contents)
        self.judge = judge
        self.decode_content = decode_content

    @lru_cache(maxsize=256)
    def __call__(self, prompt, tokens):
        content = self.decode_content(prompt, tokens)
        return sum(self.judge.equivalent(content, pilot) for pilot in self.contents) / len(self.contents)


def reward_temperature(source: str, config: dict[str, Any]) -> float:
    options = config.get("comparison", {})
    value = float(options.get(source + "_temperature", {
        "self_consistency": 0.25, "sequence_log_probability": 10.0, "consilience": 2.0,
    }[source]))
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{source} reward temperature must be finite and positive")
    return value


def compare_sir(*, backend, judge, reference, prompt, config, plan, samples,
                pilots, source, seed, render_output, score_cache):
    """Whole-sequence specialization of conditional IS (one candidate block).

    Shared candidate generation is charged independently to every method. The
    prefix of a sampled sequence has exactly the law of length-capped decoding;
    unused suffixes of a shared pool never enter selection or charged cost.
    """
    length, count = plan["max_new_tokens"], plan["candidates"]
    tokens_and_logs = [crop_sample(sample, length) for sample in samples[:count]]
    sequences = [tokens for tokens, _ in tokens_and_logs]
    contents = [render_output(backend, prompt, tokens, config)["content_text"] for tokens in sequences]
    costs = [generation_cost(len(prompt), len(tokens), backend.parameter_count) for tokens in sequences]
    if source == "self_consistency":
        pilot_tokens = [crop_sample(sample, length)[0] for sample in pilots]
        pilot_contents = [render_output(backend, prompt, tokens, config)["content_text"] for tokens in pilot_tokens]
        frozen = FrozenAnswerReward(pilot_contents, judge,
                    lambda prefix, tokens: render_output(backend, prefix, tokens, config)["content_text"])
        rewards = [frozen(prompt, tokens) for tokens in sequences]
        costs.extend(generation_cost(len(prompt), len(tokens), backend.parameter_count) for tokens in pilot_tokens)
    elif source == "sequence_log_probability":
        # Generation already returned the exact actual-policy log probabilities.
        reward = model_reward_from_config(backend, config, source=source)
        rewards = [reward.scale * sum(logs) for _, logs in tokens_and_logs]
    elif source == "consilience":
        rewards = []
        reward = model_reward_from_config(backend, config, source=source)
        for tokens in sequences:
            if tokens not in score_cache:
                before = asdict(backend.snapshot())
                value = reward(prompt, tokens)
                after = asdict(backend.snapshot())
                score_cache[tokens] = (value, {key: after[key] - before[key] for key in before})
            value, cost = score_cache[tokens]
            rewards.append(value)
            costs.append(cost)
    else:
        raise ValueError(f"unsupported reward: {source}")
    probabilities = normalize_log_weights([value / reward_temperature(source, config) for value in rewards])
    selected = categorical_index_from_uniform(probabilities, float(SeedStream(seed).generator("sir-select", source).random()))
    grades = [judge.grade(content, reference) for content in contents]
    cost = add_costs(*costs)
    used = check_budget(cost, plan["budget_forward_tokens"])
    return {"method": "is", "reward": source, "selected_index": selected, "rewards": rewards,
            "probabilities": list(probabilities), "ess": 1 / sum(p * p for p in probabilities),
            "correct": grades[selected]["correct"], "parseable": grades[selected]["parseable"],
            "conditional_expected_correct": sum(p * int(g["correct"]) for p, g in zip(probabilities, grades, strict=True)),
            "content": contents[selected], "cost": cost, "used_forward_tokens": used,
            "selected_tokens": len(sequences[selected]),
            "candidate_parseable": sum(grade["parseable"] for grade in grades),
            "candidate_correct": sum(grade["correct"] for grade in grades),
            "candidate_truncated": sum(backend.tokenizer.eos_token_id not in tokens for tokens in sequences),
            "thinking_status": render_output(backend, prompt, sequences[selected], config)["thinking_status"],
            **plan}


def compare_mh(*, backend, judge, prompt, reference, config, plan, pilots, source, seed, render_output):
    length = plan["max_new_tokens"]
    temperature = sampling_policy(config, require_full_support=True).temperature
    reference_backend = ReferencePolicyBackend(backend, temperature=temperature)
    stopped = AbsorbingEOSBackend(reference_backend, backend.tokenizer.eos_token_id, absorbing_after=len(prompt))
    pilot_cost = {}
    reward: Callable[[tuple[int, ...], tuple[int, ...]], float]
    if source == "self_consistency":
        pilot_tokens = [crop_sample(sample, length)[0] for sample in pilots]
        pilot_contents = [render_output(backend, prompt, tokens, config)["content_text"] for tokens in pilot_tokens]
        reward = FrozenAnswerReward(pilot_contents, judge,
                    lambda prefix, tokens: render_output(backend, prefix, tokens, config)["content_text"])
        pilot_cost = add_costs(*(generation_cost(len(prompt), len(tokens), backend.parameter_count) for tokens in pilot_tokens))
    else:
        reward = model_reward_from_config(stopped if source == "sequence_log_probability" else backend,
                                          config, source=source)
    # Repeated unchanged states keep the same reward without another model pass.
    cached_reward = lru_cache(maxsize=256)(reward)
    before = asdict(backend.snapshot())
    start = time.perf_counter()
    states = []

    def observe(state):
        after = asdict(backend.snapshot())
        cost = add_costs({key: after[key] - before[key] for key in before}, pilot_cost)
        states.append({"updates": state.attempts, "reward": state.reward,
                       "cost": cost, "tokens": state.token_ids})

    result = run_reward_mh_chain(stopped, prompt,
        RewardMHConfig(total_length=length, block_size=min(256, length),
                       iterations=plan["mh_updates"], suffix_schedule="uniform",
                       reward_temperature=reward_temperature(source, config)),
        SamplingConfig(), cached_reward, SeedStream(seed), on_state=observe)
    duration = time.perf_counter() - start
    cost = states[-1]["cost"]
    used = check_budget(cost, plan["budget_forward_tokens"])
    output = render_output(backend, prompt, result.token_ids, config)
    grade = judge.grade(output["content_text"], reference)
    eos = backend.tokenizer.eos_token_id
    live = result.token_ids.index(eos) + 1 if eos in result.token_ids else len(result.token_ids)
    return {"method": "mh", "reward": source, **grade,
            "content": output["content_text"], "thinking_status": output["thinking_status"],
            "selected_tokens": live, "cost": cost, "used_forward_tokens": used,
            "seconds": duration, "updates": result.attempts, "accepted": result.accepted,
            "changed_updates": sum(step.proposed_token_changes > 0 for step in result.trace),
            "accepted_changed_updates": sum(step.accepted_token_changes > 0 for step in result.trace),
            "trace": [asdict(step) for step in result.trace], **plan}
