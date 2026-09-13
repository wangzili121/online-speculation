"""Build verified GSM8K preference pairs for LLaDA VRPO training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.runtime import (
    capped_generation_length,
    checkpoint_metadata_hashes,
    implementation_hashes,
    json_fingerprint,
    llada_snapshot_delta,
    sampling_from_section,
    validate_llada_weights,
)
from experiments.dllm.profiles import apply_execution_profile
from experiments.shared.paired_protocol import load_pairing
from experiments.shared.artifacts import load_jsonl as _load_records
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.preferences import select_scored_preference_pair
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import (
    VerifierContext,
    VerifierInput,
    build_verifier,
    replace_verifier_from_file,
    verifier_spec_from_config,
)

IMPLEMENTATION_FILES = (
    "experiments/dllm/prepare_gsm8k_vrpo.py",
    "experiments/dllm/profiles.py",
    "src/inference_scaling/dllm/preferences.py",
    "src/inference_scaling/dllm/backends/llada.py",
    "src/inference_scaling/dllm/backends/loader.py",
    "src/inference_scaling/shared/verifier.py",
    "src/inference_scaling/shared/evaluation/numeric.py",
    "src/inference_scaling/shared/evaluation/gsm8k.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--include-reference-completion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    replace_verifier_from_file(config, args.verifier_config)
    verifier_spec = verifier_spec_from_config(config)
    training = config["vrpo_training"]
    output_path = args.output or Path(str(training["preference_data"]))
    manifest_path = args.manifest or Path(str(training["preference_manifest"]))
    max_pairs = int(args.max_pairs or training["preference_pairs"])
    pool_size = int(args.candidate_pool_size or training["candidate_pool_size"])
    generations = int(training["num_generations"])
    include_reference_completion = (
        bool(training.get("include_reference_completion", True))
        if args.include_reference_completion is None
        else args.include_reference_completion
    )
    for name, value in (
        ("max_pairs", max_pairs),
        ("candidate_pool_size", pool_size),
        ("num_generations", generations),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if max_pairs > pool_size:
        raise ValueError("max_pairs cannot exceed candidate_pool_size")

    model_dir = Path(str(config["model"]["path"]))
    actual_hashes = validate_llada_weights(config)

    problems = select_problems(
        load_gsm8k(args.data, split="train"),
        pool_size,
        seed=int(training["seed"]),
    )
    effective = {
        "config": config,
        "profile": args.profile,
        "problem_indices": [problem.index for problem in problems],
        "max_pairs": max_pairs,
        "candidate_pool_size": pool_size,
        "num_generations": generations,
        "include_reference_completion": include_reference_completion,
        "model_weight_sha256": actual_hashes,
        "model_metadata_sha256": checkpoint_metadata_hashes(model_dir),
        "implementation_sha256": implementation_hashes(
            REPOSITORY_ROOT,
            entrypoints=IMPLEMENTATION_FILES,
        ),
    }
    fingerprint = json_fingerprint(effective)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("fingerprint") != fingerprint:
            raise ValueError("existing VRPO preference manifest has another fingerprint")
    else:
        manifest_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "status": "running", **effective},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    records = _load_records(output_path)
    if any(record.get("fingerprint") != fingerprint for record in records):
        raise ValueError("existing VRPO preference records have another fingerprint")
    completed = {int(record["problem_index"]) for record in records}
    pair_count = sum(record.get("status") == "pair" for record in records)
    backend = load_llada_backend(config, "base")
    sampling = sampling_from_section(config["generation"])
    generation_length = capped_generation_length(
        prompt_length=0,
        maximum=int(config["generation"]["max_new_tokens"]),
        sampling=sampling,
    )
    seeds = SeedStream(int(training["seed"]))
    before = backend.snapshot()
    started = time.perf_counter()
    try:
        with output_path.open("a", encoding="utf-8", buffering=1) as output:
            for problem in problems:
                if pair_count >= max_pairs:
                    break
                if problem.index in completed:
                    continue
                prompt_text = gsm8k_prompt(problem.question)
                prompt = backend.encode_chat(prompt_text)
                requests = [
                    DiffusionGenerationRequest(
                        prefix=prompt,
                        generation_length=generation_length,
                        sampling=sampling,
                        seed=seeds.derive("vrpo-preference", problem.index, draw),
                        request_id=f"vrpo-preference:{problem.index}:{draw}",
                    )
                    for draw in range(generations)
                ]
                samples = backend.sample_batch(requests)
                texts = [backend.decode(sample.token_ids) for sample in samples]
                verifier = build_verifier(
                    verifier_spec,
                    context=VerifierContext(
                        prompt=prompt_text,
                        reference=(
                            str(problem.gold_answer)
                            if verifier_spec.requires_reference
                            else None
                        ),
                        metadata={"benchmark": "gsm8k", "problem_index": problem.index},
                    ),
                )
                scored_texts = (
                    (*texts, problem.gold_solution)
                    if include_reference_completion
                    else tuple(texts)
                )
                verifier_rewards = verifier.score_batch(
                    tuple(VerifierInput(prompt_text, text) for text in scored_texts)
                )
                pair = select_scored_preference_pair(
                    candidate_texts=texts,
                    candidate_rewards=(
                        verifier_rewards[:-1]
                        if include_reference_completion
                        else verifier_rewards
                    ),
                    reference_text=(
                        problem.gold_solution if include_reference_completion else None
                    ),
                    reference_reward=(
                        verifier_rewards[-1] if include_reference_completion else None
                    ),
                )
                record: dict[str, Any] = {
                    "fingerprint": fingerprint,
                    "problem_index": problem.index,
                    "question": problem.question,
                    "gold_answer": str(problem.gold_answer),
                    "reference_completion_verifier_reward": (
                        verifier_rewards[-1] if include_reference_completion else None
                    ),
                    "candidates": [
                        {
                            "draw": draw,
                            "text": text,
                            "prediction": (
                                str(prediction) if prediction is not None else None
                            ),
                            "correct": prediction == problem.gold_answer,
                            "verifier_reward": verifier_rewards[draw],
                        }
                        for draw, (text, prediction) in enumerate(
                            (
                                (text, extract_numeric_answer(text))
                                for text in texts
                            )
                        )
                    ],
                }
                if pair is None:
                    record["status"] = "skipped_equal_verifier_rewards"
                else:
                    record.update(
                        status="pair",
                        prompt=prompt_text,
                        chosen=pair.chosen,
                        rejected=pair.rejected,
                        chosen_source=pair.chosen_source,
                        rejected_source=pair.rejected_source,
                    )
                    pair_count += 1
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed.add(problem.index)
                print(
                    f"preferences rows={len(completed)}/{len(problems)} pairs={pair_count}/{max_pairs}",
                    flush=True,
                )
        if pair_count < max_pairs:
            raise RuntimeError(
                f"candidate pool yielded {pair_count} pairs, fewer than requested {max_pairs}"
            )
        compute = llada_snapshot_delta(before, backend.snapshot())
        manifest = {
            "fingerprint": fingerprint,
            "status": "complete",
            **effective,
            "records": len(records),
            "preference_pairs": pair_count,
            "elapsed_seconds": time.perf_counter() - started,
            "generation_compute": compute,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        del backend
        gc.collect()
        if str(config["runtime"]["device"]).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
