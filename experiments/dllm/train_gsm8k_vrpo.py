"""Train a LLaDA LoRA adapter with variance-reduced preference optimization."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.runtime import (
    checkpoint_metadata_hashes,
    implementation_hashes,
    json_fingerprint,
)
from experiments.shared.paired_protocol import load_pairing
from inference_scaling.dllm.config import VRPOSamplingConfig
from inference_scaling.dllm.vrpo import (
    AdapterDisabledReference,
    estimate_vrpo_preference_loss,
    vrpo_forward_token_slots,
)
from inference_scaling.shared.rng import SeedStream

IMPLEMENTATION_FILES = (
    "experiments/dllm/train_gsm8k_vrpo.py",
    "src/inference_scaling/dllm/vrpo.py",
    "src/inference_scaling/dllm/config.py",
)


def _tiny_llada_config(config: Any) -> Any:
    tiny = deepcopy(config)
    values = {
        "vocab_size": 64,
        "hidden_size": 64,
        "dense_intermediate_size": 128,
        "expert_intermediate_size": 32,
        "shared_expert_intermediate_size": None,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_layer_freq": [1],
        "pad_token_id": 1,
        "eos_token_id": 2,
        "torch_dtype": "float32",
    }
    for name, value in values.items():
        setattr(tiny, name, value)
    return tiny


def run_tiny_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Exercise the real LLaDA-MoE class, PEFT, VRPO backward, and AdamW on CPU."""

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoModel

    model_path = str(config["model"]["path"])
    remote_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tiny_config = _tiny_llada_config(remote_config)
    base = AutoModel.from_config(tiny_config, trust_remote_code=True).cpu()
    lora = config["vrpo_lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=[str(value) for value in lora["target_modules"]],
            bias="none",
            task_type=None,
        ),
    )
    reference = AdapterDisabledReference(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    estimate = estimate_vrpo_preference_loss(
        model,
        reference,
        prompt=(1, 2),
        chosen=(3, 4, 5),
        rejected=(6, 7, 8),
        mask_token_id=63,
        config=VRPOSamplingConfig(
            timestep_samples=2,
            masks_per_timestep=1,
            antithetic=True,
        ),
        beta=float(config["vrpo_training"]["beta"]),
        seed=7,
    )
    estimate.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not gradients or any(
        gradient is None or not torch.isfinite(gradient).all()
        for gradient in gradients
    ):
        raise RuntimeError("VRPO preflight produced a missing or non-finite LoRA gradient")
    gradient_l1 = sum(float(gradient.detach().abs().sum()) for gradient in gradients)
    if gradient_l1 <= 0:
        raise RuntimeError("VRPO preflight produced zero LoRA gradients")
    optimizer.step()
    with TemporaryDirectory(prefix="llada-vrpo-preflight-") as adapter_directory:
        model.save_pretrained(adapter_directory)
        reload_base = AutoModel.from_config(
            _tiny_llada_config(remote_config), trust_remote_code=True
        ).cpu()
        reloaded = PeftModel.from_pretrained(reload_base, adapter_directory).eval()
        with torch.inference_mode():
            reload_logits = reloaded(torch.tensor([[1, 2, 3]])).logits
        if not torch.isfinite(reload_logits).all():
            raise RuntimeError("reloaded VRPO adapter produced non-finite logits")
    result = {
        "status": "ok",
        "device": "cpu",
        "model_class": type(base).__name__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "lora_tensors_with_finite_gradient": len(gradients),
        "gradient_l1": gradient_l1,
        "loss": float(estimate.loss.detach()),
        "adapter_reload": "ok",
        "adapter_reload_logits_shape": list(reload_logits.shape),
    }
    del reloaded, reload_base, optimizer, reference, model, base
    gc.collect()
    return result


def _load_pairs(path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "VRPO preference data is absent; run experiments/dllm/prepare_gsm8k_vrpo.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("VRPO preference manifest is not complete")
    fingerprint = manifest.get("fingerprint")
    pairs = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if record.get("fingerprint") != fingerprint:
                raise ValueError("VRPO preference records and manifest disagree")
            if record.get("status") == "pair":
                pairs.append(record)
    if not pairs:
        raise ValueError("VRPO preference data contains no usable pairs")
    return pairs, manifest


def _encode_prompt(tokenizer: Any, text: str) -> tuple[int, ...]:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": text},
    ]
    return tuple(
        int(value)
        for value in tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    )


def _encode_completion(tokenizer: Any, text: str, maximum: int) -> tuple[int, ...]:
    values = tuple(
        int(value)
        for value in tokenizer(text, add_special_tokens=False)["input_ids"][:maximum]
    )
    if not values:
        raise ValueError("VRPO completion became empty after tokenization")
    return values


def _effective_parameter_counts(model: Any) -> tuple[int, int]:
    config = getattr(model, "config", None)
    expert_count = int(getattr(config, "num_experts", 0) or 0)
    active_experts = int(getattr(config, "num_experts_per_tok", 0) or 0)
    expert_fraction = (
        active_experts / expert_count
        if 0 < active_experts <= expert_count
        else 1.0
    )
    total = 0
    active = 0.0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        active += count * expert_fraction if ".experts." in name else count
    return total, int(round(active))


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    output: Path,
    *,
    update: int,
    fingerprint: str,
    metrics: Sequence[dict[str, Any]],
    cost_slots: dict[str, int],
    elapsed_seconds: float,
) -> None:
    import torch

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    (output / "training_state.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "completed_updates": update,
                "metrics": list(metrics),
                "forward_token_slots": cost_slots,
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    training = config["vrpo_training"]
    if args.preflight:
        result = run_tiny_preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    device = str(config["runtime"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(config["runtime"]["dtype"])]
    max_steps = int(args.max_steps or training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    if max_steps <= 0 or accumulation <= 0:
        raise ValueError("VRPO steps and gradient accumulation must be positive")
    data_path = Path(str(training["preference_data"]))
    preference_manifest_path = Path(str(training["preference_manifest"]))
    pairs, preference_manifest = _load_pairs(data_path, preference_manifest_path)
    output = Path(str(config["alignment"]["adapter"]))
    model_path = str(config["model"]["path"])
    source_hashes = implementation_hashes(
        REPOSITORY_ROOT,
        entrypoints=IMPLEMENTATION_FILES,
    )
    effective = {
        "config": config,
        "max_steps": max_steps,
        "preference_fingerprint": preference_manifest["fingerprint"],
        "model_metadata_sha256": checkpoint_metadata_hashes(Path(model_path)),
        "implementation_sha256": source_hashes,
    }
    fingerprint = json_fingerprint(effective)

    state_path = output / "training_state.json"
    optimizer_path = output / "optimizer.pt"
    start_update = 0
    metrics: list[dict[str, Any]] = []
    cost_slots = {"current_policy": 0, "reference_policy": 0, "total": 0}
    elapsed_before = 0.0
    if args.resume == "auto" and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError("existing VRPO checkpoint has another fingerprint")
        start_update = int(state["completed_updates"])
        metrics = list(state.get("metrics", []))
        prior_slots = state.get("forward_token_slots", {})
        cost_slots = {
            name: int(prior_slots.get(name, 0)) for name in cost_slots
        }
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        if start_update >= max_steps and (output / "training_cost.json").is_file():
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "completed_updates": start_update,
                        "output": str(output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    elif args.resume == "never" and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"VRPO output is not empty: {output}")
    if start_update and not optimizer_path.is_file():
        raise FileNotFoundError(f"VRPO optimizer state is absent: {optimizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    attention = str(config["runtime"].get("attention", "sdpa"))
    base = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attention,
    ).to(device)
    if bool(training.get("gradient_checkpointing", True)):
        enable = getattr(base, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("LLaDA model does not expose gradient checkpointing")
        enable()
    base.config.use_cache = False
    if start_update:
        model = PeftModel.from_pretrained(base, output, is_trainable=True)
    else:
        lora = config["vrpo_lora"]
        model = get_peft_model(
            base,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=[str(value) for value in lora["target_modules"]],
                bias="none",
                task_type=None,
            ),
        )
    model.train()
    reference = AdapterDisabledReference(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("VRPO model exposes no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]))
    if start_update:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    sampling = VRPOSamplingConfig(
        timestep_samples=int(training["timestep_samples"]),
        masks_per_timestep=int(training["masks_per_timestep"]),
        antithetic=bool(training["antithetic"]),
    )
    maximum = int(config["generation"]["max_new_tokens"])
    encoded = [
        (
            _encode_prompt(tokenizer, str(pair["prompt"])),
            _encode_completion(tokenizer, str(pair["chosen"]), maximum),
            _encode_completion(tokenizer, str(pair["rejected"]), maximum),
            int(pair["problem_index"]),
        )
        for pair in pairs
    ]
    order = list(range(len(encoded)))
    random.Random(int(training["seed"])).shuffle(order)
    seeds = SeedStream(int(training["seed"]))
    total_parameters, active_parameters = _effective_parameter_counts(model)
    optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for microstep in range(start_update * accumulation, max_steps * accumulation):
            prompt, chosen, rejected, problem_index = encoded[order[microstep % len(order)]]
            estimate = estimate_vrpo_preference_loss(
                model,
                reference,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                mask_token_id=int(config["model"]["mask_token_id"]),
                config=sampling,
                beta=float(training["beta"]),
                seed=seeds.derive("vrpo-train", microstep, problem_index),
            )
            (estimate.loss / accumulation).backward()
            slots = vrpo_forward_token_slots(
                prompt_length=len(prompt),
                chosen_length=len(chosen),
                rejected_length=len(rejected),
                config=sampling,
            )
            for name, value in slots.items():
                cost_slots[name] += value
            if (microstep + 1) % accumulation:
                continue
            update = (microstep + 1) // accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(training["max_grad_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metric = {
                "update": update,
                "problem_index": problem_index,
                "loss": float(estimate.loss.detach()),
                "preference_score": float(estimate.preference_score.detach()),
                "grad_norm": float(grad_norm.detach()),
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            }
            metrics.append(metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            if update % int(training["save_steps"]) == 0 or update == max_steps:
                _save_checkpoint(
                    model,
                    optimizer,
                    output,
                    update=update,
                    fingerprint=fingerprint,
                    metrics=metrics,
                    cost_slots=cost_slots,
                    elapsed_seconds=elapsed_before + time.perf_counter() - started,
                )
        current_slots = cost_slots["current_policy"]
        reference_slots = cost_slots["reference_policy"]
        policy_forward = 2.0 * active_parameters * current_slots
        reference_forward = 2.0 * active_parameters * reference_slots
        policy_backward = 2.0 * policy_forward
        checkpoint_recompute = (
            policy_forward if bool(training.get("gradient_checkpointing", True)) else 0.0
        )
        training_cost = {
            "fingerprint": fingerprint,
            "completed_updates": max_steps,
            "microsteps": max_steps * accumulation,
            "preference_pairs": len(encoded),
            "total_parameters": total_parameters,
            "active_parameters": active_parameters,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "forward_token_slots": cost_slots,
            "estimated_flops": {
                "current_policy_forward": policy_forward,
                "reference_forward": reference_forward,
                "current_policy_backward": policy_backward,
                "gradient_checkpoint_recompute": checkpoint_recompute,
                "total": policy_forward
                + reference_forward
                + policy_backward
                + checkpoint_recompute,
            },
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
            ),
            "cost_definition": (
                "forward = 2 * active parameters * model-input token slots; "
                "backward is estimated as two policy forwards; gradient-checkpoint "
                "recomputation is one additional policy forward"
            ),
        }
        (output / "training_cost.json").write_text(
            json.dumps(training_cost, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        del optimizer, reference, model, base
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
