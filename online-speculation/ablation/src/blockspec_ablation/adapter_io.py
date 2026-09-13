"""Validated PEFT tensor bridge for independently executed reference weights."""

import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import _validate_state, adapter_state


def peft_config(directory):
    config = json.loads((Path(directory) / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("peft_type") != "LORA" or config.get("bias", "none") != "none"
            or config.get("fan_in_fan_out", False) or config.get("use_dora", False)
            or config.get("use_rslora", False) or config.get("modules_to_save")
            or config.get("rank_pattern") or config.get("alpha_pattern")):
        raise ValueError("expected uniform-rank, alpha/r-scaled, bias-free LoRA")
    if type(config.get("r")) is not int or config["r"] < 1:
        raise ValueError("positive integer PEFT rank required")
    alpha = config.get("lora_alpha")
    if type(alpha) not in (int, float) or not 0 < alpha < float("inf"):
        raise ValueError("positive finite PEFT alpha required")
    return config


def load_peft_adapter(directory, model, *, expected_sha256=None):
    """Check the entire artifact and key mapping before changing adapter tensors.

    The caller may supply a local integrity check and loads the matching base. Dropout
    is an offline training setting; inference uses the deterministic A/B branch.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    config = peft_config(directory)
    if (config["r"] != model.config.adapter_rank or
            config["lora_alpha"] != model.config.adapter_alpha or
            set(config.get("target_modules", [])) != set(model.config.adapter_targets)):
        raise ValueError("PEFT rank, scaling or projection targets differ from model")
    path = directory / "adapter_model.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("PEFT artifact SHA256 differs from the declared source")
    original = load_file(str(path))
    state = {}
    for name, value in original.items():
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected PEFT tensor: {name}")
        target = name.removeprefix("base_model.model.")[:-len(".weight")]
        if target in state:
            raise ValueError("duplicate mapped PEFT tensor")
        state[target] = value
    _validate_state(adapter_state(model), state)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])
    return {"kind": "published_peft_reference", "sha256": digest, "tensors": len(state),
            "rank": config["r"], "alpha": config["lora_alpha"]}
