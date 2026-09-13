"""Role-aware LLaDA backend construction for paired experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from inference_scaling.dllm.backends.llada import LLaDATransformersBackend

LLaDARole = Literal["base", "proposal", "aligned"]


def load_llada_backend(
    config: Mapping[str, Any],
    role: LLaDARole,
    *,
    base_backend: LLaDATransformersBackend | None = None,
) -> LLaDATransformersBackend:
    """Load the full model, shared early-exit proposal, or aligned adapter."""

    model = config["model"]
    runtime = config["runtime"]
    base_path = str(model["path"])
    common = {
        "device": str(runtime.get("device", "cuda")),
        "dtype": str(runtime.get("dtype", "bfloat16")),
        "mask_token_id": int(model.get("mask_token_id", 156895)),
        "max_batch_size": int(runtime.get("max_batch_size", 8)),
    }
    attention = runtime.get("attention")
    if attention:
        common["attn_implementation"] = str(attention)
    if role == "base":
        return LLaDATransformersBackend.from_pretrained(base_path, **common)

    if role == "proposal":
        proposal = config["proposal"]
        if str(proposal.get("kind")) != "shared_prefix_layers":
            raise ValueError("only the shared prefix-layer LLaDA proposal is supported")
        layers = int(proposal["layers"])
        if layers <= 0:
            raise ValueError("proposal.layers must be positive")
        shared_base = base_backend or LLaDATransformersBackend.from_pretrained(
            base_path, **common
        )
        return shared_base.with_prefix_layers(layers)

    if role != "aligned":
        raise ValueError(f"unknown LLaDA role {role!r}")
    adapter_path = Path(str(config["alignment"]["adapter"]))
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"aligned LLaDA adapter is absent: {adapter_path}; run the VRPO stage first"
        )
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("loading the aligned LLaDA adapter requires PEFT") from exc
    base = LLaDATransformersBackend.from_pretrained(base_path, **common)
    aligned_model = PeftModel.from_pretrained(base.model, adapter_path).eval()
    return LLaDATransformersBackend(
        aligned_model,
        base.tokenizer,
        model_id=str(adapter_path),
        mask_token_id=base.mask_token_id,
        max_batch_size=int(runtime.get("max_batch_size", 8)),
    )


__all__ = ["LLaDARole", "load_llada_backend"]
