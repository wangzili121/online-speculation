"""Experimental, inference-only GLM-Flash custom proposer.

Reuse the validated synchronous propose/verify bookkeeping, not Qwen weights,
mixers or its NPU attention hooks. Native GLM owns all persistent caches.
"""

import hashlib
import json
from pathlib import Path

import torch

from .glm53_native import HYBRID_POLICY, NativeGLM53History, SharedGLM53Draft
from .glm53_ops import Glm53Architecture
from .qwen38_proposer import Qwen38OrthrusProposer
from .qwen38_runtime import install_runner_hooks


class GLM53OrthrusProposer(Qwen38OrthrusProposer):
    qwen38_native_proposer = False
    glm53_native_proposer = True

    def __init__(self, vllm_config):
        self.config = vllm_config
        spec = vllm_config.speculative_config
        options = (vllm_config.additional_config or {}).get("glm53_orthrus", {})
        allowed = {"mask_token_id", "hybrid_policy", "head_dir", "initialize_from_target"}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("invalid glm53_orthrus options")
        if options.get("hybrid_policy") != HYBRID_POLICY:
            raise ValueError("explicit experimental GLM KDA/MLA hybrid policy required")
        if type(options.get("initialize_from_target", False)) is not bool:
            raise ValueError("initialize_from_target must be a boolean")
        if bool(options.get("head_dir")) == bool(options.get("initialize_from_target", False)):
            raise ValueError("provide a head_dir OR explicitly initialize_from_target for smoke tests")
        raw = json.loads((Path(vllm_config.model_config.model) / "config.json").read_text())
        Glm53Architecture.from_config(raw)
        text = raw["text_config"]
        if (type(options.get("mask_token_id")) is not int
                or not 0 <= options["mask_token_id"] < text["vocab_size"]):
            raise ValueError("valid explicit GLM mask token ID required")
        pc, mc, sc = (vllm_config.parallel_config, vllm_config.model_config,
                      vllm_config.scheduler_config)
        quant = getattr(vllm_config, "quant_config", None)
        description = getattr(quant, "quant_description", {})
        if (getattr(mc, "quantization", None) != "ascend"
                or not isinstance(description, dict)
                or "W8A8_DYNAMIC" not in description.values()):
            raise ValueError("this GLM deployment adapter requires ModelSlim dynamic W8A8 weights")
        if (not mc.enforce_eager or mc.dtype != torch.bfloat16 or sc.max_num_seqs != 1
                or getattr(sc, "async_scheduling", False)
                or pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1
                or getattr(pc, "decode_context_parallel_size", 1) != 1
                or getattr(pc, "prefill_context_parallel_size", 1) != 1
                or getattr(pc, "use_sequence_parallel_moe", False)
                or getattr(pc, "enable_eplb", False)
                or getattr(vllm_config, "lora_config", None) is not None):
            raise ValueError("GLM adapter requires BF16 activations, eager sync batch one, no PP/DP/CP/SP/EPLB/LoRA")
        if spec.method != "custom_class" or not 1 <= spec.num_speculative_tokens <= 7:
            raise ValueError("native GLM KDA currently requires custom_class with 1..7 candidates")
        if (getattr(spec, "rejection_sample_method", "standard") != "standard"
                or getattr(spec, "draft_sample_method", "greedy") not in {"greedy", "probabilistic"}):
            raise ValueError("standard rejection sampling and a supported draft sampling method required")
        if (getattr(vllm_config.cache_config, "cache_dtype", "auto") not in {"auto", "bfloat16"}
                or getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
            raise ValueError("GLM adapter currently requires BF16 KV and disabled prefix caching")
        self.options = dict(options)
        self.block_size = spec.num_speculative_tokens + 1
        self.runner = self.model = self.learner = None
        self.history = self.history_reader = self.pending = self.request_id = None
        self.proposals = self.verified = self.accepted = 0
        self.max_verified_tokens = self.full_blocks = 0
        install_runner_hooks()

    def load_model(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        if type(target).__name__ != "Glm5NextForCausalLM":
            raise ValueError("native Ascend Glm5NextForCausalLM required")
        with torch.inference_mode(False), torch.no_grad():
            self.model = SharedGLM53Draft(target, self.config)
            self.history_reader = NativeGLM53History(target, self.runner)
            self.generator = torch.Generator(device=self.model.embedding.weight.device)
            self.generator.manual_seed(self.config.model_config.seed + 1337)
            if self.options.get("head_dir"):
                self._load_head(Path(self.options["head_dir"]))

    def _head_identity(self):
        root = Path(self.config.model_config.model)
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return {"format": "orthrus-glm53-flash-native-tp-v1",
                "base_config_sha256": sha(root / "config.json"),
                "base_index_sha256": sha(root / "quant_model_weights.safetensors.index.json"),
                "quant_description_sha256": sha(root / "quant_model_description.json"),
                "tp_size": self.tp.size, "block_size": self.block_size,
                "mask_token_id": self.options["mask_token_id"], "hybrid_policy": HYBRID_POLICY}

    def _load_head(self, directory):
        from safetensors.torch import load_file

        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest != self._head_identity():
            raise ValueError("GLM head does not match base weights, TP layout, quantization or hybrid policy")
        tensors = load_file(str(directory / f"rank{self.tp.rank}.safetensors"))
        parameters = self.model.layers.state_dict()
        if tensors.keys() != parameters.keys():
            raise ValueError("GLM head tensor names do not match native draft projections")
        for name, value in tensors.items():
            if (value.shape != parameters[name].shape or value.dtype != parameters[name].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"invalid GLM head tensor: {name}")
        self.model.layers.load_state_dict(tensors, strict=True)


class GLM53WorkerExtension:
    def orthrus_stats(self):
        draft = self.model_runner.drafter
        if not getattr(draft, "glm53_native_proposer", False):
            return {"orthrus": False}
        return dict(orthrus=True, proposals=draft.proposals, verified=draft.verified,
                    accepted=draft.accepted, full_blocks=draft.full_blocks,
                    history_reads=draft.history_reader.reads, hybrid_policy=HYBRID_POLICY,
                    draft_parameter_bytes=sum(p.numel() * p.element_size()
                                              for p in draft.model.layers.parameters()))
