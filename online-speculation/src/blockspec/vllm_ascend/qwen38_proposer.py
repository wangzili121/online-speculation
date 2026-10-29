"""Qwen3.8 Orthrus on the native vLLM target, with serialized online replay.

Initial integration: eager, synchronous batch-one text generation, BF16/FP32,
TP-sharded target weights. History is replayed from committed tokens into a
private snapshot; native paged KV/GDN cache sharing is not implemented here.
"""

import json
import logging
from pathlib import Path

import torch

from ..feedback import Feedback
from ..parallel.online import SuffixConfig
from .qwen38_runtime import install_runner_hooks
from .qwen38_attention import install_attention_hooks
from .qwen38_weights import NativeSuffixLearner, TensorParallel, from_native_model


logger = logging.getLogger("vllm.blockspec.qwen38")


class OrthrusWorkerExtension:
    """Read worker counters through vLLM's named RPC, without pickled callables."""

    def orthrus_use_cann_gdn_prefill(self):
        from .qwen38_gdn_backend import install_cann_gdn_prefill
        install_cann_gdn_prefill()
        return "cann"

    def orthrus_stats(self):
        draft = self.model_runner.drafter
        if not getattr(draft, "qwen38_native_proposer", False):
            return {"orthrus": False}
        return {"orthrus": True, "rank": draft.tp.rank, "proposals": draft.proposals,
                "verified": draft.verified, "accepted": draft.accepted,
                "max_verified_tokens": draft.max_verified_tokens, "full_blocks": draft.full_blocks,
                "updates": 0 if draft.learner is None else draft.learner.updates,
                "loss": None if draft.learner is None else draft.learner.last_loss}


class Qwen38OrthrusProposer:
    qwen38_native_proposer = True

    def __init__(self, vllm_config):
        self.config = vllm_config
        spec = vllm_config.speculative_config
        options = (vllm_config.additional_config or {}).get("qwen38_orthrus", {})
        allowed = {"mask_token_id", "online", "last_layers", "update_stride", "learning_rate",
                   "head_dir", "checkpoint_dir", "checkpoint_every"}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("invalid qwen38_orthrus options")
        self.options = dict(options)
        native_backend = (vllm_config.additional_config or {}).get("qwen38_gdn_prefill_backend", "native")
        if native_backend not in {"native", "cann"}:
            raise ValueError("qwen38_gdn_prefill_backend must be native or cann")
        if native_backend == "cann" and vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("CANN GDN prefill requires a BF16 target")
        if type(options.get("online", False)) is not bool:
            raise ValueError("online must be a boolean")
        for name in ("last_layers", "update_stride", "checkpoint_every"):
            if name in options and (type(options[name]) is not int or options[name] < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(options.get("mask_token_id")) is not int:
            raise ValueError("explicit qwen38_orthrus.mask_token_id required")
        parallel = vllm_config.parallel_config
        schedule = vllm_config.scheduler_config
        if (not vllm_config.model_config.enforce_eager or schedule.max_num_seqs != 1
                or getattr(schedule, "async_scheduling", False)
                or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or getattr(parallel, "decode_context_parallel_size", 1) != 1
                or getattr(parallel, "prefill_context_parallel_size", 1) != 1
                or getattr(vllm_config, "lora_config", None) is not None
                or vllm_config.quant_config is not None
                or vllm_config.model_config.dtype not in {torch.bfloat16, torch.float32}):
            raise ValueError("Qwen3.8 Orthrus requires eager sync max_num_seqs=1, PP=DP=DCP=PCP=1, no LoRA, BF16/FP32")
        multimodal = getattr(vllm_config.model_config, "multimodal_config", None)
        if multimodal is not None and not multimodal.language_model_only:
            raise ValueError("Qwen3.8 Orthrus currently requires language_model_only=True")
        if spec.method != "custom_class" or not 1 <= spec.num_speculative_tokens <= 31:
            raise ValueError("custom_class with 1..31 speculative tokens required")
        if getattr(spec, "draft_sample_method", "greedy") not in {"greedy", "probabilistic"}:
            raise ValueError("greedy or probabilistic draft sampling required")
        if getattr(spec, "rejection_sample_method", "standard") != "standard":
            raise ValueError("standard exact rejection sampling required")
        self.block_size = spec.num_speculative_tokens + 1
        self.runner = self.model = self.learner = None
        self.history = self.prefix = self.pending = self.request_id = None
        self.proposals = self.verified = self.accepted = 0
        self.max_verified_tokens = self.full_blocks = 0
        install_runner_hooks()
        install_attention_hooks()
        from .qwen38_gdn_backend import install_recurrent_hook
        install_recurrent_hook()
        if (vllm_config.additional_config or {}).get("qwen38_gdn_prefill_backend") == "cann":
            from .qwen38_gdn_backend import install_cann_gdn_prefill
            install_cann_gdn_prefill()

    def bind_runner(self, runner):
        self.runner = runner
        from vllm.distributed import get_tp_group
        group = get_tp_group()
        self.tp = TensorParallel(group.device_group, group.ranks)

    def load_model(self, target_model):
        with torch.inference_mode(False), torch.no_grad():
            text_config = self.config.model_config.hf_text_config
            self.model = from_native_model(target_model, text_config, tp=self.tp,
                                          block_size=self.block_size,
                                          mask_token_id=self.options["mask_token_id"])
            self.generator = torch.Generator(device=self.model.embedding.weight.device)
            self.generator.manual_seed(self.config.model_config.seed + 1337)
            if self.options.get("head_dir"):
                self._load_head(Path(self.options["head_dir"]))
            if self.options.get("online", False):
                policy = SuffixConfig(last_layers=self.options.get("last_layers", 1),
                                      stride=self.options.get("update_stride", 16), loss="tv",
                                      learning_rate=self.options.get("learning_rate", 1e-5))
                self.learner = NativeSuffixLearner(self.model, policy, self.tp)
        logger.info("Orthrus native target ready: TP=%d, K=%d, online=%s, history=replay_native_weights",
                    self.tp.size, self.block_size, self.learner is not None)

    def propose(self, *args, **kwargs):
        raise RuntimeError("neural Orthrus requires the scoped Ascend runner adapter")

    def dummy_run(self, **kwargs):
        # Weights and FP32 optimizer masters are allocated before KV profiling.
        # Correctness-only replay is bounded by max_model_len, not graph-captured.
        pass

    def _clear_request(self, request_id):
        self.request_id = request_id
        self.prefix = self.history = self.pending = None
        if self.learner is not None:
            self.learner.clear_replay()

    def propose_batch(self, sampled, scheduler_output):
        if self.model is None:
            raise RuntimeError("draft weights have not been bound to the native target")
        runner = self.runner
        if runner.input_batch.num_reqs != 1 or len(sampled) != 1:
            raise RuntimeError("only synchronous batch-one Orthrus is supported")
        if isinstance(sampled, torch.Tensor):
            sampled = [[v for v in row if v >= 0] for row in sampled.cpu().tolist()]
        request_id = runner.input_batch.req_ids[0]
        if request_id != self.request_id:
            self._clear_request(request_id)
        if not sampled[0]:
            return [[]]
        count = int(runner.input_batch.num_tokens_no_spec[0])
        committed = runner.input_batch.token_ids_cpu[0, :count].tolist()
        if not committed or committed[-1] != sampled[0][-1]:
            raise RuntimeError("draft must run after native synchronous token bookkeeping")
        prefix = committed[:-1]
        device = self.model.embedding.weight.device
        with torch.inference_mode(False), torch.no_grad():
            if self.prefix is None or prefix[:len(self.prefix)] != self.prefix:
                self.history, self.prefix = None, []
            new = prefix[len(self.prefix):]
            if new:
                ids = torch.tensor([new], device=device)
                self.history = self.model(ids, view="ar", cache=self.history, logits_to_keep=1).cache
            self.prefix = prefix
            # Anchor 后一次预测整个 MASK 块；所有位置共享同一份已提交的 AR 历史。
            width = min(self.block_size, runner.max_model_len - len(prefix))
            if width < 2:
                self.pending = None
                return [[]]
            inputs = torch.full((1, width), self.options["mask_token_id"], device=device, dtype=torch.long)
            inputs[0, 0] = committed[-1]
            capture = (self.learner.capture_layer if self.learner is not None
                       and self.learner.needs_decoder_feedback else None)
            output = self.model(inputs, view="draft", cache=self.history, capture_layer=capture)
            logits = output.logits[0, :-1].float()
            q = None
            if self.config.speculative_config.draft_sample_method == "probabilistic":
                q = logits.softmax(-1)
                candidates = torch.multinomial(q, 1, generator=self.generator).squeeze(-1)
            else:
                candidates = logits.argmax(-1)
            self.tp.broadcast(candidates)
            self.pending = {"request_id": request_id, "inputs": inputs, "cache": self.history,
                            "boundary": output.boundary, "q": q, "candidates": candidates}
            self.proposals += 1
            return [candidates.cpu().tolist()]

    def _verification_count(self, metadata):
        if len(metadata.num_draft_tokens) != 1:
            raise RuntimeError("batch-one verification metadata required")
        count = metadata.num_draft_tokens[0]
        if not count:
            self.pending = None
            return 0
        pending = self.pending
        if pending is None or pending["request_id"] != self.runner.input_batch.req_ids[0]:
            raise RuntimeError("missing proposal for native verifier request")
        if count > pending["candidates"].numel() or not torch.equal(
                metadata.draft_token_ids.to(pending["candidates"]), pending["candidates"][:count]):
            raise RuntimeError("native verifier draft tokens do not match the pending proposal")
        return count

    def verification_probs(self, metadata):
        count = self._verification_count(metadata)
        if not count:
            return None
        q = self.pending["q"]
        return None if q is None else q[:count].contiguous()

    def teacher_rows(self, logits, metadata):
        count = self._verification_count(metadata)
        if not count or self.learner is None or self.pending["boundary"] is None:
            return None
        with torch.inference_mode(False), torch.no_grad():
            return logits[metadata.target_logits_indices].detach().clone()

    def observe_verified(self, metadata, teacher, sampled):
        count = metadata.num_draft_tokens[0]
        if not count:
            return
        pending, self.pending = self.pending, None
        used = min(int((sampled[0] >= 0).sum().item()), count)
        if used < 1:
            raise RuntimeError("native verifier produced no reached token")
        self.verified += 1
        self.max_verified_tokens = max(self.max_verified_tokens, count)
        self.full_blocks += count == self.block_size - 1
        self.accepted += min(int((sampled[0] >= 0).sum().item()) - 1, count)
        if self.learner is None:
            return
        with torch.inference_mode(False):
            if pending["boundary"] is None:
                self.learner._skip_decoder_feedback(used)
                return
            if teacher is None:
                raise RuntimeError("online update requires raw reached teacher logits")
            feedback = Feedback(pending["inputs"], pending["cache"], teacher[:used], used, pending["boundary"])
            result = self.learner.observe(feedback)
            if result is not None:
                logger.info("Orthrus online update: rank=%d version=%d reached=%d loss=%.7f grad_norm=%.7f",
                            self.tp.rank, result["version"], used, result["loss"], result["gradient_norm"])
                every = self.options.get("checkpoint_every", 16)
                if self.options.get("checkpoint_dir") and self.learner.version % every == 0:
                    self._save_head(Path(self.options["checkpoint_dir"]))

    def _head_identity(self):
        import hashlib
        source = Path(self.config.model_config.model)
        files = (source / "config.json", source / "model.safetensors.index.json")
        if not all(path.is_file() for path in files):
            raise ValueError("head exchange requires a local indexed checkpoint and config.json")
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
        return {"format": "orthrus-qwen38-native-tp-v1", "base_config_sha256": hashes[0],
                "base_index_sha256": hashes[1], "tp_size": self.tp.size,
                "block_size": self.block_size, "mask_token_id": self.options["mask_token_id"],
                "hybrid_policy": "orthrus_qwen35_reference"}

    def _draft_parameters(self):
        return {name: p for name, p in self.model.named_parameters() if ".attention.draft." in name}

    def _load_head(self, directory):
        from safetensors.torch import load_file
        if json.loads((directory / "manifest.json").read_text()) != self._head_identity():
            raise ValueError("head must match the base model, TP layout, block and hybrid policy")
        tensors = load_file(str(directory / f"rank{self.tp.rank}.safetensors"))
        parameters = self._draft_parameters()
        if tensors.keys() != parameters.keys():
            raise ValueError("head tensor names do not match the draft mixers")
        for name, value in tensors.items():
            if (value.shape != parameters[name].shape or value.dtype != parameters[name].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"invalid head tensor: {name}")
        for name, value in tensors.items():
            parameters[name].copy_(value)

    def _save_head(self, directory):
        from safetensors.torch import save_file
        # 每版独立保存；全部 TP 分片写完后，rank 0 才发布 manifest。
        directory = directory / f"version{self.learner.version}"
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: p.detach().cpu().contiguous() for name, p in self._draft_parameters().items()},
                  str(directory / f"rank{self.tp.rank}.safetensors"))
        if self.tp.size > 1:
            torch.distributed.barrier(group=self.tp.group)
        if self.tp.rank == 0:
            (directory / "manifest.json").write_text(json.dumps(self._head_identity(), indent=2) + "\n")
