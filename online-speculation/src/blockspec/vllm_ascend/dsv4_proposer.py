"""Dedicated vLLM-Ascend proposer for DSV4 Orthrus parallel drafting."""

from __future__ import annotations

import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

from .head_contract import BLOCK_SIZE, NUM_SPECULATIVE_TOKENS


class AscendOrthrusProposer(AscendDSparkProposer):
    """Reuse Ascend's block-expansion buffers, not DSpark generation semantics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner=runner)
        if self.method not in {"orthrus", "dflash"}:
            raise ValueError(f"AscendOrthrusProposer received method={self.method!r}")
        if self.num_speculative_tokens != NUM_SPECULATIVE_TOKENS:
            raise ValueError("DSV4 Orthrus requires 31 speculative tokens")
        if self.sample_from_anchor is not False or self.num_query_per_req != BLOCK_SIZE:
            raise ValueError(
                "DSV4 Orthrus requires anchor plus 31 masked query positions"
            )
        if not self.speculative_config.parallel_drafting:
            raise ValueError("DSV4 Orthrus requires parallel_drafting=true")
        if self.speculative_config.rejection_sample_method != "standard":
            raise ValueError("DSV4 Orthrus requires standard rejection sampling")
        self.dynamic_spec = None
        # Eager is the correctness default. Graph support is enabled only after
        # the exact path passes the NPU test matrix.
        self.use_cuda_graph = False

    def model_returns_tuple(self) -> bool:
        # The generic base treats unknown methods as recurrent MTP models.
        # Orthrus emits one hidden tensor for the entire parallel block.
        return False

    def set_inputs_first_pass(self, *args, **kwargs):
        cad = kwargs.get("cad")
        if cad is None:
            raise TypeError("Orthrus set_inputs_first_pass requires cad")
        original_seq_lens = cad.seq_lens
        original_seq_lens_cpu = cad._seq_lens_cpu
        original_ascend_seq_lens_cpu = getattr(cad, "seq_lens_cpu", None)
        original_max_seq_len = cad.max_seq_len
        num_rejected = kwargs.get("num_rejected_tokens_gpu")
        result = super().set_inputs_first_pass(*args, **kwargs)
        num_tokens, token_indices, draft_cad, long_seq = result

        # DSpark appends the block to its writable draft cache. Orthrus reads
        # committed target history and computes the K32 local block separately,
        # so cache-visible sequence lengths must never include the query block.
        effective_seq_lens = original_seq_lens
        if num_rejected is not None:
            effective_seq_lens = effective_seq_lens - num_rejected
        draft_cad.seq_lens = effective_seq_lens
        draft_cad._seq_lens_cpu = original_seq_lens_cpu
        if hasattr(draft_cad, "seq_lens_cpu"):
            draft_cad.seq_lens_cpu = original_ascend_seq_lens_cpu
        draft_cad.max_seq_len = original_max_seq_len
        draft_cad.causal = False
        draft_cad.attn_mask = None
        return num_tokens, token_indices, draft_cad, long_seq

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        del num_actual_tokens
        if not self.draft_attn_groups:
            raise RuntimeError("Orthrus target cache groups were not initialized")
        per_layer: dict[str, Any] = {}
        shared_prefill_metadata: dict[str, Any] = {}
        shared_decode_metadata: dict[str, Any] = {}
        shared_dsa_metadata: dict[str, Any] = {}
        for group in self.draft_attn_groups:
            builder = group.get_metadata_builder()
            metadata_input = copy.copy(common_attn_metadata)
            gid = group.kv_cache_group_id
            block_table = self._per_group_block_table_buffers.get(gid)
            if block_table is None:
                raise RuntimeError(f"Orthrus has no block table for cache group {gid}")
            metadata_input.block_table_tensor = block_table[: metadata_input.num_reqs]
            slot_mapping = self._per_group_query_slot_mapping_buffers.get(gid)
            if slot_mapping is not None:
                metadata_input.slot_mapping = slot_mapping[:num_input_tokens]
            extra = (
                {
                    "prefill_ratio_to_sas_metadata": shared_prefill_metadata,
                    "decode_ratio_to_sas_metadata": shared_decode_metadata,
                    "common_ratio_to_sas_metadata": shared_dsa_metadata,
                    "block_size": group.kv_cache_spec.block_size,
                }
                if self.use_compress
                else {}
            )
            metadata = builder.build(
                0,
                metadata_input,
                self.runner.get_model(),
                **extra,
            )
            if hasattr(metadata, "causal"):
                metadata.causal = False
            if hasattr(metadata, "attn_mask"):
                metadata.attn_mask = None
            for layer_name in group.layer_names:
                per_layer[layer_name] = metadata
        first = self.draft_attn_groups[0].layer_names[0]
        return [per_layer], per_layer[first]

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        **kwargs,
    ) -> None:
        return super().dummy_run(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            **kwargs,
        )
