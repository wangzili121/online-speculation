"""MRV2 Orthrus scheduler adapter for the DeepSeek-V4.1 draft model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, replace
from vllm.compilation.backends import set_model_tag
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
    prepare_dflash_inputs_factory,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_profiling_chunk_for_draft

from .dsv41_contract import BLOCK_SIZE
from .dsv41_runtime import compose_draft_lookback


class AscendOrthrusV41Speculator(AscendDFlashSpeculator):
    """Reuse DFlash's parallel K-query lifecycle, not its model semantics."""

    _speculator_name = "OrthrusV41"

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        speculative = vllm_config.speculative_config
        if speculative is None or speculative.method not in {"orthrus", "dflash"}:
            raise ValueError("AscendOrthrusV41Speculator requires the Orthrus/DFlash parallel scheduler")
        architectures = set(speculative.draft_model_config.architectures)
        if not architectures & {
            "OrthrusDSV41DraftModel",
            "DFlashOrthrusDSV41DraftModel",
        }:
            raise ValueError("the draft artifact is not an OrthrusDSV41DraftModel")
        if not speculative.enforce_eager:
            raise ValueError("Orthrus V4.1 requires eager mode until graph correctness is qualified")
        super().__init__(vllm_config, device)
        if self.num_query_per_req != BLOCK_SIZE or self.num_speculative_steps != BLOCK_SIZE - 1:
            raise ValueError("Orthrus V4.1 requires anchor plus 31 MASK positions")
        if self.sample_from_anchor:
            raise ValueError("Orthrus V4.1 samples MASK rows, not the anchor row")
        self.requires_non_causal = True

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        del target_attn_layer_names
        speculative = self.vllm_config.speculative_config
        assert speculative is not None
        draft_config = speculative.draft_model_config
        draft_vllm_config = replace(
            self.vllm_config,
            load_config=get_pp_safe_draft_load_config(get_draft_load_config(self.vllm_config)),
        )
        with disable_profiling_chunk_for_draft(draft_vllm_config), set_model_tag("orthrus_v41_head"):
            model = get_model(vllm_config=draft_vllm_config, model_config=draft_config)
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        model.bind_shared_backbone(language_model)
        return model

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        """Bind the draft to target cache groups without allocating draft KV."""
        self.draft_attn_layer_names = {
            name
            for group in kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }
        # Initialize metadata builders for the target cache specifications.
        # This allocates no second KV cache; physical storage remains the
        # model runner's existing target cache.
        DraftModelSpeculator.set_attn(
            self,
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.supports_decode_metadata_update = False

        self.draft_kv_cache_group_ids = [
            group_id for group_id, groups in enumerate(target_attn_groups) if groups
        ]
        if not self.draft_kv_cache_group_ids:
            raise RuntimeError("V4.1 target has no cache groups for Orthrus history")
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
        self._context_slot_mappings = torch.full(
            (len(self.draft_kv_cache_group_ids), self.max_num_tokens),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=self.device,
        )
        self._layer_group_idx = None
        self._group_causal = False
        # DFlash's input kernel remains useful: it creates one known anchor
        # row followed by 31 MASK rows.  Only its writable slot outputs are
        # ignored and replaced with PAD during metadata construction below.
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
            self.vllm_config.cache_config.block_size,
            block_tables,
        )

    def _precompute_context_kv(self, start: int, end: int, dummy_run: bool = False) -> None:
        # The target has already produced all committed history. Orthrus owns
        # no second context cache and must never append to the target here.
        del start, end, dummy_run

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: Any,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Any = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del step, causal, slot_mappings
        if dcp_local_seq_lens is not None:
            raise NotImplementedError("Orthrus V4.1 does not yet support DCP")
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens = int(query_start_loc_np[num_reqs])
        query_start_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
        query_start_cpu[: num_reqs + 1] = torch.from_numpy(query_start_loc_np[: num_reqs + 1])
        query_start_cpu[num_reqs:] = query_start_cpu[num_reqs]
        max_query_len = int((query_start_cpu[1:] - query_start_cpu[:-1]).max())

        # prepare_dflash_inputs stores anchor+K as the draft cache length.  The
        # target cache stops at the anchor, so subtract K before V4.1 metadata
        # is built and provide only PAD slots as a second write barrier.
        committed = (self.input_buffers.seq_lens[:num_reqs_padded] - BLOCK_SIZE).clamp_min(0)
        padded_slots = torch.full_like(
            self.block_tables.slot_mappings[:, :num_tokens],
            PAD_SLOT_ID,
        )
        cpu_upper = seq_lens_cpu_upper_bound[:num_reqs_padded].clone()
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_actual_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=query_start_cpu,
            max_query_len=max_query_len,
            seq_lens=committed,
            max_seq_len=max(int(cpu_upper[:num_reqs].max()), 1),
            block_tables=[table[:num_reqs_padded] for table in self.block_tables.input_block_tables],
            slot_mappings=padded_slots,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=cpu_upper,
            positions=self.input_buffers.positions[:num_tokens],
            causal=False,
            skip_ring_state_update=True,
            draft_layer_names=self.draft_attn_layer_names,
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        lookback = getattr(self.model_state, "lookback_token_ids", None)
        if lookback is not None:
            num_reqs = input_batch.num_reqs
            lookback = compose_draft_lookback(
                input_batch.input_ids[: input_batch.num_tokens],
                input_batch.query_start_loc[: num_reqs + 1],
                lookback[:num_reqs],
                num_rejected[:num_reqs],
            )
        self.model.set_lookback_token_ids(lookback)
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
