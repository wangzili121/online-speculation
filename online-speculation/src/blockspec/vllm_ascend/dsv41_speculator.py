"""MRV2 Orthrus scheduler adapter for the DeepSeek-V4.1 draft model."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig, replace
from vllm.compilation.backends import set_model_tag
from vllm.distributed import get_tp_group
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

logger = logging.getLogger(__name__)


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
            raise ValueError("Orthrus V4.1 requires DFlash's K-1 destination positions")
        self.requires_non_causal = True
        self.online_stride = int(os.environ.get("ORTHRUS_DSV41_ONLINE_STRIDE", "0"))
        self.online_pending = None
        self.online_round = 0
        self.online_learner = None
        parallel = vllm_config.parallel_config
        if self.online_stride < 0 or (self.online_stride and (
            parallel.data_parallel_size != 1 or vllm_config.scheduler_config.max_num_seqs != 1
        )):
            raise ValueError("V4.1 online update requires a positive stride, DP1 and batch size 1")

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

    def consume_online_feedback(self, target, hidden, batch, sampled, *, structured):
        pending, self.online_pending = self.online_pending, None
        if pending is None or structured or batch.num_reqs != 1 or batch.num_draft_tokens == 0:
            return
        request, anchor, proposal, snapshot = pending
        if request != batch.req_ids[0] or int(batch.positions[batch.logits_indices[0]]) != anchor:
            return
        count = batch.num_draft_tokens
        inputs = batch.input_ids[batch.logits_indices[1:]]
        if count > proposal.numel() or not torch.equal(inputs, proposal[:count].to(inputs)):
            return
        reached = int(sampled[0])
        if reached == 0:
            return
        # Use raw target logits. The sampling path may apply penalties or
        # mutate its own logits in place before returning a verifier verdict.
        teacher = target.compute_logits(hidden[batch.logits_indices])
        group = get_tp_group()
        status = torch.zeros(1, dtype=torch.int32, device=self.device)
        metrics = None
        if group.rank_in_group == 0:
            try:
                from .dsv41_last_layer import LastLayerLearner

                with torch.inference_mode(False):
                    if self.online_learner is None:
                        self.online_learner = LastLayerLearner(
                            self.vllm_config.model_config.model,
                            self.vllm_config.speculative_config.model,
                        )
                    metrics = self.online_learner.update(
                        snapshot, teacher.detach().cpu(), count, reached,
                    )
                status.fill_(1)
            except Exception:
                # Every TP rank must reach this broadcast even when the
                # host-only learner cannot finish an update.
                status.fill_(-1)
                logger.exception("V4.1 Orthrus last-layer update failed before publication")
        group.broadcast(status, src=0)
        outcome = int(status[0])
        if outcome < 0:
            self.online_stride = 0
            self.online_learner = None
        elif outcome == 1:
            changed = self._publish_online_head(group)
            if changed and metrics is not None:
                loss, norm, version = metrics
                logger.warning(
                    "V4.1 Orthrus last-layer update %d published, TV %.6g, gradient norm %.6g",
                    version, loss, norm,
                )

    def _publish_online_head(self, group):
        from .dsv41_contract import head_shapes
        from .dsv41_model import _topology_config

        last = len(self.model.model.layers) - 1
        prefix = f"layers.{last}.attn."
        raw = _topology_config(self.vllm_config.model_config.hf_config)
        weights = {}
        for name, shape in head_shapes(raw, self.model.model.topology).items():
            if not name.startswith(prefix):
                continue
            suffix = name.removeprefix(prefix)
            dtype = torch.float32 if suffix == "attn_sink" else torch.bfloat16
            value = torch.empty(shape, dtype=dtype, device=self.device)
            if group.rank_in_group == 0:
                value.copy_(self.online_learner.master[suffix].detach())
            group.broadcast(value, src=0)
            weights[suffix] = value
        live = self.model.live_last_layer_parameters()
        before = {name: value.detach().clone() for name, value in live.items()}
        result = torch.zeros(2, dtype=torch.int32, device=self.device)
        try:
            self.model.publish_last_layer(weights)
            result[1] = any(not torch.equal(value, before[name]) for name, value in live.items())
        except Exception:
            result[0] = 1
            logger.exception("V4.1 Orthrus last-layer publication failed")
        failed, changed = group.all_reduce(result).tolist()
        if failed:
            with torch.no_grad():
                for name, value in live.items():
                    value.copy_(before[name])
            self.online_stride = 0
            self.online_learner = None
        return not failed and bool(changed)

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

    def _generate_draft(
        self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        hidden = self._run_model(
            num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode,
        )
        count = num_reqs * self.num_speculative_steps
        # DFlash computes destination positions 1..31. The Orthrus head
        # predicts the next token from source rows 0..30 of the same block.
        source = self.sample_indices[:count] - 1
        if attn_metadata is None:
            source = source.clamp_min(0)
        sampled = self.sample_draft(
            hidden[source], self.sample_pos[:count] - 1, self.sample_idx_mapping[:count],
            self.temperature, self.seeds, self.sample_col[:count], self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = sampled.view(num_reqs, self.num_speculative_steps)

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

        # The first query position is the committed token count. DFlash caps
        # its anchor+K length at the context limit, so subtracting K can drop
        # valid history at the end of the context.
        committed = torch.zeros_like(self.input_buffers.seq_lens[:num_reqs_padded])
        committed[:num_reqs] = self.input_buffers.positions[:num_tokens:BLOCK_SIZE]
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
        lookback = getattr(getattr(self, "model_state", None), "lookback_token_ids", None)
        if lookback is not None:
            num_reqs = input_batch.num_reqs
            lookback = compose_draft_lookback(
                input_batch.input_ids[: input_batch.num_tokens],
                input_batch.query_start_loc[: num_reqs + 1],
                lookback[:num_reqs],
                num_rejected[:num_reqs],
            )
        self.model.set_lookback_token_ids(lookback)
        # MRV2 can pass the target's widened mHC stream. Orthrus gets its
        # prefix through the shared KV pages and never consumes this buffer.
        last_hidden_states = last_hidden_states[..., :self.hidden_size]
        if last_hidden_states.shape[-1] < self.hidden_size:
            last_hidden_states = torch.nn.functional.pad(
                last_hidden_states, (0, self.hidden_size - last_hidden_states.shape[-1]),
            )
        aux_hidden_states = None
        capture = False
        if self.online_stride and not dummy_run and input_batch.num_reqs == 1 and int(num_sampled[0]) > 0:
            self.online_round += 1
            capture = self.online_round % self.online_stride == 0
        inner = self.model.model
        inner.capture_last_layer = capture
        inner.last_layer_snapshot = None
        try:
            result = super().propose(
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
            if capture and inner.last_layer_snapshot is not None:
                anchor = int(self.input_buffers.positions[0])
                self.online_pending = (input_batch.req_ids[0], anchor, result[0].clone(), inner.last_layer_snapshot)
            return result
        finally:
            inner.capture_last_layer = False
            inner.last_layer_snapshot = None
