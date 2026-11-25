"""Orthrus greedy verification using the resident DSV4.1 A3 Decode graph."""

from dataclasses import dataclass
from time import perf_counter

import torch

from .dsv41_packed_verification import verify_packed_round


_LIVE_ROWS = (
    "position_ids", "window_slots", "window_indices", "c1a_compressed_slots",
    "c2a_compressed_slots", "token_to_req_indices",
)


@dataclass
class DecodeRoundStats:
    verify_calls: int = 0
    replay_calls: int = 0
    input_bytes: int = 0
    sampled_id_bytes: int = 0
    snapshot_bytes: int = 0
    restore_bytes: int = 0
    snapshot_seconds: float = 0.0
    restore_seconds: float = 0.0
    prepare_seconds: float = 0.0
    upload_seconds: float = 0.0
    graph_seconds: float = 0.0
    verify_seconds: float = 0.0
    replay_seconds: float = 0.0


class PackedDecodeVerifier:
    """Drive the same Decode buffers and weight views as native generation.

    One request owns a DP partition while this verifier runs. The native LM
    head exposes at most 16 sampled rows, including the pending anchor.
    """

    def __init__(self, runtime, graph, config, graph_args, counts, decode_inputs,
                 cache_ranks, decode_ranks, weights, engram, uploads, *, tp_size, has_replay_mode=False):
        self.runtime = runtime
        self.graph = graph
        self.config = config
        self.graph_args = tuple(graph_args)
        self.counts = tuple(counts)
        self.inputs = decode_inputs
        self.cache_ranks = cache_ranks
        self.decode_ranks = decode_ranks
        self.weights = weights
        self.engram = engram
        self.uploads = uploads
        self.tp_size = tp_size
        self.has_replay_mode = has_replay_mode
        self.world = len(decode_ranks)
        self.sampled = torch.empty(decode_ranks[0]["sampled_ids"].shape, dtype=torch.int32)
        state = cache_ranks[0]["state_cache"]
        self._state_snapshot = [torch.empty(state.shape, dtype=state.dtype) for _ in range(tp_size)]
        # Native generation initializes all inputs from the empty Decode image.
        self._previous_owner = None
        self.stats = DecodeRoundStats()

    def verify(self, anchor_token, candidates, *, position, owner):
        """Return kept candidates plus the correction/bonus, and their count."""
        if not 0 <= owner < self.world or owner % self.tp_size:
            raise ValueError("verification owner must be the first TP rank")
        candidates = tuple(candidates)
        if len(candidates) + 1 > self.sampled.shape[0]:
            raise ValueError("candidate block exceeds the native Decode LM-head row limit")
        from models.deepseek_v4_1_flash_a3.utils import paged_slots, window_metadata

        fields = self.inputs.fields[0]
        block_size = fields["window_indices"].shape[1]
        active_ranks = tuple(range(owner, owner + self.tp_size))
        engram_history = self.engram.history.get(owner)
        # Native 1122 writes these paged banks at absolute, append-only
        # positions. SWA and the compressed Indexer lengths mask their suffix.
        # The C2A ring is the only allocation that overwrites committed state.
        states = tuple(self.cache_ranks[rank]["state_cache"] for rank in active_ranks)
        call_number = 0
        verified_history = None

        def save(_writes):
            start = perf_counter()
            for rank, handle, snapshot in zip(active_ranks, states, self._state_snapshot, strict=True):
                self.runtime.copy_from(snapshot.data_ptr(), handle.data_ptr, snapshot.nbytes, worker_id=rank)
                self.stats.snapshot_bytes += snapshot.nbytes
            self.stats.snapshot_seconds += perf_counter() - start
            return engram_history

        def restore(history):
            start = perf_counter()
            for rank, handle, snapshot in zip(active_ranks, states, self._state_snapshot, strict=True):
                self.runtime.copy_to(handle.data_ptr, snapshot.data_ptr(), snapshot.nbytes, worker_id=rank)
                self.stats.restore_bytes += snapshot.nbytes
            if history is None:
                self.engram.history.pop(owner, None)
            else:
                self.engram.history[owner] = history
            self.stats.restore_seconds += perf_counter() - start

        def forward(tokens):
            nonlocal call_number, verified_history
            start = perf_counter()
            replay = call_number > 0
            call_number += 1
            if replay:
                # A causal n-gram lookup for a kept row has the same value
                # before and after discarding a suffix.
                self.engram.history[owner] = verified_history[:position + len(tokens)]
            else:
                self._stage_rows(tokens, position, owner, block_size=block_size,
                                 paged_slots=paged_slots, window_metadata=window_metadata)
                verified_history = self.engram.history[owner]
            self._set_live_rows(len(tokens), sample=not replay)
            self.stats.prepare_seconds += perf_counter() - start
            dirty = active_ranks
            if self._previous_owner is not None and self._previous_owner != owner:
                dirty += tuple(range(self._previous_owner, self._previous_owner + self.tp_size))
            # Conservatively reset the previous group before dispatch. Keep
            # the current owner recorded even if launch fails after uploads.
            self._previous_owner = owner
            before_upload = perf_counter()
            list(self.uploads.map(lambda rank: self.inputs.upload(self.runtime, rank, owner), dirty))
            self.stats.upload_seconds += perf_counter() - before_upload
            self.stats.input_bytes += len(dirty) * self.inputs.images[0].nbytes
            for counts in self.counts:
                counts.zero_()
                counts[owner:owner + self.tp_size] = len(tokens)
            graph_args = self.graph_args
            if replay and self.has_replay_mode:
                graph_args = (*graph_args[:-1], 2)
            before_graph = perf_counter()
            self.runtime.run(self.graph, *graph_args, config=self.config)
            self.stats.graph_seconds += perf_counter() - before_graph
            if replay:
                self.stats.replay_calls += 1
                self.stats.replay_seconds += perf_counter() - start
                return ()

            # One transfer for sampled IDs, with each hardware row padded to
            # eight INT32 values. Greedy serving needs no vocabulary download.
            sampled = self.sampled[:len(tokens)]
            self.runtime.copy_from(sampled.data_ptr(), self.decode_ranks[owner]["sampled_ids"].data_ptr,
                                   sampled.nbytes, worker_id=owner)
            self.stats.sampled_id_bytes += sampled.nbytes
            self.stats.verify_calls += 1
            self.stats.verify_seconds += perf_counter() - start
            return sampled[:, 0].tolist()

        if not candidates:
            return tuple(forward((anchor_token,))), 0
        return verify_packed_round(anchor_token, candidates, (), forward=forward, save=save, restore=restore)

    def _stage_rows(self, tokens, position, owner, *, block_size, paged_slots, window_metadata):
        ids = torch.as_tensor(tokens, dtype=torch.int64)
        live = len(tokens)
        pos = torch.arange(position, position + live, dtype=torch.int32)
        request = torch.zeros(live, dtype=torch.int32)
        table = self.inputs.fields[0]["c1a_index_block_table"]
        c2a_table = self.inputs.fields[0]["c2a_index_block_table"]
        slots, window, _ = window_metadata(pos, request, table)
        compressed_slots = paged_slots(pos, request, c2a_table, block_size, 2, True)
        embedding = self.weights.lookup_rows(ids)
        lookup = self.engram.lookup(ids, live, position, request_id=owner)
        for image in self.inputs.fields[1:]:
            image["x_hc"][:live].copy_(embedding)
            image["engram_lookup"][:, :live].copy_(lookup)
            for name, values in zip(
                _LIVE_ROWS, (pos, slots, window, slots, compressed_slots, request), strict=True
            ):
                image[name][:live].copy_(values)

    def _set_live_rows(self, live, *, sample):
        for index, image in enumerate(self.inputs.fields[1:]):
            for name in _LIVE_ROWS:
                image[name][live:].fill_(-1)
            image["query_start_loc"][1] = live
            logit_rows = image["logit_row_indices"]
            logit_rows.fill_(-1)
            if index == 1 and sample:
                torch.arange(live, dtype=logit_rows.dtype, out=logit_rows[:live])
