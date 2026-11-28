# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stream confirmed target predictions from a resident Orthrus session."""

from collections import Counter
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

import torch


@dataclass
class FrozenRequest:
    handle: object
    prompt: list[int]
    max_tokens: int
    ignore_eos: bool
    on_token: Callable[[int], None]
    cancelled: Callable[[], bool]
    offset: int = 0
    tokens: list[int] = field(default_factory=list)
    finished: bool = False
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0


class SpeculativeSession:
    def __init__(self, target, draft, eos_id, chunk_size, verify_width, *, learner=None,
                 update_every=16, teacher_rows=7, learning_rate=1e-5, warmup_updates=8):
        self.target = target
        self.draft = draft
        self.eos_id = eos_id
        self.chunk_size = chunk_size
        self.verify_width = verify_width
        self.learner = learner
        self.update_every = update_every
        self.teacher_rows = teacher_rows
        self.learning_rate = learning_rate
        self.warmup_updates = warmup_updates
        self.rounds = 0
        self.last_wave_seconds = {}

    def _candidate_count(self, position, remaining):
        geometry = self.draft.geometry
        if position + geometry.block > geometry.context:
            return 0
        return min(remaining - 1, self.verify_width - 1)

    def begin_frozen(self, request_id, dp, prompt, max_tokens, ignore_eos, on_token, cancelled):
        if self.learner is not None:
            raise ValueError("Batched Orthrus serving requires a frozen head")
        handle = self.target.admit(request_id, dp, len(prompt))
        return FrozenRequest(handle, prompt, max_tokens, ignore_eos, on_token, cancelled)

    def _emit_frozen(self, request, tokens):
        for token in tokens:
            if request.cancelled():
                break
            request.tokens.append(token)
            request.on_token(token)
            if len(request.tokens) == request.max_tokens or (token == self.eos_id and not request.ignore_eos):
                request.finished = True
                break

    def step_frozen(self, requests, *, speculate=True):
        """Advance one Prefill chunk and one generation round for the ready DP groups."""
        phases = {}
        try:
            prefills = [r for r in requests if r.offset < len(r.prompt) and not r.cancelled()]
            if prefills:
                start = perf_counter()
                rows = self.target.advance_batch(
                    "prefill", [(r.handle, r.prompt[r.offset:r.offset + self.chunk_size]) for r in prefills],
                    greedy_ids=True,
                )
                for r in prefills:
                    r.offset = min(r.offset + self.chunk_size, len(r.prompt))
                    if r.offset == len(r.prompt):
                        self._emit_frozen(r, (rows[r.handle],))
                phases["prefill"] = perf_counter() - start
            ready = [r for r in requests if r.tokens and not r.finished and not r.cancelled()]
            proposals = {}
            if ready and speculate:
                queries = []
                for r in ready:
                    position = len(r.prompt) + len(r.tokens) - 1
                    count = self._candidate_count(position, r.max_tokens - len(r.tokens))
                    if count:
                        queries.append((r.handle, r.tokens[-1], count))
                if queries:
                    start = perf_counter()
                    proposals = self.draft.propose_batch(self.target, queries)
                    phases["draft"] = perf_counter() - start
                    phases.update(getattr(self.draft, "last_propose_seconds", {}))
            if proposals:
                start = perf_counter()
                replies = self.target.verify_batch(
                    [(r.handle, r.tokens[-1], proposals.get(r.handle, ())) for r in ready],
                )
                phases["verify_and_commit"] = perf_counter() - start
                phases.update(getattr(self.target, "last_verify_seconds", {}))
                for r in ready:
                    reply = replies[r.handle]
                    if r.handle in proposals:
                        r.rounds += 1
                        r.proposed += len(proposals[r.handle])
                        r.accepted += reply.accepted
                    self._emit_frozen(r, reply.tokens)
            elif ready:
                start = perf_counter()
                ids = self.target.advance_batch(
                    "decode", [(r.handle, (r.tokens[-1],)) for r in ready], greedy_ids=True,
                )
                phases["decode"] = perf_counter() - start
                for r in ready:
                    self._emit_frozen(r, (ids[r.handle],))
            completed = {}
            start = perf_counter()
            for r in requests:
                cancelled = r.cancelled()
                if r.finished or cancelled:
                    self.target.release(r.handle)
                    if cancelled and not r.finished:
                        completed[r.handle] = None
                    else:
                        completed[r.handle] = dict(
                            output_sequences=[dict(request_id=r.handle.request_id, token_ids=r.tokens)],
                            speculation=dict(rounds=r.rounds, proposed=r.proposed, accepted=r.accepted),
                        )
            phases["release"] = perf_counter() - start
            self.last_wave_seconds = phases
            return completed
        except BaseException:
            self.target.failed = True
            raise

    def generate(self, request_id, prompt, max_tokens, ignore_eos, on_token, cancelled):
        handle = self.target.admit(request_id, 0, len(prompt))
        emitted = []
        proposed = accepted = rounds = 0
        position = len(prompt)
        feedback = None
        updates = []
        phase_seconds = Counter(prefill=0.0, draft=0.0, verify_and_commit=0.0, decode=0.0, release=0.0)

        def check_cancelled():
            if cancelled():
                raise CancelledError()

        def publish(tokens):
            for token in tokens:
                check_cancelled()
                emitted.append(token)
                on_token(token)
                if len(emitted) == max_tokens or (token == self.eos_id and not ignore_eos):
                    return True
            return False

        try:
            start = perf_counter()
            for offset in range(0, len(prompt), self.chunk_size):
                check_cancelled()
                rows = self.target.advance_batch(
                    "prefill", [(handle, prompt[offset:offset + self.chunk_size])], greedy_ids=True,
                )
            phase_seconds["prefill"] = perf_counter() - start
            done = publish((rows[handle],))
            while not done:
                check_cancelled()
                anchor = emitted[-1]
                remaining = max_tokens - len(emitted)
                count = self._candidate_count(position, remaining)
                if count:
                    capture = 0
                    if self.learner is not None and feedback is None and self.rounds % self.update_every == 0:
                        capture = min(remaining, self.teacher_rows)
                    start = perf_counter()
                    if capture:
                        from models.deepseek_v4_1_flash_a3.orthrus_draft.online.loss import PackedTeacherRows

                        candidates = self.draft.propose(self.target, handle, anchor, count, capture_rows=capture)
                        feedback = PackedTeacherRows(self.draft.student)
                    else:
                        candidates = self.draft.propose(self.target, handle, anchor, count)
                    phase_seconds["draft"] += perf_counter() - start
                    phase_seconds.update(getattr(self.draft, "last_propose_seconds", {}))
                    check_cancelled()
                    start = perf_counter()
                    query = [(handle, anchor, candidates)]
                    if feedback is not None:
                        teacher = {handle: feedback.remaining}
                        reply = self.target.verify_batch(query, teacher=teacher)[handle]
                        teacher = teacher[handle]
                    else:
                        reply = self.target.verify_batch(query)[handle]
                    phase_seconds["verify_and_commit"] += perf_counter() - start
                    phase_seconds.update(getattr(self.target, "last_verify_seconds", {}))
                    self.rounds += 1
                    proposed += len(candidates)
                    accepted += reply.accepted
                    rounds += 1
                    tokens = reply.tokens
                else:
                    start = perf_counter()
                    rows = self.target.advance_batch("decode", [(handle, (anchor,))], greedy_ids=feedback is None)
                    phase_seconds["decode"] += perf_counter() - start
                    if feedback is None:
                        tokens = (rows[handle],)
                    else:
                        tokens = (_greedy(rows[handle]),)
                        teacher = rows[handle].unsqueeze(0)
                before = len(emitted)
                done = publish(tokens)
                consumed = len(emitted) - before
                if feedback is not None:
                    available = min(consumed, feedback.remaining)
                    feedback.append(position, teacher[:available])
                    batch = feedback.logits(finished=done)
                    if batch is not None:
                        version = self.learner.optimizer.version
                        rate = self.learning_rate * min(1, (version + 1) / max(1, self.warmup_updates))
                        start = perf_counter()
                        update = self.learner.update(batch, rate=rate)
                        updates.append(dict(asdict(update), wall_seconds=perf_counter() - start))
                        feedback = None
                position += consumed
        except CancelledError:
            raise
        except Exception:
            # A failed Draft, Verify, or update can poison the resident worker.
            # Preserve the original error and let the engine tear down the worker.
            self.target.failed = True
            raise
        finally:
            if self.learner is not None:
                self.draft.student = None
                if self.learner.optimizer.failed:
                    self.target.failed = True
            if not self.target.failed:
                start = perf_counter()
                self.target.release(handle)
                phase_seconds["release"] = perf_counter() - start
        return dict(output_sequences=[dict(request_id=request_id, token_ids=emitted)],
                    speculation=dict(rounds=rounds, proposed=proposed, accepted=accepted),
                    online_updates=updates, phase_seconds=dict(phase_seconds))


def _greedy(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite A3 target logits")
    return int(logits.argmax())
