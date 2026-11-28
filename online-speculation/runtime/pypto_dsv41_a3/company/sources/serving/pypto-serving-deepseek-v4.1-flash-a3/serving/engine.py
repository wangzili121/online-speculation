# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serialize native graph dispatch and keep request ownership until completion."""

import asyncio
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import threading
from uuid import uuid4

log = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    """The active slot and waiting queue are full."""


class Unavailable(RuntimeError):
    """The worker cannot accept another request."""


@dataclass
class RequestJob:
    key: str = field(default_factory=lambda: uuid4().hex)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass
class ScheduledRequest:
    job: RequestJob
    tokens: list[int]
    max_tokens: int
    ignore_eos: bool
    state: object = None


class Engine:
    """Run initialization, inference and cleanup on the same worker thread."""

    def __init__(self, session_context, max_waiting, max_active=1):
        self.context = session_context
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a3-inference")
        self.max_active = max_active
        self.limit = max_active + max_waiting
        self.jobs = {}
        self.pending = deque()
        self.batch_task = None
        self.opened = False
        self.failed = threading.Event()
        self.completed = 0

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        startup = self.loop.run_in_executor(self.pool, self.context.__enter__)
        try:
            self.session = await asyncio.shield(startup)
        except asyncio.CancelledError:
            try:
                await startup
            except Exception:
                log.exception("A3 startup failed during shutdown")
            else:
                await self.loop.run_in_executor(self.pool, self.context.__exit__, None, None, None)
            finally:
                self.pool.shutdown(wait=True)
            raise
        except BaseException:
            self.pool.shutdown(wait=True)
            raise
        self.opened = True
        return self

    async def __aexit__(self, *_error):
        self.opened = False
        for job in self.jobs.values():
            job.cancelled.set()
        try:
            if self.batch_task is not None:
                await self.batch_task
            await self.loop.run_in_executor(self.pool, self.context.__exit__, None, None, None)
        finally:
            self.pool.shutdown(wait=True)

    @property
    def ready(self):
        return self.opened and not self.failed.is_set()

    def submit(self, tokens, max_tokens, ignore_eos):
        if not self.ready:
            raise Unavailable("The A3 worker is unavailable")
        if len(self.jobs) >= self.limit:
            raise QueueFull("All request slots are occupied")
        job = RequestJob()
        self.jobs[job.key] = job
        if self.max_active > 1:
            self.pending.append(ScheduledRequest(job, tokens, max_tokens, ignore_eos))
            if self.batch_task is None:
                self.batch_task = self.loop.create_task(self._drive_frozen())
        else:
            future = self.loop.run_in_executor(self.pool, self._run, job, tokens, max_tokens, ignore_eos)
            future.add_done_callback(lambda done: self._finish(job, done))
        return job

    def _emit(self, job, token):
        self.loop.call_soon_threadsafe(job.events.put_nowait, ("token", token))

    def _advance_frozen(self, active):
        runner = self.session.orthrus
        for dp, work in active.items():
            if work.state is None:
                job = work.job
                work.state = runner.begin_frozen(
                    job.key, dp, work.tokens, work.max_tokens, work.ignore_eos,
                    lambda token, job=job: self._emit(job, token), job.cancelled.is_set,
                )
        completed = runner.step_frozen([work.state for work in active.values()])
        return {dp: completed[work.state.handle] for dp, work in active.items() if work.state.handle in completed}

    async def _drive_frozen(self):
        active = {}
        try:
            while active or self.pending:
                for dp in range(self.max_active):
                    while dp not in active and self.pending:
                        work = self.pending.popleft()
                        if work.job.cancelled.is_set():
                            self._deliver(work.job, "cancelled")
                        else:
                            active[dp] = work
                if not active:
                    continue
                completed = await self.loop.run_in_executor(self.pool, self._advance_frozen, active)
                for dp, result in completed.items():
                    job = active.pop(dp).job
                    self._deliver(job, "cancelled" if result is None else "done", result)
        except Exception:
            self.failed.set()
            log.exception("A3 batched generation failed")
            for work in (*active.values(), *self.pending):
                self._deliver(work.job, "error")
            self.pending.clear()
        finally:
            self.batch_task = None

    def _run(self, job, tokens, max_tokens, ignore_eos):
        if job.cancelled.is_set():
            raise CancelledError()
        if not self.ready:
            raise Unavailable("The A3 worker is unavailable")
        try:
            return self.session.generate(
                job.key, tokens, max_tokens, ignore_eos,
                lambda token: self._emit(job, token),
                job.cancelled.is_set,
            )
        except CancelledError:
            raise
        except Exception:
            self.failed.set()
            raise

    def _deliver(self, job, kind, result=None):
        self.jobs.pop(job.key, None)
        if kind == "done":
            self.completed += 1
        job.events.put_nowait((kind, result))

    def _finish(self, job, future):
        try:
            result = future.result()
        except (CancelledError, asyncio.CancelledError):
            self._deliver(job, "cancelled")
        except Exception:
            log.exception("A3 request %s failed", job.key)
            self._deliver(job, "error")
        else:
            self._deliver(job, "done", result)

    def snapshot(self):
        return {
            "status": "ready" if self.ready else "unavailable",
            "inflight_requests": len(self.jobs),
            "completed_requests": self.completed,
            "active_capacity": self.max_active,
            "waiting_capacity": self.limit - self.max_active,
            "runtime": self.session.metadata,
        }
