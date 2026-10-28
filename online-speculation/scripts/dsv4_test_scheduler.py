"""Test-only allocation failure injection; keep native preemption/recovery."""

import json
import os
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


class AllocationFailureScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failure_log = Path(os.environ["ORTHRUS_TEST_ALLOCATION_LOG"])
        self._injected_requests = set()
        allocate = self.kv_cache_manager.allocate_slots

        def allocate_with_failure(request, *args, **kwargs):
            # Fail once per eligible pressure request, at most four times.
            # Returning None is the native allocator's lack-of-capacity signal.
            if (request.status == RequestStatus.RUNNING and len(self.running) >= 2
                    and request is self.running[-1] and request.num_output_tokens >= 64
                    and request.max_tokens == 1536 and len(self._injected_requests) < 4
                    and request.request_id not in self._injected_requests):
                self._injected_requests.add(request.request_id)
                self._log_allocation_event("injected_allocation_failure", request)
                return None
            return allocate(request, *args, **kwargs)

        self.kv_cache_manager.allocate_slots = allocate_with_failure

    def _log_allocation_event(self, kind, request):
        with self._failure_log.open("a") as stream:
            stream.write(json.dumps(dict(kind=kind, request_id=request.request_id,
                                         computed=request.num_computed_tokens,
                                         output_tokens=request.num_output_tokens,
                                         scheduler_base=Scheduler.__module__ + "." + Scheduler.__name__)) + "\n")

    def _preempt_request(self, request, timestamp, **kwargs):
        self._log_allocation_event("native_preempt", request)
        return super()._preempt_request(request, timestamp, **kwargs)
