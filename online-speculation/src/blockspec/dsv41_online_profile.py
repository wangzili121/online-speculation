"""Opt-in timing for a sequence of real online updates in one serving session."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from statistics import median
from time import perf_counter


_PROFILE = ContextVar("dsv41_online_profile", default=None)


def active_profile():
    return _PROFILE.get()


@contextmanager
def profile_online(profile):
    token = _PROFILE.set(profile)
    try:
        yield
    finally:
        _PROFILE.reset(token)


@contextmanager
def timed_stage(name):
    profile = active_profile()
    if profile is None or profile.current is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        profile.current["stage_seconds"][name] += perf_counter() - started


def profiled_stage(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with timed_stage(name):
                return function(*args, **kwargs)
        return measured
    return decorate


class OnlineUpdateProfile:
    """Record both sides of each update, including first use on each rank."""

    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.request_id = None
        self.updated_request = None
        self.records = []
        self.requests = []
        self.dispatched = set()

    def begin_request(self, request_id):
        self.request_id = request_id
        self.requests.append({"request_id": request_id, "decode_seconds": 0.0})

    def can_capture(self):
        return self.request_id != self.updated_request

    @contextmanager
    def update(self, contract, *, owner_is_local):
        self.current = {
            "update": contract["update_number"], "request_id": contract["request_id"],
            "owner_dp": contract["owner"][0], "owner_is_local": owner_is_local,
            "teacher_rows": contract["teacher_rows"],
            "stage_seconds": Counter(), "kernel_dispatch_seconds": Counter(),
            "counts": Counter(), "compiled": [],
            "first_dispatch": [],
        }
        print(json.dumps({"stage": "online_profile_update_start",
                          "update": contract["update_number"], "owner_dp": contract["owner"][0],
                          "owner_is_local": owner_is_local}), flush=True)
        try:
            yield self.current
        except Exception as error:
            self.current["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record = self.current
            self.current = None
            self.records.append(record)
            self.updated_request = contract["request_id"]
            with self.path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({
                "stage": "online_profile_update", "update": record["update"],
                "owner_dp": record["owner_dp"], "teacher_rows": record["teacher_rows"],
                "update_seconds": record.get("update_seconds"),
                "new_programs": len(record["compiled"]),
                "new_rank_dispatches": len(record["first_dispatch"]),
                "error": record.get("error"),
            }), flush=True)

    def dispatch_seen(self, rank, key):
        pair = rank, key
        if pair in self.dispatched:
            return True
        self.dispatched.add(pair)
        return False

    @staticmethod
    def summary(local, peer):
        """Classify a run as hot only when both hosts reuse all dispatches."""
        if [row["update"] for row in local] != [row["update"] for row in peer]:
            raise ValueError("Hosts profiled different online updates")
        rows = []
        for a, b in zip(local, peer, strict=True):
            if (a["request_id"], a["owner_dp"], a["teacher_rows"]) != (
                b["request_id"], b["owner_dp"], b["teacher_rows"],
            ) or a["owner_is_local"] == b["owner_is_local"]:
                raise ValueError("Hosts profiled different feedback blocks")
            owner, participant = (a, b) if a["owner_is_local"] else (b, a)
            rows.append({
                "update": a["update"], "owner_dp": a["owner_dp"],
                "teacher_rows": a["teacher_rows"],
                "loss": owner["loss"],
                "gradient_norm": owner["gradient_norm"],
                "hot": not any(row["compiled"] or row["first_dispatch"] for row in (a, b)),
                "update_seconds": max(a["update_seconds"], b["update_seconds"]),
                "publication_seconds": max(a["publication_seconds"], b["publication_seconds"]),
                "owner_stage_seconds": owner["stage_seconds"],
                "peer_stage_seconds": participant["stage_seconds"],
                "owner_kernel_dispatch_seconds": owner["kernel_dispatch_seconds"],
                "peer_kernel_dispatch_seconds": participant["kernel_dispatch_seconds"],
                "new_programs": sum(len(row["compiled"]) for row in (a, b)),
                "new_rank_dispatches": sum(len(row["first_dispatch"]) for row in (a, b)),
                "jit_seconds_by_host": [sum(entry["seconds"] for entry in row["compiled"])
                                        for row in (a, b)],
                "first_dispatch_seconds_by_host": [
                    sum(entry["seconds"] for entry in row["first_dispatch"]) for row in (a, b)
                ],
                "counts_by_host": [a["counts"], b["counts"]],
            })
        hot = [row for row in rows if row["hot"]]
        stages = sorted({name for row in hot for name in row["owner_stage_seconds"]})
        return {
            "updates": rows, "hot_updates": len(hot),
            "hot_update_median_seconds": median(row["update_seconds"] for row in hot) if hot else None,
            "hot_owner_stage_median_seconds": {
                name: median(row["owner_stage_seconds"].get(name, 0.0) for row in hot)
                for name in stages
            },
        }
