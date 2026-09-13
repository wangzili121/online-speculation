"""Fixed-version screening state for a later, fresh full-request speed trial."""

import math


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class FullAnswerGate:
    def __init__(self, count=2, margin=1.02, initial_cost=.4, *, reuse_speculative=False):
        if (type(count) is not int or count < 1 or not _positive(margin) or margin < 1
                or not _positive(initial_cost) or type(reuse_speculative) is not bool):
            raise ValueError("positive screening count/cost and margin >= 1 required")
        self.count, self.margin = count, margin
        self.screen_estimate = initial_cost
        self.confirmation_estimate = 0.
        self.ready = False
        self.pending = []
        self.reuse_speculative = reuse_speculative
        self.ar_tps = 0.

    @property
    def estimate(self):
        return self.confirmation_estimate if self.ready else self.screen_estimate

    def observe(self, row, elapsed):
        reference_step = row.get("reference_step", 0)
        if (self.ready or not _positive(elapsed) or type(row["step"]) is not int or row["step"] < 1
                or type(row["reference_tokens"]) is not int or row["reference_tokens"] < 1
                or any(not _positive(row[key]) for key in ("reference_seconds", "predicted_seconds"))
                or any(item["step"] != row["step"] for item in self.pending)
                or type(reference_step) is not int or not 0 <= reference_step < row["step"]
                or (reference_step and (not self.reuse_speculative or not _positive(self.ar_tps)))
                or any(item.get("reference_step", 0) != reference_step for item in self.pending)):
            raise ValueError("positive same-version screening evidence required")
        self.screen_estimate = max(1.2 * elapsed, .8 * self.screen_estimate)
        keys = ("step", "reference_tokens", "reference_seconds", "predicted_seconds")
        if self.reuse_speculative:
            row = {**row, "reference_step": reference_step}
            keys += ("reference_step",)
        self.pending.append({key: row[key] for key in keys})
        reference = sum(item["reference_seconds"] for item in self.pending)
        prediction = sum(item["predicted_seconds"] for item in self.pending)
        ratio = reference / prediction
        count = len(self.pending)
        complete = count == self.count
        self.ready = complete and ratio >= self.margin
        if self.ready:
            switch_cost = max(0., elapsed - row.get("evaluation_seconds", 0.))
            ar_seconds = (sum(item["reference_tokens"] for item in self.pending) / (count * self.ar_tps)
                          if reference_step else 0.)
            self.confirmation_estimate = 1.2 * (prediction / count + ar_seconds + switch_cost)
        comparison = "candidate_over_serving" if reference_step else "candidate_over_ar"
        result = {**row, "ratio": ratio, comparison: ratio, "screen_requests": count,
                  "screen_passed": self.ready, "awaiting_confirmation": self.ready,
                  "passed": False, "complete": complete and not self.ready, "fallback": False}
        if result["complete"]:
            self.pending = []
        return result

    def confirmed(self, *, ar_tps=None):
        if ar_tps is not None:
            if not _positive(ar_tps):
                raise ValueError("positive measured AR throughput required")
            self.ar_tps = ar_tps
        self.pending = []
        self.ready = False

    def state_dict(self):
        result = {"pending": [dict(row) for row in self.pending], "ready": self.ready,
                  "screen_estimate": self.screen_estimate, "confirmation_estimate": self.confirmation_estimate}
        if self.reuse_speculative:
            result["ar_tps"] = self.ar_tps
        return result

    def load_state_dict(self, state, *, step, next_probe, speculating, requests, serving_step=0):
        keys = {"pending", "ready", "screen_estimate", "confirmation_estimate"}
        if self.reuse_speculative:
            keys.add("ar_tps")
        if (not isinstance(state, dict) or state.keys() != keys or type(state["ready"]) is not bool
                or not _positive(state["screen_estimate"])
                or type(state["confirmation_estimate"]) not in (int, float)
                or not math.isfinite(state["confirmation_estimate"]) or state["confirmation_estimate"] < 0):
            raise ValueError("finite full-answer gate state required")
        rows, ready = state["pending"], state["ready"]
        fields = {"step", "reference_tokens", "reference_seconds", "predicted_seconds"}
        if self.reuse_speculative:
            fields.add("reference_step")
            if (type(state["ar_tps"]) not in (int, float) or not math.isfinite(state["ar_tps"])
                    or state["ar_tps"] < 0 or (speculating and state["ar_tps"] == 0)):
                raise ValueError("finite measured AR rate required for live speculative trials")
        reference_step = serving_step if speculating else 0
        if (not isinstance(rows, list) or len(rows) > self.count or len(rows) > requests
                or (ready and (len(rows) != self.count or state["confirmation_estimate"] <= 0))
                or (not ready and len(rows) >= self.count)
                or (rows and ((speculating and (not self.reuse_speculative or not 0 < serving_step < step))
                              or step < 1 or next_probe > step))
                or any(not isinstance(row, dict) or row.keys() != fields or type(row["step"]) is not int
                       or row["step"] != step or type(row["reference_tokens"]) is not int
                       or row["reference_tokens"] < 1
                       or (self.reuse_speculative and (type(row["reference_step"]) is not int
                                                      or row["reference_step"] != reference_step))
                       or any(not _positive(row[key]) for key in ("reference_seconds", "predicted_seconds"))
                       for row in rows)):
            raise ValueError("bounded same-version full-answer evidence required")
        if ready and (sum(r["reference_seconds"] for r in rows) / sum(r["predicted_seconds"] for r in rows)
                      < self.margin):
            raise ValueError("confirmation requires the screening margin")
        self.pending = [dict(row) for row in rows]
        self.ready = ready
        self.screen_estimate = state["screen_estimate"]
        self.confirmation_estimate = state["confirmation_estimate"]
        self.ar_tps = state.get("ar_tps", 0.)
