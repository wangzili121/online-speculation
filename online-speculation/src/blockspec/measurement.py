"""Paired request-level throughput intervals and internal state equality checks."""

import hashlib
import numpy as np
import torch


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare(records, numerator, denominator, count, rng):
    values = []
    for request in range(count):
        row = []
        for method in (denominator, numerator):
            group = [r for r in records if r["request"] == request and r["method"] == method]
            row.extend((sum(r["tokens"] for r in group), sum(r["seconds"] for r in group)))
        values.append(row)
    values = np.asarray(values)
    summed = values.sum(0)
    draws = values[rng.integers(0, count, size=(2000, count))].sum(1)
    return {"ratio": (summed[2] / summed[3]) / (summed[0] / summed[1]),
            "paired_request_ci95": np.quantile((draws[:, 2] / draws[:, 3]) / (draws[:, 0] / draws[:, 1]),
                                                [.025, .975]).tolist()}
