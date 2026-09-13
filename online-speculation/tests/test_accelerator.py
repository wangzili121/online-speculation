"""Portable device helpers keep CPU tests independent of CUDA and torch-npu."""

import pytest
import torch

from blockspec import accelerator


def test_cpu_runtime_helpers_are_noops():
    value = accelerator.device("cpu")
    assert value == torch.device("cpu")
    accelerator.synchronize(value)
    accelerator.reset_peak_memory_stats(value)
    assert accelerator.memory_allocated(value) == 0
    assert accelerator.max_memory_allocated(value) == 0
    assert accelerator.name(value) == "cpu"
    assert accelerator.rng_state(value) is None
    assert accelerator.autocast(value, torch.bfloat16).__class__.__name__ == "nullcontext"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="auto, cpu, cuda, or npu"):
        accelerator.device("meta")
