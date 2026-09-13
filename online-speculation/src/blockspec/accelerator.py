"""Small device bridge shared by the reference PyTorch runners."""

from contextlib import nullcontext
import importlib

import torch


def device(value="auto"):
    """Resolve a caller-selected CPU, CUDA, or Ascend NPU device."""
    value = str(value)
    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        else:
            try:
                importlib.import_module("torch_npu")
            except ImportError:
                value = "cpu"
            else:
                value = "npu" if torch.npu.is_available() else "cpu"
    if value.split(":", 1)[0] == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as error:
            raise RuntimeError("an Ascend NPU device requires torch_npu") from error
    result = torch.device(value)
    if result.type not in ("cpu", "cuda", "npu"):
        raise ValueError("device must be auto, cpu, cuda, or npu")
    if result.type != "cpu" and result.index is None:
        result = torch.device(result.type, getattr(torch, result.type).current_device())
    return result


def _module(value):
    value = device(value)
    return None if value.type == "cpu" else getattr(torch, value.type)


def synchronize(value):
    module = _module(value)
    if module is not None:
        module.synchronize(device(value))


def reset_peak_memory_stats(value):
    module = _module(value)
    if module is not None:
        module.reset_peak_memory_stats(device(value))


def memory_allocated(value):
    module = _module(value)
    return 0 if module is None else module.memory_allocated(device(value))


def max_memory_allocated(value):
    module = _module(value)
    return 0 if module is None else module.max_memory_allocated(device(value))


def name(value):
    value = device(value)
    module = _module(value)
    if module is None:
        return "cpu"
    return module.get_device_name() if value.index is None else module.get_device_name(value)


def rng_state(value):
    module = _module(value)
    return None if module is None else module.get_rng_state(device(value))


def set_rng_state(state, value):
    module = _module(value)
    if module is not None:
        module.set_rng_state(state.cpu(), device(value))


def autocast(value, dtype):
    value = device(value)
    return nullcontext() if value.type == "cpu" else torch.autocast(value.type, dtype=dtype)


def profiler(value, **kwargs):
    """Create the native CPU/CUDA or torch-npu profiler for a device."""
    value = device(value)
    if value.type == "npu":
        torch_npu = importlib.import_module("torch_npu")
        activities = [torch_npu.profiler.ProfilerActivity.CPU,
                      torch_npu.profiler.ProfilerActivity.NPU]
        return torch_npu.profiler.profile(activities=activities, **kwargs)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if value.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities, **kwargs)
