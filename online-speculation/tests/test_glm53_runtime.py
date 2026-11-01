import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest


pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "probe_glm53", Path(__file__).parents[1] / "scripts" / "probe_glm53_ops.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("enabled,available", [(True, True), (False, True), (True, False)])
def test_device_binding_precedes_lazy_operator_registration(monkeypatch, enabled, available):
    calls = []

    def enable():
        calls.append("register")
        return enabled

    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", NS(enable_custom_op=enable))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.kda", NS(
        run_recurrent_kda=(lambda *args: None) if available else None))
    monkeypatch.setattr(probe, "torch", NS(
        npu=NS(set_device=lambda device: calls.append(("device", device))),
    ))
    if enabled and available:
        probe.initialize_npu_runtime()
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            probe.initialize_npu_runtime()
    assert calls == [("device", 0), "register"]
