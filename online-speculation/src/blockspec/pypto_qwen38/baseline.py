"""Keep the independently delivered native model behind a versioned boundary."""

import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types


ONESERVING_COMMIT = "7aa40ba5238e3a83bb5f25e18817497c181e3e9b"
PYPTO_COMMIT = "98752784548b777b2377a9084411c3716eb341d6"
SIMPLER_COMMIT = "c1f89ea431f5306f78c0e4ab9573c829756d7f1e"
PTO_ISA_COMMIT = "418de18dde8fa33773dc5c384fdfe5506463b628"
PACKAGE = "pypto-serving-qwen3.8-27B"
NAMESPACE = "blockspec.pypto_qwen38._native"


def checked_source(checkout):
    root = Path(checkout).resolve()
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if commit != ONESERVING_COMMIT:
        raise ValueError(f"OneServing must be pinned to {ONESERVING_COMMIT}, found {commit}")
    changed = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=all", "--", PACKAGE],
        text=True).strip()
    if changed:
        raise ValueError("OneServing package is modified; preserve the native baseline")
    package = root / PACKAGE
    if (package / "pypto.pin").read_text().strip() != PYPTO_COMMIT:
        raise ValueError("OneServing PyPTO dependency pin changed")
    return package


def native_module(checkout, name):
    package = checked_source(checkout)
    if not name.isidentifier():
        raise ValueError("a native module name, not a path, is required")
    if NAMESPACE in sys.modules:
        if list(sys.modules[NAMESPACE].__path__) != [str(package / "serving")]:
            raise RuntimeError("another OneServing checkout is already imported")
    else:
        module = types.ModuleType(NAMESPACE)
        module.__path__ = [str(package / "serving")]
        module.__spec__ = importlib.machinery.ModuleSpec(NAMESPACE, loader=None, is_package=True)
        sys.modules[NAMESPACE] = module
    return importlib.import_module(f"{NAMESPACE}.{name}")


def runtime_identity(pypto_module):
    if os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "").lower() not in {"", "0", "false", "no", "off"}:
        raise ValueError("runtime pin checking must remain enabled")
    root = Path(pypto_module.__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PYPTO_COMMIT:
        raise ValueError(f"PyPTO must be pinned to {PYPTO_COMMIT}, found {commit}")
    from pypto.runtime.runtime_pin import check_runtime_pin
    pin = check_runtime_pin()
    if not pin.matches or pin.off_pin or pin.installed != SIMPLER_COMMIT:
        raise ValueError("a verified build of the pinned Simpler submodule is required")
    isa = root / "runtime/build/pto-isa"
    isa_revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={isa}", "-C", str(isa), "rev-parse", "HEAD"], text=True).strip()
    if isa_revision != PTO_ISA_COMMIT:
        raise ValueError("the pinned PTO-ISA source is required")
    toolchain = importlib.metadata.version("ptoas")
    if toolchain not in {"0.65", "0.65.0"} or os.environ.get("PTOAS_ROOT"):
        raise ValueError("use the pinned PTOAS v0.65 wheel without a PTOAS_ROOT override")
    from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
    executable = find_ptoas_binary()
    if executable is None:
        raise ValueError("PTOAS v0.65 executable must be on PATH; package installation alone is insufficient")
    version = check_ptoas_version(executable)
    found = re.search(r"\bptoas(?:\s+version)?\s+v?(\d+(?:\.\d+)+)", version)
    if found is None or found[1] not in {"0.65", "0.65.0"}:
        raise ValueError("the selected PTOAS executable differs from the v0.65 pin")
    return dict(pypto=commit, simpler=pin.installed, pto_isa=isa_revision,
                ptoas=toolchain, ptoas_executable_version=found[1])


def read_manifest(directory):
    value = json.loads((Path(directory) / "manifest.json").read_text())
    if not isinstance(value, dict):
        raise ValueError("head manifest must be an object")
    return value
