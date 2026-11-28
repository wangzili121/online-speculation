# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind the chip entry of standalone PyPTO programs to the resident ABI."""

import os
from pathlib import Path


def with_resident_entry(host, chip):
    host.resident_entry = chip
    return host


def resident_entry(program):
    return getattr(program, "resident_entry", program)


def kernel_cache_options(output_dir=None):
    """Share the local PyPTO artifact store between Serving and its startup checks."""
    cache_dir = os.environ.get("PYPTO_DSV41_JIT_CACHE_DIR")
    if not cache_dir:
        return {} if output_dir is None else {"save_kernels_dir": str(output_dir)}
    from pypto import CacheConfig

    return {"cache_config": CacheConfig(
        enabled=True, root=Path(cache_dir), extra_fingerprint=os.environ.get("ORTHRUS_CLUSTER_LIB_TREE"),
    )}


def compile_resident_entry(program, arguments, config, *, abi, name):
    compiled = program.compile(*arguments, config=config)
    if not (Path(compiled.output_dir) / "compiled_meta.json").is_file():
        raise TypeError(f"{name}: resident execution requires a PyPTO L2 chip entry")
    return compiled, abi(compiled)
