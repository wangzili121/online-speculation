# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Draft query projection and read-only A3 QLI over the committed prefix."""

from dataclasses import dataclass


INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_CAPACITIES = (512, 1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class IndexGeometry:
    ratio: int
    bucket: int
    code_rows: int
    scale_rows: int
    code_stride: int
    scale_stride: int
    kv_stride: int
    code_offset: int
    scale_offset: int
    kv_offset: int
