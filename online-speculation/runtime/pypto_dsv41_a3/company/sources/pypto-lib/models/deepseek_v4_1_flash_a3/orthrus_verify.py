# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolve the emitted prefix of a greedy Orthrus verification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Verification:
    """Emitted tokens end with the pending correction or bonus token."""

    tokens: tuple[int, ...]
    accepted: int

    @classmethod
    def resolve(cls, candidates, predictions):
        if len(predictions) != len(candidates) + 1:
            raise ValueError("Verification requires one target prediction per input row")
        accepted = 0
        for candidate, prediction in zip(candidates, predictions):
            if candidate != prediction:
                break
            accepted += 1
        return cls((*candidates[:accepted], predictions[accepted]), accepted)
