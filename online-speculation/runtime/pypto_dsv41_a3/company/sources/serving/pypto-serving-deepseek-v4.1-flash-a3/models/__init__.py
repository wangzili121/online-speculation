# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select one complete A3 model package for this service process."""

import os
from pathlib import Path


LIB_ROOT = os.environ.get("PYPTO_LIB_ROOT")
if LIB_ROOT:
    model_root = Path(LIB_ROOT).expanduser().resolve() / "models"
    if not (model_root / "deepseek_v4_1_flash_a3" / "__init__.py").is_file():
        raise ImportError(f"DeepSeek V4.1 Flash A3 is absent from PYPTO_LIB_ROOT={LIB_ROOT!r}")
    __path__ = [str(model_root)]
