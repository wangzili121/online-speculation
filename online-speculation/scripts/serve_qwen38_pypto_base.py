#!/usr/bin/env python3
"""Run the unmodified, pinned OneServing model as the migration baseline."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if (len(args.devices) != args.tp or len(set(args.devices)) != args.tp
            or min(args.devices) < 0 or not 1 <= args.port <= 65535):
        parser.error("require distinct assigned devices matching TP and a valid port")
    from blockspec.pypto_qwen38.baseline import checked_source, native_module, runtime_identity
    checked_source(args.source)
    import pypto
    runtime_identity(pypto)
    native_module(args.source, "weight").verify_model_identity(args.model)
    os.environ.update(MODEL_ROOT=str(args.model.resolve()), TASK_DEVICE=",".join(map(str, args.devices)),
                      TP=str(args.tp), DP="1", EP="1", MAX_ACTIVE="1", PREFIX_CACHE="0")
    import uvicorn
    uvicorn.run(native_module(args.source, "api").app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
