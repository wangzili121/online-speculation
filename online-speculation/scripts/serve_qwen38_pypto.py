#!/usr/bin/env python3
"""Experimental Orthrus service over pinned OneServing; greedy, serialized TP1."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.port <= 65535:
        parser.error("require an assigned nonnegative device and a valid port")
    import asyncio
    import pypto
    import uvicorn
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource

    runtime_identity(pypto)
    head = HeadSource(args.model, args.head_dir)
    native_module(args.source, "weight").verify_model_identity(args.model)
    app = native_module(args.source, "api").app

    @asynccontextmanager
    async def lifespan(application):
        engine = make_engine(args.source, args.model, args.device, head)
        application.state.engine = engine
        try:
            await asyncio.to_thread(engine.start)
            yield
        finally:
            await asyncio.to_thread(engine.stop)

    # 复用原生请求校验、SSE、停止条件和取消逻辑，只替换本进程的模型生命周期。
    app.router.lifespan_context = lifespan
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
