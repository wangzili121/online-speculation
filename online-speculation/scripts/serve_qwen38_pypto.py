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
    parser.add_argument("--head-storage", choices=("auto", "resident", "stream"), default="auto")
    parser.add_argument("--head-transfer", choices=("individual", "packed"), default="individual")
    parser.add_argument("--attention-backend", choices=("native", "fused", "paged"), default="native")
    parser.add_argument("--draft-group-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--gdn-backend", choices=("native", "vector"), default="native")
    parser.add_argument("--serving-backend", choices=("device", "host"), default="device",
                        help="input preparation, greedy selection and GDN transaction snapshots")
    parser.add_argument("--weight-backend", choices=("pypto", "host"), default="pypto")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--last-layers", type=int, choices=range(1,65), default=1)
    parser.add_argument("--update-stride", type=int, default=16)
    parser.add_argument("--replay-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--learner-device", type=int, help="separate visible NPU for last-N replay and optimizer")
    parser.add_argument("--learner-head-dir", type=Path, help="raw training head when --head-dir uses prepared banks")
    parser.add_argument("--checkpoint-dir", type=Path, help="save each committed update into a versioned directory")
    parser.add_argument("--resume-online", type=Path, help="resume a saved last-layer Adam checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.port <= 65535:
        parser.error("require an assigned nonnegative device and a valid port")
    if not args.online and any(value is not None for value in
                               (args.learner_head_dir, args.checkpoint_dir, args.resume_online, args.learner_device)):
        parser.error("learner/checkpoint options require --online")
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
    online_config = None
    if args.online:
        from blockspec.pypto_qwen38.online import OnlineConfig
        online_config = OnlineConfig(last_layers=args.last_layers, stride=args.update_stride,
            replay_blocks=args.replay_blocks, learning_rate=args.learning_rate, clip_norm=args.clip_norm,
            raw_head_dir=args.learner_head_dir,
            checkpoint_dir=args.checkpoint_dir, resume_dir=args.resume_online, learner_device=args.learner_device)

    @asynccontextmanager
    async def lifespan(application):
        engine = make_engine(args.source, args.model, args.device, head,
                             head_storage=args.head_storage, head_transfer=args.head_transfer,
                             attention_backend=args.attention_backend, group_size=args.draft_group_size,
                             gdn_backend=args.gdn_backend, device_serving=args.serving_backend == "device",
                             weight_backend=args.weight_backend, online_config=online_config)
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
