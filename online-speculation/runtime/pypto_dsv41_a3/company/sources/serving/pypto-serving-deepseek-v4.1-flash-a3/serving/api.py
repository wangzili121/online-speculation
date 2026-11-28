# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""OpenAI-compatible text completions with the native A3 generation lifecycle."""

import argparse
from math import isfinite
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import uvicorn

from .engine import Engine, QueueFull, Unavailable
from .runtime import LaunchConfig, open_session

DEFAULT_MODEL = "DeepSeek-V4.1-Flash-A3"


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool = False


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str | None = None
    max_tokens: Annotated[int, Field(
        ge=1, le=256, validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )] = 32
    temperature: float = 0.0
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None
    return_token_ids: bool = False


class CompletionRequest(GenerationRequest):
    prompt: str | list[int]


class ChatRequest(GenerationRequest):
    messages: list[dict]


def event(data):
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


async def job_events(job, request):
    while not job.cancelled.is_set():
        if await request.is_disconnected():
            return
        try:
            kind, data = await asyncio.wait_for(job.events.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        yield kind, data
        if kind != "token":
            return


def create_app(session_context, *, model_name=DEFAULT_MODEL, max_waiting=8, max_active=1):
    @asynccontextmanager
    async def lifespan(app):
        async with Engine(session_context(), max_waiting, max_active) as engine:
            app.state.engine = engine
            yield

    app = FastAPI(lifespan=lifespan, title=model_name)

    @app.get("/health")
    async def health():
        engine = app.state.engine
        if not engine.ready:
            raise HTTPException(503, "A3 worker unavailable")
        return {"status": "ready", "model": model_name}

    @app.get("/metrics")
    async def metrics():
        return app.state.engine.snapshot()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"object": "model", "id": model_name}]}

    async def generate(payload, request, *, chat):
        engine = app.state.engine
        session = engine.session
        if payload.model is not None and payload.model != model_name:
            raise HTTPException(404, "Unknown model")
        if payload.temperature != 0:
            raise HTTPException(400, "The A3 graph supports greedy decoding (temperature=0)")
        try:
            tokens = session.encode(payload.messages if chat else payload.prompt, chat=chat)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not tokens or any(type(token) is not int or not 0 <= token < session.metadata["vocab_size"] for token in tokens):
            raise HTTPException(400, "Supply a nonempty prompt or valid token IDs")
        if len(tokens) + payload.max_tokens > session.config.context:
            raise HTTPException(400, "Prompt and requested output exceed --max-model-len")
        try:
            job = engine.submit(tokens, payload.max_tokens, payload.ignore_eos)
        except QueueFull as error:
            raise HTTPException(429, str(error)) from error
        except Unavailable as error:
            raise HTTPException(503, str(error)) from error

        prefix = "chatcmpl-" if chat else "cmpl-"
        base = {"id": prefix + job.key, "created": int(time.time()), "model": model_name}
        object_type = "chat.completion" if chat else "text_completion"
        generated = []

        def usage():
            return {
                "prompt_tokens": len(tokens), "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            }

        def finish_reason():
            if not payload.ignore_eos and generated[-1] == session.eos_id:
                return "stop"
            return "length"

        async def stream():
            obj = "chat.completion.chunk" if chat else object_type
            meta = {**base, "object": obj}
            emitted = ""
            try:
                if chat:
                    choice = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    yield event({**meta, "choices": [choice]})
                async for kind, data in job_events(job, request):
                    if kind == "token":
                        generated.append(data)
                        text = session.decode(generated).rstrip("\ufffd")
                        if not text.startswith(emitted):
                            raise RuntimeError("Tokenizer changed a streamed prefix")
                        delta = text[len(emitted):]
                        emitted = text
                        choice = {"index": 0, "finish_reason": None}
                        choice.update({"delta": {"content": delta}} if chat else {"text": delta})
                        if payload.return_token_ids:
                            choice["token_ids"] = [data]
                        yield event({**meta, "choices": [choice]})
                    elif kind == "done":
                        tail = session.decode(generated)[len(emitted):]
                        choice = {"index": 0, "finish_reason": finish_reason()}
                        choice.update({"delta": {"content": tail}} if chat else {"text": tail})
                        yield event({**meta, "choices": [choice]})
                        if payload.stream_options and payload.stream_options.include_usage:
                            yield event({**meta, "choices": [], "usage": usage()})
                        yield "data: [DONE]\n\n"
                    elif kind == "error":
                        yield event({"error": {"message": "A3 generation failed"}})
            finally:
                job.cancelled.set()

        if payload.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            async for kind, data in job_events(job, request):
                if kind == "token":
                    generated.append(data)
                elif kind == "done":
                    choice = {"index": 0, "finish_reason": finish_reason()}
                    text = session.decode(generated)
                    choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
                    if payload.return_token_ids:
                        choice["token_ids"] = generated
                    result = {**base, "object": object_type, "choices": [choice], "usage": usage()}
                    if payload.return_token_ids:
                        result["prompt_token_ids"] = tokens
                    return result
                elif kind == "error":
                    raise HTTPException(503, "A3 generation failed")
            raise HTTPException(499, "Request cancelled")
        finally:
            job.cancelled.set()

    @app.post("/v1/completions")
    async def completions(payload: CompletionRequest, request: Request):
        return await generate(payload, request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatRequest, request: Request):
        return await generate(payload, request, chat=True)

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE"))
    parser.add_argument("--tp", type=int, choices=(4,), default=4)
    parser.add_argument("--dp", type=int, choices=(4,), default=4)
    parser.add_argument("--ep", type=int, choices=(16,), default=16)
    parser.add_argument("--max-model-len", type=int, default=8200)
    parser.add_argument("--chunk-size", "--long-prefill-token-threshold", type=int, default=8192)
    parser.add_argument("--prefill-moe-slab-rows", choices=(64, 128, 256, 512, 1024, 2048), type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--max-num-waiting-reqs", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true", help="Compile and validate without allocating NPUs")
    parser.add_argument("--graph-dir", type=Path, help="Existing native Prefill and Decode artifacts")
    parser.add_argument("--orthrus-head", type=Path, help="Enable the A3 Orthrus draft with this BF16 head")
    parser.add_argument("--draft-graph", type=Path, help="Reuse a compiled Orthrus draft artifact")
    parser.add_argument("--draft-block", type=int, choices=(8, 16, 32), default=16,
                        help="Orthrus draft block width. K8 supports frozen-head inference.")
    parser.add_argument("--verify-width", type=int, choices=(8, 16), default=8)
    parser.add_argument("--online-last-layers", type=int, choices=range(27), default=0, metavar="N")
    parser.add_argument("--online-loss", choices=("tv", "forward_kl"), default="tv")
    parser.add_argument("--online-graphs", type=Path)
    parser.add_argument("--update-every", type=int, default=16)
    parser.add_argument("--teacher-rows", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=8)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pypto-deepseek-v41-a3"
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("PYPTO_SERVING_CACHE_DIR", cache)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    args.model_root = args.model_root.expanduser().resolve()
    if not args.model_root.is_dir():
        parser.error("--model-root must be an external checkpoint directory")
    try:
        args.devices = tuple(map(int, (args.devices or "").split(",")))
    except ValueError:
        parser.error("--devices must be a comma-separated list of 16 allocated physical IDs")
    if len(args.devices) != 16 or len(set(args.devices)) != 16 or min(args.devices) < 0:
        parser.error("--devices must contain exactly 16 distinct nonnegative IDs")
    allocated = os.environ.get("TASK_DEVICE")
    if allocated is not None and args.devices != tuple(map(int, allocated.split(","))):
        parser.error("--devices must match TASK_DEVICE in order")
    if not 1 <= args.max_model_len <= 16384 or not 1 <= args.chunk_size <= min(8192, args.max_model_len):
        parser.error("Use a context in 1..16384 and a chunk no greater than 8192 or the context")
    if args.max_num_waiting_reqs < 0 or not 1 <= args.port <= 65535:
        parser.error("Use a nonnegative waiting limit and a port in 1..65535")
    if args.orthrus_head is None and args.draft_graph is not None:
        parser.error("--draft-graph requires --orthrus-head")
    if args.max_num_seqs > 1 and (args.orthrus_head is None or args.online_last_layers):
        parser.error("Multiple active sequences require a frozen Orthrus head")
    if args.orthrus_head is not None:
        args.orthrus_head = args.orthrus_head.expanduser().resolve()
        if not (args.orthrus_head / "manifest.json").is_file() or args.max_model_len < args.draft_block:
            parser.error("Orthrus requires a head manifest and space for one complete draft block")
        if args.verify_width > args.draft_block:
            parser.error("Verify width must fit within the draft block")
        if args.draft_block == 8 and args.online_last_layers:
            parser.error("K8 Draft currently supports frozen-head inference")
    if args.online_graphs is not None and not args.online_last_layers:
        parser.error("--online-graphs requires --online-last-layers")
    if args.online_last_layers and (
        args.orthrus_head is None or not 0 < args.teacher_rows < args.verify_width or args.update_every < 1
        or args.warmup_updates < 0 or not isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("Online updates require a head, a positive cadence and rate, and teacher rows inside Verify")
    return args


def launch_config(args):
    return LaunchConfig(
        args.model_root, args.devices, args.max_model_len, args.chunk_size, args.prefill_moe_slab_rows,
        args.graph_dir and args.graph_dir.expanduser().resolve(), args.cache_dir.expanduser().resolve(),
        args.orthrus_head, args.draft_graph and args.draft_graph.expanduser().resolve(),
        args.draft_block, args.verify_width,
        args.online_last_layers, args.online_loss, args.online_graphs and args.online_graphs.expanduser().resolve(),
        args.update_every, args.teacher_rows, args.learning_rate, args.warmup_updates,
    )


def main(argv=None):
    args = parse_args(argv)
    settings = launch_config(args)
    if args.prepare_only:
        with open_session(settings, prepare_only=True) as report:
            print(json.dumps(dict(passed=True, **report)), flush=True)
        return
    app = create_app(lambda: open_session(settings), model_name=args.served_model_name,
                     max_waiting=args.max_num_waiting_reqs, max_active=args.max_num_seqs)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
