"""Open-loop OpenAI-compatible serving benchmark with streaming SLO metrics."""

import asyncio
import json
import random
import re
import time

import numpy as np

from blockspec.vllm_ascend.workload import read_manifest


_METRIC = re.compile(r'^(vllm:[^{ ]+)(?:\{([^}]*)\})?\s+([^ ]+)$')
_TELEMETRY_NAMES = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
}


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match is None or match.group(1) not in _TELEMETRY_NAMES:
            continue
        name, labels, value = match.groups()
        key = name
        if labels and 'position="' in labels:
            continue
        result[key] = float(value)
    return result


async def _metrics(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return parse_metrics(await response.text())


async def _poll_metrics(session, url, origin, stop, samples):
    while not stop.is_set():
        try:
            samples.append({"seconds": time.perf_counter() - origin,
                            **await _metrics(session, url)})
        except Exception as exc:
            samples.append({"seconds": time.perf_counter() - origin,
                            "error": f"{type(exc).__name__}: {exc}"})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass


def _percentiles(values):
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    return {f"p{quantile}": float(np.percentile(values, quantile))
            for quantile in (50, 95, 99)}


async def _request(session, semaphore, *, url, model, record, launch_at,
                   timeout):
    await asyncio.sleep(max(0.0, launch_at - time.perf_counter()))
    queued = time.perf_counter()
    payload = {
        "model": model,
        "messages": record["messages"],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": record["max_tokens"],
        "seed": record["seed"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first = None
    usage = {}
    error = None
    status = None
    async with semaphore:
        started = time.perf_counter()
        try:
            async with session.post(
                    url, json=payload, timeout=timeout,
                    headers={"X-Request-Id": record["request_id"]}) as response:
                status = response.status
                if status != 200:
                    error = (await response.text())[:500]
                else:
                    async for raw in response.content:
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        if event.get("usage"):
                            usage = event["usage"]
                        choices = event.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            if first is None and (delta.get("content") or
                                                  delta.get("reasoning_content")):
                                first = time.perf_counter()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()
    output_tokens = int(usage.get("completion_tokens") or 0)
    ttft = None if first is None else first - started
    tpot = (None if first is None or output_tokens <= 1 else
            (finished - first) / (output_tokens - 1))
    return {
        "request_id": record["request_id"],
        "prompt_tokens": record["actual_prompt_tokens"],
        "output_tokens": output_tokens,
        "queue_seconds": started - queued,
        "latency_seconds": finished - started,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "http_status": status,
        "success": error is None and status == 200 and output_tokens > 0,
        "error": error,
        "started_at": started,
        "finished_at": finished,
    }


async def run_benchmark(*, endpoint, model, manifest, requests, arrival_rate,
                        client_concurrency, timeout, seed, ttft_slo=None,
                        tpot_slo=None, metrics_url=None):
    import aiohttp

    records = read_manifest(manifest)
    if requests > len(records):
        raise ValueError("request count exceeds the frozen manifest")
    records = records[:requests]
    rng = random.Random(seed)
    offsets = []
    elapsed = 0.0
    for _ in records:
        offsets.append(elapsed)
        if arrival_rate > 0:
            elapsed += rng.expovariate(arrival_rate)
    origin = time.perf_counter() + 0.25
    metrics_url = metrics_url or endpoint.split("/v1/", 1)[0] + "/metrics"
    semaphore = asyncio.Semaphore(client_concurrency)
    connector = aiohttp.TCPConnector(limit=client_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        before = await _metrics(session, metrics_url)
        stop = asyncio.Event()
        telemetry = []
        poller = asyncio.create_task(_poll_metrics(
            session, metrics_url, origin, stop, telemetry))
        rows = await asyncio.gather(*[
            _request(
                session, semaphore, url=endpoint, model=model, record=record,
                launch_at=origin + offset, timeout=timeout)
            for record, offset in zip(records, offsets, strict=True)
        ])
        stop.set()
        await poller
        after = await _metrics(session, metrics_url)
    successful = [row for row in rows if row["success"]]
    wall = max(row["finished_at"] for row in rows) - min(
        row["started_at"] for row in rows)
    good = [row for row in successful
            if (ttft_slo is None or row["ttft_seconds"] <= ttft_slo)
            and (tpot_slo is None or row["tpot_seconds"] <= tpot_slo)]
    return {
        "config": {
            "requests": requests,
            "arrival_rate": arrival_rate,
            "client_concurrency": client_concurrency,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": records[0]["max_tokens"],
            "ttft_slo": ttft_slo,
            "tpot_slo": tpot_slo,
        },
        "summary": {
            "wall_seconds": wall,
            "success_rate": len(successful) / len(rows),
            "request_throughput": len(successful) / wall,
            "output_token_throughput": sum(
                row["output_tokens"] for row in successful) / wall,
            "slo_goodput": len(good) / wall,
            "latency_seconds": _percentiles([
                row["latency_seconds"] for row in successful]),
            "ttft_seconds": _percentiles([
                row["ttft_seconds"] for row in successful]),
            "tpot_seconds": _percentiles([
                row["tpot_seconds"] for row in successful
                if row["tpot_seconds"] is not None]),
            "prompt_tokens": _percentiles([
                row["prompt_tokens"] for row in rows]),
        },
        "engine": {
            "counter_delta": {
                name: after.get(name, 0.0) - before.get(name, 0.0)
                for name in _TELEMETRY_NAMES if name.endswith("_total")
            },
            "max_running": max((row.get("vllm:num_requests_running", 0.0)
                                for row in telemetry), default=0.0),
            "max_waiting": max((row.get("vllm:num_requests_waiting", 0.0)
                                for row in telemetry), default=0.0),
            "max_kv_cache_usage": max((row.get("vllm:kv_cache_usage_perc", 0.0)
                                       for row in telemetry), default=0.0),
            "telemetry": telemetry,
        },
        "records": rows,
    }
