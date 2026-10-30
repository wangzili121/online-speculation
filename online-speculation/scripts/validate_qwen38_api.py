#!/usr/bin/env python3
"""Exercise the actual vLLM OpenAI API, including streaming and cancellation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.error
import urllib.request


class Client:
    def __init__(self, base_url, model):
        self.base_url, self.model = base_url.rstrip("/"), model

    def request(self, route, payload=None):
        request = urllib.request.Request(
            self.base_url + route, headers={"Content-Type": "application/json"},
            data=None if payload is None else json.dumps(payload).encode())
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    def payload(self, prompt, length, **kwargs):
        return dict(model=self.model, prompt=prompt, max_tokens=length,
                    temperature=0, ignore_eos=True, **kwargs)

    def complete(self, payload):
        result = self.request("/v1/completions", payload)
        assert result["usage"]["completion_tokens"] == payload["max_tokens"], result
        return result["choices"][0]["text"], result["usage"]

    def stream(self, payload, *, cancel=False):
        request = urllib.request.Request(
            self.base_url + "/v1/completions", headers={"Content-Type": "application/json"},
            data=json.dumps(dict(payload, stream=True)).encode())
        text, chunks, done = "", 0, False
        with urllib.request.urlopen(request, timeout=300) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                part = json.loads(data)["choices"][0]["text"]
                text += part
                chunks += bool(part)
                if cancel and part:
                    break
        assert chunks > 0 and (cancel or done)
        return text, chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:29038")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-rounds", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.recovery_rounds <= 100:
        parser.error("--recovery-rounds must be in 1..100")
    client = Client(args.base_url, args.model)
    expected = {item["case"]: item["text"] for item in json.loads(args.baseline.read_text())["outputs"]}
    checks = []

    def record(name, **fields):
        checks.append(dict(case=name, passed=True, **fields))
        print(json.dumps(checks[-1]), flush=True)

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=10) as response:
        assert response.status == 200
    assert args.model in {item["id"] for item in client.request("/v1/models")["data"]}
    record("health_and_model")
    hello = client.payload("Hello", 8)
    text, usage = client.complete(hello)
    assert text == expected["short-greedy"], text
    record("greedy", usage=usage)
    sampled = dict(hello, temperature=.7, top_k=20, top_p=.9, seed=7)
    text, usage = client.complete(sampled)
    record("sampling_topk", usage=usage, text=text)
    text, chunks = client.stream(hello)
    assert text == expected["short-greedy"], text
    record("streaming", chunks=chunks)

    count = client.payload("Count the integers from 1 to 20:", 48)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(client.complete, (hello, count)))
    assert [result[0] for result in results] == [expected["short-greedy"], expected["reject-rounds"]]
    record("queued_requests", requests=2, engine_max_num_seqs=1)

    client.stream(dict(hello, max_tokens=128), cancel=True)
    text, _ = client.complete(hello)
    assert text == expected["short-greedy"], text
    record("cancel_then_new_request")

    long_prompt = "A tree has green leaves. " * 300 + "Summarize:"
    text, usage = client.complete(client.payload(long_prompt, 8))
    assert usage["prompt_tokens"] > 1536 and text, usage
    record("cross_page_prefill", usage=usage)
    text, _ = client.complete(hello)
    assert text == expected["short-greedy"], text
    record("history_reset_after_long_request")
    for index in range(args.recovery_rounds):
        client.stream(dict(hello, max_tokens=128), cancel=True)
        try:
            client.request("/v1/completions", dict(hello, max_tokens=-1))
        except urllib.error.HTTPError as error:
            if error.code not in (400, 422):
                raise
        else:
            raise AssertionError("invalid max_tokens was accepted")
        text, _ = client.complete(hello)
        if text != expected["short-greedy"]:
            raise AssertionError("cancelled or invalid request contaminated the next request")
        record("cancel_invalid_recover", round=index)
    args.output.write_text(json.dumps(dict(model=args.model, passed=True, checks=checks),
                                      ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
