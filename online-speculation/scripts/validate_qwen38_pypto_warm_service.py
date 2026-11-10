#!/usr/bin/env python3
"""通过真实 HTTP 检查预热 head 的恢复、在线更新和 SSE 输出。"""

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from benchmark_qwen38_pypto_online import audit_warm_initialization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "checkpoint-report", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int, required=True)
    p.add_argument("--partial-teacher-capture", action="store_true")
    p.add_argument("--require-released-storage", action="store_true")
    p.add_argument("--save-final-checkpoint", action="store_true")
    a = p.parse_args()
    if a.output.exists() or a.device == a.learner_device:
        p.error("new report and two different devices required")
    source = json.loads(a.checkpoint_report.read_text())
    if not source.get("completed") or not source.get("clean_shutdown"):
        p.error("completed checkpoint experiment required")
    import uvicorn
    import serve_qwen38_pypto

    report = dict(completed=False, requests=[], checkpoint=source["checkpoint"],
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urlopen(base+path, timeout=10) as response:
            return json.load(response)

    def run(app, **kwargs):
        server = uvicorn.Server(uvicorn.Config(app, **kwargs))
        stopped = threading.Event()

        def exercise():
            learner = original_update = None
            try:
                deadline = time.monotonic()+1800
                while True:
                    if stopped.is_set(): raise RuntimeError("server stopped before readiness")
                    try:
                        if get("/health")["status"] == "ready": break
                    except (URLError, ConnectionError): pass
                    if time.monotonic()>deadline: raise TimeoutError("service readiness timed out")
                    time.sleep(.5)
                engine = app.state.engine
                learner = engine.runner.native.learner
                report["initialization"] = audit_warm_initialization(learner, source["checkpoint"]["path"])
                original_update = learner.update
                report["updates"] = []
                def update():
                    result = original_update()
                    report["updates"].append(dict(step=learner.updates, **result))
                    return result
                learner.update = update
                report["before"] = get("/metrics")
                if report["before"]["orthrus"]["verification_backend"] != "tiled":
                    raise AssertionError("service did not select the requested verifier")
                if a.require_released_storage:
                    released = report["before"]["orthrus"]["head_storage_release"]
                    if not released or released["released_bytes"] <= 0 or released["free_bytes_after"] <= released["free_bytes_before"]:
                        raise AssertionError("unused head storage was not released on the NPU")
                if report["before"]["orthrus"].get("partial_teacher_capture", False) != a.partial_teacher_capture:
                    raise AssertionError("service did not select the requested teacher capture mode")
                refs = source["evaluations"][-1]["dev"]["target_rows"][:2]
                prompts = {row["id"]: row for row in source["dataset"]["splits"]["dev"]}
                for index, reference in enumerate(refs):
                    payload = dict(prompt=prompts[reference["id"]]["prompt_ids"], max_tokens=64,
                                   temperature=0, stream=bool(index))
                    request = Request(base+"/v1/completions", json.dumps(payload).encode(),
                                      {"Content-Type": "application/json"})
                    tokens, done = [], False
                    started = time.monotonic()
                    with urlopen(request, timeout=900) as response:
                        if not index:
                            reply = json.load(response)
                            tokens = reply["choices"][0]["token_ids"]
                            done = True
                        else:
                            for raw in response:
                                line = raw.decode().strip()
                                if not line.startswith("data: "): continue
                                if line == "data: [DONE]": done=True; break
                                value = json.loads(line[6:])
                                if "error" in value: raise RuntimeError(value["error"])
                                for choice in value.get("choices", []): tokens.extend(choice.get("token_ids", []))
                    if not done or tokens != reference["tokens"][:64]:
                        raise AssertionError("HTTP/SSE tokens differ from the verified AR reference")
                    row = dict(id=reference["id"], stream=bool(index), tokens=tokens, greedy_equal=True,
                               seconds=time.monotonic()-started)
                    report["requests"].append(row)
                deadline = time.monotonic()+30
                while get("/health")["active"] or get("/health")["queued"]:
                    if time.monotonic()>deadline: raise TimeoutError("service did not release requests")
                    time.sleep(.1)
                report["after"] = get("/metrics")
                online = report["after"]["orthrus"]["online"]
                if not online["updates"] or online["version"] != online["updates"] or online["last_result"].get("skipped"):
                    raise AssertionError("online updates were not committed")
                if not all(math.isfinite(online["last_result"][key]) for key in ("loss", "gradient_norm")):
                    raise AssertionError("online update returned nonfinite statistics")
                if len(report["updates"]) != online["updates"]:
                    raise AssertionError("unaccounted service update")
                if a.save_final_checkpoint:
                    if source["arguments"]["last_layers"] == 1:
                        from blockspec.pypto_qwen38.online_checkpoint import save
                    else:
                        from blockspec.pypto_qwen38.online_suffix_checkpoint import save
                    directory = a.output.with_suffix(".checkpoint")
                    save(learner.optimizer, directory, learner.head_identity)
                    report["final_checkpoint"] = str(directory)
                report["verified"] = True
            except BaseException as error:
                report["error"] = repr(error)
            finally:
                if original_update is not None:
                    learner.update = original_update
                server.should_exit = True

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()
        try:
            server.run()
        finally:
            stopped.set()
            thread.join(timeout=30)
        if not report.get("verified") or thread.is_alive():
            raise RuntimeError(report.get("error", "HTTP verifier did not finish"))
        if app.state.engine.scheduler.thread.is_alive():
            raise RuntimeError("service scheduler remained active after shutdown")

    argv, original_run = sys.argv, uvicorn.run
    try:
        uvicorn.run = run
        sys.argv = [serve_qwen38_pypto.__file__, "--source", str(a.source), "--model", str(a.model),
                    "--device", str(a.device), "--learner-device", str(a.learner_device),
                    "--port", str(port), "--online", "--last-layers", str(source["arguments"]["last_layers"]),
                    "--update-stride", "8", "--replay-blocks", "8", "--learning-rate", "1e-5",
                    "--resume-online", source["checkpoint"]["path"], "--reset-online-optimizer",
                    "--share-ar-prefix", "--verification-backend", "tiled", "--attention-backend", "paged",
                    "--gdn-backend", "vector", "--draft-group-size", "4", "--head-transfer", "packed"]
        if a.partial_teacher_capture:
            sys.argv.append("--partial-teacher-capture")
        serve_qwen38_pypto.main()
        report.update(completed=True, clean_shutdown=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        sys.argv, uvicorn.run = argv, original_run
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(completed=True, requests=len(report["requests"]),
                          online=report["after"]["orthrus"]["online"])), flush=True)


if __name__ == "__main__":
    main()
