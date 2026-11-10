#!/usr/bin/env python3
"""真实 27B：K32 反馈、末层更新、下一轮生效，以及调度器自动在线更新。"""

import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "head-dir", "learner-head-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--learner-device", type=int)
    p.add_argument("--last-layers", type=int,choices=(1,2,3,4),default=1)
    p.add_argument("--resume-online", type=Path, help="repeat the closed loop from a saved optimizer state")
    p.add_argument("--replay-blocks", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    if args.output.exists():
        p.error("new output required")
    import torch
    from blockspec.pypto_qwen38.baseline import native_module
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.online import OnlineConfig
    from blockspec.pypto_qwen38.target import NativeTarget
    from blockspec.pypto_qwen38.greedy import verify_transaction
    from validate_qwen38_pypto_flow import collect
    from validate_qwen38_pypto_neural import resident_checksum, history_checksum
    torch.set_num_threads(4)
    engine, request = None, None
    report = dict(passed=False, cases=[])
    try:
        head = HeadSource(args.model, args.head_dir)
        engine = make_engine(args.source, args.model, args.device, head, head_transfer="packed",
            attention_backend="paged", group_size=4, gdn_backend="vector",
            online_config=OnlineConfig(last_layers=args.last_layers,stride=args.replay_blocks, replay_blocks=args.replay_blocks,
                raw_head_dir=str(args.learner_head_dir),
                learner_device=args.learner_device, resume_dir=None if args.resume_online is None else str(args.resume_online),
                checkpoint_dir=str(args.output.with_suffix(".updates"))))
        engine.start()
        native, draft = engine.runner.native, engine.runner.draft
        learner = draft.learner
        def audit_zero(optimizer):
            if args.last_layers == 1:
                return
            original = optimizer.zero_grad
            def zero():
                original()
                norm,_ = optimizer.statistics(optimizer.candidate[0])
                assert norm == 0., ("gradient buffer not fully cleared",norm)
                report.setdefault("gradient_zero_checks",[]).append(dict(step=optimizer.steps,passed=True))
            optimizer.zero_grad = zero
        audit_zero(learner.optimizer)
        initial_version = draft.weights.version
        if args.resume_online is not None:
            assert initial_version == learner.optimizer.steps > 0
            report["cases"].append(dict(name="cold_start_resume", version=initial_version, passed=True))
        schedule = native_module(args.source, "schedule")
        request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False), 40, frozenset())
        native.admit(request)
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position:position+16])
            anchor = native.run_batch([schedule.BatchSlot(request, tokens, position, "prefill", True)])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        initial = target.snapshot()
        expected, token = [], anchor
        for _ in range(32):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(initial)
        previous, previous_version = None, None
        for iteration in range(2 * args.replay_blocks):
            target.restore(initial)
            before = resident_checksum(native)
            candidates = draft.propose(request, anchor, 32)
            logits = draft.read_logits(31).clone()
            assert resident_checksum(native) == before, "draft mutated target history"
            if previous is not None and previous_version != draft.weights.version:
                assert not torch.equal(logits, previous), "published head did not change the next draft"
            previous, previous_version = logits, draft.weights.version
            collector = learner.current
            checked_target = NativeTarget(native, request, schedule.BatchSlot, collector=collector)
            result = verify_transaction(checked_target, anchor, candidates, budget=32)
            assert list(result.tokens) == expected[:len(result.tokens)]
            history = history_checksum(native, request)
            start = time.monotonic()
            update = learner.observe(result)
            assert history_checksum(native, request) == history, "training modified target history"
            row = dict(name="k32_online_round", iteration=iteration, accepted=result.accepted,
                       target_history_unchanged=True, greedy_matches=True,
                       version=draft.weights.version, passed=True)
            if (iteration + 1) % args.replay_blocks == 0:
                assert update["version"] == initial_version + (iteration + 1) // args.replay_blocks
                assert update["gradient_norm"] > 0
                handle = learner.workspace.values[learner.graph.logits.name]
                replay_logits = torch.empty((32, 248320), dtype=torch.bfloat16)
                learner.trainer.copy_from(replay_logits.data_ptr(), handle.data_ptr, replay_logits.numel()*2)
                replay_error = float((replay_logits[:31].float()-logits.float()).norm()/logits.float().norm())
                assert replay_error < .02, ("training suffix differs from serving forward", replay_error)
                row.update(replay_logits_relative_l2=replay_error,
                           update_seconds=time.monotonic()-start, **update)
            else:
                assert update is None and draft.weights.version == previous_version
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
        native.release(request)
        request = None
        del initial
        engine.scheduler.runner = native
        baseline = collect(engine, 3, 1200)
        deadline = time.monotonic()+30
        while engine.scheduler.counts()["active"]:
            if time.monotonic() > deadline:
                raise TimeoutError("baseline request not released")
            time.sleep(.01)
        engine.scheduler.runner = engine.runner
        old_version = draft.weights.version
        for _ in range(args.replay_blocks):
            actual = collect(engine, 3, 1200)
            assert actual == baseline
        assert draft.weights.version > old_version
        report["cases"].append(dict(name="scheduler_online", greedy_matches=True,
            before_version=old_version, after_version=draft.weights.version, passed=True))
        deadline = time.monotonic()+30
        while engine.scheduler.counts()["active"]:
            if time.monotonic() > deadline:
                raise TimeoutError("online request not released")
            time.sleep(.01)
        if args.last_layers == 1:
            from blockspec.pypto_qwen38.online_checkpoint import save, restore
            from blockspec.pypto_qwen38.online_optimizer import DeviceOptimizer
        else:
            from blockspec.pypto_qwen38.online_suffix_checkpoint import save, restore
            from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer as DeviceOptimizer
        checkpoint = args.output.with_suffix(".checkpoint")
        version = learner.optimizer.steps
        save(learner.optimizer, checkpoint, learner.head_identity)
        # 停止读旧 bank 后释放旧优化器，再从文件创建全新的状态，避免双份状态挤占 HBM。
        draft.weights.overrides.clear()
        draft.weights.version = 0
        learner.optimizer.close()
        learner.optimizer = DeviceOptimizer(learner.optimization, learner.trainer)
        audit_zero(learner.optimizer)
        restore(learner.optimizer, learner.publisher, checkpoint, learner.head_identity)
        assert learner.optimizer.steps == draft.weights.version == version
        for _ in range(args.replay_blocks):
            response = http_round(engine, args.source)
            assert response["token_ids"] == baseline
        assert draft.weights.version > version
        report["cases"].append(dict(name="checkpoint_restore_http_update", restored_version=version,
            final_version=draft.weights.version, **response, passed=True))
        report["memory_after_requests"] = dict(serving=native.worker.device_memory_info(),
            learner=learner.trainer.device_memory_info(), learner_device=args.learner_device)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if engine is not None:
                if request is not None:
                    engine.runner.native.release(request)
                engine.scheduler.runner = engine.runner
                engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")


def http_round(engine, source):
    import socket
    import threading
    from urllib.request import Request, urlopen
    import uvicorn
    from blockspec.pypto_qwen38.baseline import native_module
    api = native_module(source, "api")
    api.app.state.engine = engine
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(api.app, lifespan="off", log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        deadline = time.monotonic()+30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise TimeoutError("HTTP validation did not start")
            time.sleep(.01)
        with urlopen(address+"/health", timeout=30) as response:
            assert json.load(response)["status"] == "ready"
        payload = json.dumps(dict(prompt="Hello", max_tokens=3, temperature=0)).encode()
        with urlopen(Request(address+"/v1/completions", data=payload,
                             headers={"Content-Type": "application/json"}), timeout=1200) as response:
            result = json.load(response)
        assert result["usage"]["completion_tokens"] == 3
        return dict(completion_tokens=3, text=result["choices"][0]["text"],
                    token_ids=result["choices"][0]["token_ids"], health_ready=True)
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        sock.close()
        if thread.is_alive():
            raise TimeoutError("HTTP validation did not stop")


if __name__ == "__main__":
    main()
