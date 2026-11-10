"""用真实反馈测生产更新链路：双卡传输、反向、Adam 与参数发布。"""

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


class PublicationTarget:
    """保留生产双缓冲发布协议；本工具不加载完整推理模型。"""

    def __init__(self):
        self.version = 0
        self.layers = {}

    def publish(self, layer, values, *, expected_version):
        if expected_version != self.version:
            raise AssertionError("stale parameter publication")
        self.layers[layer] = values
        self.version += 1

    def publish_many(self, values, *, expected_version):
        if expected_version != self.version:
            raise AssertionError("stale suffix publication")
        self.layers = values
        self.version += 1


class Timings:
    def __init__(self):
        self.values = defaultdict(float)
        self.publishing = False

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] += time.perf_counter() - start

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def call(*args, **kwargs):
            with self.measure(label):
                return original(*args, **kwargs)

        setattr(obj, name, call)


def run(a):
    import torch
    import pypto
    from safetensors.torch import load_file
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.online import OnlineConfig, OnlineLearner, FeedbackBlock
    from blockspec.pypto_qwen38.online_suffix import SuffixLearner
    from blockspec.pypto_qwen38.online_bridge import TensorBridge, HeadPublisher, SuffixPublisher
    from blockspec.pypto_qwen38.online_optimizer import DeviceOptimizer, OptimizerPrograms
    from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer, SuffixOptimizerPrograms
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.online_feedback_programs import make_copy, make_loss_sum
    from blockspec.pypto_qwen38.loading import DeviceLoader
    from diagnose_qwen38_pypto_training import read_tensor, state_views, execution_views

    torch.set_num_threads(4)
    runtime_identity(pypto)
    source = json.loads(a.capture_report.read_text())
    if not source.get("capture_ready"):
        raise ValueError("immutable real feedback snapshot required")
    old = Path(source["directory"])
    n = source["last_layers"]
    opts = OnlineConfig(objective="kl", last_layers=n, stride=8, replay_blocks=8,
                        learner_device=a.learner_device)
    learner = (OnlineLearner if n == 1 else SuffixLearner)(None, opts)
    graph = learner.graph
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.learner_device], num_sub_workers=0))
    serving_config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    meta = lambda shape, dtype=torch.float32: torch.empty(shape, dtype=dtype, device="meta")
    training = TrainingPrograms(graph, config, combined=True, accumulate_gradients=n == 1,
                               external_gradients=n > 1, chunk_size=24 if n > 1 else None)
    optim = (OptimizerPrograms if n == 1 else SuffixOptimizerPrograms)(
        graph, config, clip_norm=1., **({"combined": True} if n == 1 else {}))
    loss_program = make_loss_sum().compile(meta((32, 8)), meta((1, 8)), config=config)
    copy_program = make_copy(32, 64).compile(meta((32, 64), torch.bfloat16),
        meta((32, 64), torch.bfloat16), config=serving_config)
    learner.bridge = TensorBridge()
    inherited = [learner.bridge.scratch] if a.shared_transfer else []
    trainer = serving = workspace = optimizer = None
    handles = []
    report = dict(completed=False, last_layers=n, updates=a.updates, warmup=a.warmup,
        source_snapshot_sha256=hashlib.sha256(a.capture_report.read_bytes()).hexdigest(),
        shared_transfer=a.shared_transfer, scope="real frozen feedback; production two-device update and publication; no full serving model",
        runs={})

    def persist():
        a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def alloc(worker, value):
        value = value.contiguous()
        result = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)
        handles.append((worker, result))
        return result

    try:
        serving = DistributedWorker([copy_program], config=serving_config, inherited_host_tensors=inherited)
        trainer = DistributedWorker([*training.programs.values(), *optim.programs(), loss_program], config=config,
                                    inherited_host_tensors=inherited)
        workspace = TrainingWorkspace(training, trainer)
        optimizer = (DeviceOptimizer if n == 1 else SuffixOptimizer)(optim, trainer)
        learner.trainer, learner.native = trainer, SimpleNamespace(worker=serving)
        learner.dedicated, learner.training_config = True, config
        learner.workspace, learner.optimizer = workspace, optimizer
        learner.shared = {k: alloc(trainer, v) for k, v in load_file(str(old / "shared.safetensors")).items()}
        learner.replay = deque()
        for i, desc in enumerate(source["blocks"]):
            values = load_file(str(old / f"block-{i:02d}.safetensors"))
            values = {k: v for k, v in values.items() if k in graph.inputs
                      and k not in learner.shared and k not in graph.gradients}
            block = FeedbackBlock(learner, {k: alloc(serving, v) for k, v in values.items()})
            # 不监督的教师行填 NaN，验证部分传输不会偷读旧数据。
            block.training_inputs = {k: alloc(trainer, torch.full_like(v, float("nan"))) for k, v in values.items()}
            block.past, block.width, block.valid = desc["past"], desc["width"], desc["valid"]
            block.rows = block.width
            learner.replay.append(block)
        learner.loss_program, learner.loss_device = loss_program, alloc(trainer, torch.empty((1, 8)))
        learner.loss_host = torch.empty((1, 8))
        if n == 1:
            learner.serving_banks = [tuple(alloc(serving, torch.empty(spec.shape,
                dtype=torch.float32 if "gamma" in name else torch.bfloat16)) for name in
                ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
                for spec in (graph.inputs[name],)) for _ in range(2)]
        else:
            learner.serving_banks = [(alloc(serving, torch.empty(optimizer.layout.shape, dtype=torch.bfloat16)),
                alloc(serving, torch.empty(optimizer.layout.special_shape))) for _ in range(2)]
        target = PublicationTarget()
        learner.draft = SimpleNamespace(weights=target)
        learner.publisher = (HeadPublisher if n == 1 else SuffixPublisher)(learner, target)
        initial = load_file(str(old / "initial.master.safetensors"))
        timer = Timings()
        timer.wrap(optimizer, "zero_grad", "zero_grad")
        timer.wrap(workspace, "run", "forward_backward")
        timer.wrap(optimizer, "statistics", "gradient_and_candidate_checks")
        if n > 1:
            timer.wrap(optimizer, "accumulate", "accumulate")
        original_run = optimizer.run

        def opt_run(program, *args):
            # statistics 自己计时；这里只统计 Adam / 执行参数转换。
            entries = [optim.candidate_update] if n == 1 else [optim.adam, optim.render]
            if any(program is p for p in entries):
                with timer.measure("adam_render"):
                    return original_run(program, *args)
            return original_run(program, *args)

        optimizer.run = opt_run
        original_copy = learner.bridge.copy

        def bridge(*args, **kwargs):
            with timer.measure("parameter_transfer" if timer.publishing else "feedback_transfer"):
                return original_copy(*args, **kwargs)

        learner.bridge.copy = bridge
        publish_name = "publish" if n == 1 else "publish_flat"
        original_publish = getattr(learner.publisher, publish_name)

        def publish(*args, **kwargs):
            timer.publishing = True
            try:
                return original_publish(*args, **kwargs)
            finally:
                timer.publishing = False

        setattr(learner.publisher, publish_name, publish)
        original_transfer = FeedbackBlock.transfer
        mode = None

        def transfer(block):
            if mode == "full_teacher":
                for name, value in block.inputs.items():
                    learner.bridge.copy(serving, value, trainer, block.training_inputs[name])
            else:
                original_transfer(block)

        # 只改变本基准的实例；生产方法与数值计算不作替换。
        from types import MethodType
        for block in learner.replay:
            block.transfer = MethodType(transfer, block)
        for mode in ("valid_teacher", "full_teacher"):
            optimizer.steps = optimizer.index = learner.updates = target.version = 0
            learner.publisher.active = 0
            if n == 1:
                loader = DeviceLoader(optim.loader, trainer)
                try:
                    for name, states in optimizer.state.items():
                        value = initial[name]
                        trainer.copy_to(states[0].data_ptr, value.data_ptr(), value.numel() * 4)
                        for v in (*states[1:], *optimizer.candidate[name]):
                            loader.zero(v)
                finally:
                    loader.close()
                optimizer.render(0, optimizer.state)
            else:
                for value in (*optimizer.state, *optimizer.candidate):
                    original_run(optim.zero, value)
                for entry in optimizer.layout.entries:
                    value = initial[entry.parameter.name]
                    trainer.copy_to(optimizer.state[0].data_ptr, value.data_ptr(), value.numel() * 4,
                                    dst_offset=entry.offset * 4)
                optimizer.render(0, optimizer.state[0])
            for block in learner.replay:
                value = torch.full(tuple(block.training_inputs["teacher"].shape), float("nan"), dtype=torch.bfloat16)
                trainer.copy_to(block.training_inputs["teacher"].data_ptr, value.data_ptr(), value.numel() * 2)
            rows = []
            for i in range(a.warmup + a.updates):
                timer.values.clear()
                before = learner.bridge.bytes
                start = time.perf_counter()
                result = learner.update()
                total = time.perf_counter() - start
                row = dict(total=total, phases=dict(timer.values), transferred_bytes=learner.bridge.bytes - before,
                           loss=result["loss"], gradient_norm=result["gradient_norm"], version=target.version)
                row["unattributed"] = total - sum(timer.values.values())
                rows.append(row)
                print(json.dumps(dict(event="update_profile", mode=mode, iteration=i, **row)), flush=True)
            digests = {}
            for label, values in [(label, state_views(optimizer, c, n))
                    for c, label in enumerate(("master", "first", "second"))] + [("execution", execution_views(optimizer, n))]:
                h = hashlib.sha256()
                for name, value in sorted(values.items()):
                    h.update(name.encode())
                    h.update(read_tensor(trainer, value).view(torch.uint8).numpy().tobytes())
                digests[label] = h.hexdigest()
            measured = rows[a.warmup:]
            report["runs"][mode] = dict(rows=rows, state_sha256=digests,
                median_seconds=statistics.median(r["total"] for r in measured),
                median_phases={k: statistics.median(r["phases"].get(k, 0) for r in measured)
                               for k in timer.values})
            persist()
        candidate, baseline = (report["runs"][k] for k in ("valid_teacher", "full_teacher"))
        report["identical_optimizer_and_execution"] = candidate["state_sha256"] == baseline["state_sha256"]
        report["identical_losses"] = [r["loss"] for r in candidate["rows"]] == [r["loss"] for r in baseline["rows"]]
        if not report["identical_optimizer_and_execution"] or not report["identical_losses"]:
            raise AssertionError("teacher prefix transfer changed training results")
        report["completed"] = True
    finally:
        if workspace is not None:
            workspace.close()
        if optimizer is not None:
            optimizer.close()
        for worker, handle in reversed(handles):
            worker.free_tensor(handle)
        for worker in (trainer, serving):
            if worker is not None:
                worker.close()
        report["clean_shutdown"] = True
        persist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--learner-device", type=int, default=1)
    p.add_argument("--source", type=Path)
    p.add_argument("--shared-transfer", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--updates", type=int, default=3)
    args = p.parse_args()
    if args.device == args.learner_device or args.warmup < 0 or args.updates < 1:
        p.error("distinct devices and positive measured updates required")
    run(args)
