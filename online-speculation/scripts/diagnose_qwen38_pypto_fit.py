"""单卡复用真实反馈，执行生产 learner 的反向、累积和 Adam 更新。"""

from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


class ResidentPublisher:
    """本测试的执行参数留在学习卡；跨设备发布由整模型路径另外验收。"""

    def __init__(self):
        self.version = 0
        self.layers = {}

    def publish(self, layer, values, *, expected_version):
        if expected_version != self.version:
            raise AssertionError("stale parameter version")
        self.layers[layer] = values
        self.version += 1

    def publish_flat(self, layout, values, *, expected_version):
        if expected_version != self.version:
            raise AssertionError("stale parameter version")
        self.layers = layout.banks(*values)
        self.version += 1


def same_device_copy(source_worker, source, target_worker, target, **kwargs):
    if source_worker is not target_worker or source is not target or kwargs:
        raise AssertionError("resident feedback must keep the same tensor")


def run(a):
    import torch
    import pypto
    from safetensors.torch import load_file, save_file
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.online import OnlineConfig, OnlineLearner, FeedbackBlock
    from blockspec.pypto_qwen38.online_suffix import SuffixLearner
    from blockspec.pypto_qwen38.online_optimizer import (
        DeviceOptimizer,
        OptimizerPrograms,
    )
    from blockspec.pypto_qwen38.online_suffix_optimizer import (
        SuffixOptimizer,
        SuffixOptimizerPrograms,
    )
    from blockspec.pypto_qwen38.training_executor import (
        TrainingPrograms,
        TrainingWorkspace,
    )
    from blockspec.pypto_qwen38.online_feedback_programs import make_loss_sum
    from blockspec.pypto_qwen38.loading import DeviceLoader
    from benchmark_qwen38_pypto_cold import replay_loss
    from diagnose_qwen38_pypto_training import (
        read_tensor,
        state_views,
        flat_views,
        execution_views,
    )

    torch.set_num_threads(4)
    runtime_identity(pypto)
    source = json.loads(a.capture_report.read_text())
    old = Path(source["directory"])
    n = source["last_layers"]
    if not source.get("capture_ready"):
        raise ValueError("complete immutable feedback snapshot required")
    output = a.output.with_suffix(".tensors")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("shared.safetensors", "initial.master.safetensors"):
        (output / name).symlink_to(old / name)
    report = {
        **source,
        "completed": False,
        "phase": "fixed_saved_feedback",
        "learning_rate": a.learning_rate,
        "updates": [],
        "directory": str(output),
        "arguments": {
            **source["arguments"],
            "output": str(a.output),
            "learning_rate": a.learning_rate,
            "steps": a.steps,
        },
        "input_snapshot_sha256": hashlib.sha256(
            a.capture_report.read_bytes()
        ).hexdigest(),
        "publication_scope": "resident learner banks; no full serving engine in this test",
    }
    report.pop("first_step", None)
    options = OnlineConfig(
        objective="kl",
        last_layers=n,
        stride=8,
        replay_blocks=8,
        learning_rate=a.learning_rate,
        learner_device=a.device if n > 1 else None,
    )
    learner = (OnlineLearner if n == 1 else SuffixLearner)(None, options)
    graph = learner.graph
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0),
    )
    training = TrainingPrograms(
        graph,
        config,
        combined=True,
        accumulate_gradients=n == 1,
        external_gradients=n > 1,
        chunk_size=24 if n > 1 else None,
    )
    optim = (OptimizerPrograms if n == 1 else SuffixOptimizerPrograms)(
        graph, config, clip_norm=1.0, **({"combined": True} if n == 1 else {})
    )
    meta = lambda shape: torch.empty(shape, dtype=torch.float32, device="meta")
    loss_program = make_loss_sum().compile(meta((32, 8)), meta((1, 8)), config=config)
    worker = workspace = optimizer = None
    handles = []

    def persist():
        a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def save(name, values):
        save_file(
            {k: read_tensor(worker, v) for k, v in values.items()}, str(output / name)
        )

    try:
        worker = DistributedWorker(
            [*training.programs.values(), *optim.programs(), loss_program],
            config=config,
        )
        workspace = TrainingWorkspace(training, worker)
        optimizer = (DeviceOptimizer if n == 1 else SuffixOptimizer)(optim, worker)
        initial = load_file(str(old / "initial.master.safetensors"))
        if n == 1:
            loader = DeviceLoader(optim.loader, worker)
            try:
                for name, states in optimizer.state.items():
                    value = initial[name]
                    worker.copy_to(
                        states[0].data_ptr, value.data_ptr(), value.numel() * 4
                    )
                    for v in (*states[1:], *optimizer.candidate[name]):
                        loader.zero(v)
            finally:
                loader.close()
            optimizer.render(0, optimizer.state)
        else:
            for value in (*optimizer.state, *optimizer.candidate):
                optimizer.run(optim.zero, value)
            for entry in optimizer.layout.entries:
                value = initial[entry.parameter.name]
                worker.copy_to(
                    optimizer.state[0].data_ptr,
                    value.data_ptr(),
                    value.numel() * 4,
                    dst_offset=entry.offset * 4,
                )
            optimizer.render(0, optimizer.state[0])

        def alloc(value):
            result = worker.alloc_tensor(
                tuple(value.shape), value.dtype, init=value.contiguous()
            )
            handles.append(result)
            return result

        learner.shared = {
            k: alloc(v) for k, v in load_file(str(old / "shared.safetensors")).items()
        }
        learner.replay = deque()
        for i, desc in enumerate(source["blocks"]):
            values = load_file(str(old / f"block-{i:02d}.safetensors"))
            dynamic = {
                k: alloc(v)
                for k, v in values.items()
                if k in graph.inputs
                and k not in learner.shared
                and k not in graph.gradients
            }
            block = FeedbackBlock(learner, dynamic)
            block.past, block.width, block.valid = (
                desc["past"],
                desc["width"],
                desc["valid"],
            )
            block.rows = block.width
            learner.replay.append(block)
        learner.trainer = worker
        learner.native = SimpleNamespace(worker=worker)
        learner.dedicated = False
        learner.training_config = config
        learner.workspace = workspace
        learner.optimizer = optimizer
        learner.bridge = SimpleNamespace(copy=same_device_copy)
        learner.loss_program = loss_program
        learner.loss_device = alloc(torch.empty((1, 8), dtype=torch.float32))
        learner.loss_host = torch.empty((1, 8), dtype=torch.float32)
        learner.publisher = ResidentPublisher()
        learner.draft = SimpleNamespace(weights=learner.publisher)
        count = 0
        original_run = workspace.run
        original_step = optimizer.step

        def execute(inputs, **kwargs):
            nonlocal count
            result = original_run(inputs, **kwargs)
            if learner.updates == 0:
                tensors = load_file(str(old / f"block-{count:02d}.safetensors"))
                logits = read_tensor(worker, workspace.values[graph.logits.name])
                tensors["observed_logits"] = logits[: kwargs["valid"]].clone()
                save_file(tensors, str(output / f"block-{count:02d}.safetensors"))
                if count == 0:
                    save(
                        "gradient-first.safetensors",
                        result if n == 1 else flat_views(optimizer, optimizer.scratch),
                    )
                count += 1
            return result

        workspace.run = execute

        def step(pub, rate):
            first = optimizer.steps == 0
            if first:
                save(
                    "gradient-sum.safetensors",
                    optimizer.accum
                    if n == 1
                    else flat_views(optimizer, optimizer.candidate[0]),
                )
            result = original_step(pub, rate)
            if first:
                for c, label in enumerate(("master", "first", "second")):
                    save(f"step1.{label}.safetensors", state_views(optimizer, c, n))
                save("step1.execution.safetensors", execution_views(optimizer, n))
                report["first_step"] = result
            return result

        optimizer.step = step
        for i in range(a.steps):
            result = learner.update()
            report["updates"].append(dict(step=learner.updates, **result))
            if i == 0:
                (output / "manifest.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            if i == 0 or (i + 1) % 8 == 0:
                print(
                    json.dumps(
                        dict(event="resident_fixed_batch", **report["updates"][-1])
                    ),
                    flush=True,
                )
                persist()
        report["final_loss"] = replay_loss(learner)
        save("final.master.safetensors", state_views(optimizer, 0, n))
        report["completed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if workspace is not None:
                workspace.close()
            if optimizer is not None:
                optimizer.close()
            if worker is not None:
                for handle in reversed(handles):
                    worker.free_tensor(handle)
                worker.close()
            report["clean_shutdown"] = True
        finally:
            persist()
