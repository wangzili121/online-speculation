"""真实形状线性前向 / VJP 的归约 tile 对照；独立 Torch 结果作为参考。"""

import argparse
import json
from pathlib import Path
import statistics
import time


def run(a):
    import torch
    import pypto
    from safetensors import safe_open
    from safetensors.torch import load_file
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.training_programs import make_linear
    from diagnose_qwen38_pypto_training import read_tensor

    torch.set_num_threads(8)
    runtime_identity(pypto)
    manifest = json.loads(a.capture_report.read_text())
    if manifest["last_layers"] != 1 or not manifest.get("capture_ready"):
        raise ValueError("real last-1 feedback required")
    directory = Path(manifest["directory"])
    feedback = load_file(str(directory / "block-00.safetensors"))
    valid = manifest["blocks"][0]["valid"]
    generator = torch.Generator().manual_seed(7301)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    meta = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
    cases = [("lm_dx", "lm_head", False), ("mlp_down_dx", "mlp_down", False),
             ("mlp_gate_dx", "mlp_gate", False), ("mlp_gate_forward", "mlp_gate", True)]
    programs = {}
    with safe_open(directory / "shared.safetensors", framework="pt") as file:
        for name, weight, transpose in cases:
            shape = tuple(file.get_slice(weight).get_shape())
            k, n = shape[::-1] if transpose else shape
            for tile in a.tiles:
                entry = make_linear((32, k), shape, transpose_b=transpose, reduction_limit=tile)
                programs[name, tile] = entry.compile(meta((32, k)), meta(shape), meta((32, n)), config=config)
    report = dict(completed=False, scope="real model shapes and frozen weights; isolated synchronous worker runs",
                  repetitions=a.repeats, results={})
    worker = None

    def persist():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        worker = DistributedWorker(list(programs.values()), config=config)
        with safe_open(directory / "shared.safetensors", framework="pt") as file:
            for name, weight, transpose in cases:
                b = file.get_tensor(weight)
                k, n = b.shape[::-1] if transpose else b.shape
                if name == "lm_dx":
                    x = torch.zeros((32, k), dtype=torch.bfloat16)
                    student = feedback["observed_logits"][:valid].float()
                    teacher = feedback["teacher"][:valid].float()
                    x[:valid] = ((student.softmax(-1) - teacher.softmax(-1)) /
                                 manifest["blocks"][0]["normalizer"]).to(torch.bfloat16)
                else:
                    x = (torch.randn((32, k), generator=generator) * .01).to(torch.bfloat16)
                    x[valid:] = 0
                reference = (x.float() @ (b.float().T if transpose else b.float())).to(torch.bfloat16)
                x_dev = worker.alloc_tensor(tuple(x.shape), x.dtype, init=x)
                b_dev = worker.alloc_tensor(tuple(b.shape), b.dtype, init=b)
                output = worker.alloc_tensor((32, n), torch.bfloat16)
                try:
                    results = {str(t): dict(seconds=[]) for t in a.tiles}
                    for tile in a.tiles:
                        for _ in range(2):
                            worker.run(programs[name, tile], x_dev, b_dev, output, config=config)
                        actual = read_tensor(worker, output).float()
                        delta = actual - reference.float()
                        rel = float(delta.norm() / reference.float().norm().clamp_min(1e-20))
                        inactive_zero = bool(torch.equal(actual[valid:], torch.zeros_like(actual[valid:])))
                        results[str(tile)].update(relative_l2=rel, max_absolute=float(delta.abs().max()),
                            inactive_rows_zero=inactive_zero)
                        if rel > .01 or not inactive_zero or not torch.isfinite(actual).all():
                            raise AssertionError(f"{name}, tile {tile}: relative L2 {rel}, inactive {inactive_zero}")
                    # 交错顺序，避免所有大 tile 都在后一轮测量。
                    for repeat in range(a.repeats):
                        order = a.tiles if repeat % 2 == 0 else a.tiles[::-1]
                        for tile in order:
                            start = time.perf_counter()
                            worker.run(programs[name, tile], x_dev, b_dev, output, config=config)
                            results[str(tile)]["seconds"].append(time.perf_counter() - start)
                    for result in results.values():
                        result["median_seconds"] = statistics.median(result["seconds"])
                    report["results"][name] = dict(a_shape=list(x.shape), b_shape=list(b.shape),
                        transpose_b=transpose, tiles=results)
                    print(json.dumps(dict(event="linear_profile", name=name, **report["results"][name])), flush=True)
                    persist()
                finally:
                    for value in (output, b_dev, x_dev):
                        worker.free_tensor(value)
                del reference, b, x
        report["completed"] = True
    finally:
        if worker is not None:
            worker.close()
        report["clean_shutdown"] = True
        persist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--source", type=Path)
    p.add_argument("--tiles", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--repeats", type=int, default=9)
    args = p.parse_args()
    if args.repeats < 3:
        p.error("at least three measurements required")
    run(args)
