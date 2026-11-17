#!/usr/bin/env python3
"""Exercise the GLM Serving verifier using proposals with known acceptance lengths."""

import argparse
import json
import os
from pathlib import Path


def prepare_programs(api, devices, directory, enable_draft):
    from pypto_serving.model.glm53.npu_executor import prepare_kernels

    phases = ("prefill", "decode", "verify", "orthrus") if enable_draft else ("prefill", "decode", "verify")
    geometry = {name: api.forward_geometry(name) for name in phases}
    manifest = directory / "manifest.json" if directory is not None else None
    if manifest is not None and manifest.is_file():
        from pypto.ir import DistributedConfig
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
        from pypto_serving.model.common.compiler.l3_callable import L3Callable

        stored = json.loads(manifest.read_text())
        if stored["geometry"] != geometry or set(stored["programs"]) != set(phases):
            raise ValueError("compiled assets do not match the verifier geometry")
        distributed = DistributedConfig(device_ids=devices, num_sub_workers=0)

        def load(name, path):
            compiled = DistributedCompiledProgram.from_dir(path, platform="a2a3", distributed_config=distributed)
            return L3Callable(name=name, compiled=compiled)

        programs = {name: load(name, path) for name, path in stored["programs"].items()}
        programs["_auxiliary"] = {name: load(name, path) for name, path in stored["auxiliary"].items()}
        return programs

    programs = prepare_kernels(api, devices, pypto_build_dir=directory, orthrus=enable_draft)
    if manifest is not None:
        def artifact(entry):
            path = getattr(entry.compiled, "output_dir", None) or getattr(entry.compiled, "work_dir", None)
            if path is None:
                raise RuntimeError(f"compiled {entry.name} has no artifact directory")
            return str(Path(path).resolve())

        data = {"geometry": geometry, "programs": {name: artifact(programs[name]) for name in phases},
                "auxiliary": {name: artifact(value) for name, value in programs["_auxiliary"].items()}}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(manifest)
    return programs


def run_cases(runner, vocab_size, mask_id=None):
    """Compare greedy decoding after zero, partial and full draft acceptance."""
    capacity = runner._api.CAPACITY
    page_size = runner._api.BLOCK_SIZE
    report = []
    for matched in (0, 1, 2, 5) if mask_id is None else (None,):
        runner.reset_requests([0, 1])
        # The verify block straddles a page. A partial commit at exactly the
        # edge must also publish the corresponding KDA checkpoint.
        prompt = [1949, 373, 358, 264, 6823] * (page_size // 5 + 1)
        prompt = prompt[:page_size - 2]
        for start in range(0, len(prompt), capacity // 2):
            chunk = prompt[start:start + capacity // 2]
            previous = runner.forward_chunks([chunk, chunk], [0, 1]).argmax(dim=1).tolist()
            if previous[0] != previous[1]:
                raise AssertionError("identical prompts differ between native request slots")
        baseline = [previous[0]]
        for _ in range(8):
            logits = runner.forward_chunks([[baseline[-1]]], [0], phase="decode")
            baseline.append(int(logits[0].argmax()))
        if matched is None:
            from blockspec.pypto_glm53_serving import propose_greedy

            guesses = propose_greedy(runner, 1, len(prompt), baseline[0], mask_id)
            block = [baseline[0], *guesses]
            matched = next((i for i, token in enumerate(guesses) if token != baseline[i + 1]), len(guesses))
            if int(runner._seq_lens[1]) != len(prompt):
                raise AssertionError("draft advanced the committed cache")
        else:
            block = baseline[:6]
            if matched < 5:
                block[matched + 1] = (block[matched + 1] + 1) % vocab_size
        verify, = runner.verify_orthrus_greedy([block], [1])
        if verify.committed != 1 + matched or list(verify.emitted) != baseline[1:2 + matched]:
            raise AssertionError(f"unexpected batch verifier result for {matched=} {verify=}")
        if int(runner._seq_lens[1]) != len(prompt) + verify.committed:
            raise AssertionError("the cache length includes a rejected token")
        next_token = verify.emitted[-1]
        for expected in baseline[2 + matched:]:
            logits = runner.forward_chunks([[next_token]], [1], phase="decode")
            next_token = int(logits[0].argmax())
            if next_token != expected:
                raise AssertionError(f"greedy output diverged after commit for {matched=}")
        evidence = {"accepted_draft_tokens": matched, "target_tokens_match": True}
        if mask_id is not None:
            from blockspec.pypto_glm53_serving import collect_clean_logits

            student, teacher = collect_clean_logits(runner, prompt + baseline, len(prompt), 2, mask_id)
            if student.argmax(1).tolist() != guesses or teacher.argmax(1).tolist() != baseline[1:6]:
                raise AssertionError("online replay did not reproduce the draft and clean target predictions")
            evidence["teacher_replay_matches"] = True
        report.append(evidence)
    return report


def run_resident_distillation(runner, mask_id, *, frozen_suffix=False):
    """Compare two real Serving replay rounds with a full-vocabulary reference."""
    import torch

    from blockspec.pypto_glm53_online_serving import ResidentDistillation
    from blockspec.pypto_glm53_serving import collect_clean_logits

    def download(bank, rank, dtype):
        shard = bank.shards[rank]
        host = torch.empty(shard.shape, dtype=dtype).share_memory_()
        runner.runtime.copy_from(host.data_ptr(), shard.data_ptr, host.nbytes, worker_id=rank)
        return host

    tokens = [1949, 373, 358, 264, 6823] * 3
    api = runner._api
    valid = api.VERIFY_QUERY_WIDTH - 1
    report = []
    last_tape = None
    with ResidentDistillation(runner, frozen_suffix=frozen_suffix) as training:
        for anchor in (2, 7):
            student, teacher = collect_clean_logits(runner, tokens, anchor, 2, mask_id)
            logits_a = student.double().log_softmax(1)
            logits_b = teacher.double().log_softmax(1)
            expected_loss = ((logits_b - logits_a) * logits_b.exp()).sum(1) / valid
            loss, gradient = training.step(tokens, anchor, 2, mask_id)
            total = sum(download(loss, rank, torch.float32)[:, 0].double() for rank in range(api.RANKS))
            torch.testing.assert_close(total[:valid], expected_loss, atol=3e-5, rtol=2e-4)
            if torch.count_nonzero(total[valid:]):
                raise AssertionError("resident replay included inactive logit rows")
            first = download(gradient, 0, torch.float32)
            if not torch.isfinite(first).all() or not torch.count_nonzero(first[:valid * api.HC_MULT]):
                raise AssertionError("resident GLM gradient is nonfinite or empty")
            if torch.count_nonzero(first[valid * api.HC_MULT:]):
                raise AssertionError("resident replay included inactive gradient rows")
            for rank in range(1, api.RANKS):
                torch.testing.assert_close(download(gradient, rank, torch.float32), first, atol=3e-6, rtol=2e-4)
            evidence = {"anchor": anchor, "loss": float(total.sum()),
                        "loss_max_abs": float((total[:valid] - expected_loss).abs().max())}
            if training.suffix_tape is not None:
                tape = download(training.suffix_tape, 0, torch.bfloat16)
                views = tape.reshape(4, api.VERIFY_QUERY_WIDTH, tape.shape[1], api.D)
                if not torch.isfinite(tape).all() or any(not t.count_nonzero() for t in views):
                    raise AssertionError("GLM replay did not capture all four suffix boundaries")
                if tape.shape[1] > api.HC_MULT and any(not t[:, -1].count_nonzero() for t in views):
                    raise AssertionError("GLM replay did not capture the four sublayer outputs")
                if last_tape is not None and torch.equal(last_tape, tape):
                    raise AssertionError("GLM suffix tape is stale after changing the anchor")
                last_tape = tape
                evidence["suffix_rows"] = len(tape)
            if frozen_suffix:
                mla_grad = download(training.mla_output_gradient, 0, torch.float32)
                if not torch.isfinite(mla_grad).all() or not mla_grad[:valid].count_nonzero():
                    raise AssertionError("resident suffix produced an invalid MLA output gradient")
                if mla_grad[valid:].count_nonzero():
                    raise AssertionError("resident suffix differentiated padded output rows")
                for rank in range(1, api.RANKS):
                    replica = download(training.mla_output_gradient, rank, torch.float32)
                    torch.testing.assert_close(replica, mla_grad, atol=3e-6, rtol=2e-4)
                evidence["mla_output_grad_norm"] = float(mla_grad.norm())
                core = download(training.frozen_suffix.kda.core, 0, torch.bfloat16)[:api.VERIFY_QUERY_WIDTH]
                if not torch.isfinite(core).all() or not core.count_nonzero():
                    raise AssertionError("resident KDA replay did not emit a valid block")
                evidence["kda_replay_core_norm"] = float(core.float().norm())
            report.append(evidence)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--devices", type=int, nargs=8, default=list(range(8)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, help="save or reuse precompiled programs")
    parser.add_argument("--mask-token-id", type=int, help="also compile and exercise the actual draft model")
    parser.add_argument("--device-greedy", action="store_true", help="return draft and verifier token IDs from the NPU")
    parser.add_argument("--resident-distillation", action="store_true", help="check the Serving-to-PyPTO replay and KL path")
    parser.add_argument("--frozen-suffix", action="store_true", help="extend resident distillation through the final MoE")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.devices)) != 8 or args.output.exists():
        parser.error("use eight distinct devices and a new output file")
    if args.checkpoint is None and not args.compile_only:
        parser.error("a GLM W8A8 checkpoint is required for device execution")
    if args.frozen_suffix:
        args.resident_distillation = True
    if (args.resident_distillation or args.device_greedy) and args.mask_token_id is None:
        parser.error("--resident-distillation and --device-greedy require --mask-token-id")

    from pypto_serving.model.glm53.kernel_api import import_kernel
    from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner

    result = {"passed": False, "compiled": False, "executed": False, "devices": args.devices}
    runner = None
    try:
        api = import_kernel("prefill_fwd", tp_size=8, ep_size=8, capacity=256, max_seq_len=1280)
        os.environ["PYPTO_GLM_ORTHRUS_VERIFY"] = "1"
        programs = prepare_programs(api, args.devices, args.artifact_dir, args.mask_token_id is not None)
        if args.resident_distillation or args.device_greedy:
            from pypto_serving.model.glm53.npu_executor import KernelCompiler, build_pypto_run_config

            config = build_pypto_run_config(platform="a2a3", device_ids=args.devices, pypto_build_dir=None)
            compiler = KernelCompiler(run_config=config)
        if args.device_greedy:
            from blockspec.pypto_glm53_greedy import PROGRAM, prepare_greedy

            programs["_auxiliary"][PROGRAM] = prepare_greedy(api, compiler)
            result["device_greedy_compiled"] = True
        if args.resident_distillation:
            from blockspec.pypto_glm53_online_serving import PROGRAM, prepare_distillation

            programs["_auxiliary"][PROGRAM] = prepare_distillation(api, compiler)
            result["resident_distillation_compiled"] = True
            result["suffix_tape_enabled"] = hasattr(api, "ORTHRUS_TAPE_ROWS")
        if args.frozen_suffix:
            from blockspec.pypto_glm53_online_frozen import prepare_frozen_suffix

            programs["_auxiliary"].update(prepare_frozen_suffix(api, compiler))
            result["frozen_suffix_compiled"] = True
        result["compiled"] = True
        if args.compile_only:
            view = Glm53ModelRunner(api, programs, None, args.devices, "a2a3", orthrus_mask_token_id=args.mask_token_id)
            view._initialize_execution(None, programs, args.devices)
            result["abi_checked"] = True
            result["passed"] = True
            return
        checkpoint = api.Checkpoint(args.checkpoint)
        runner = Glm53ModelRunner(api, programs, checkpoint, args.devices, "a2a3", host_num_threads=8,
                                  orthrus_mask_token_id=args.mask_token_id)
        runner.open()
        vocab_size = runner.weights.model.vocab_size
        result["verifier"] = run_cases(runner, vocab_size)
        if args.mask_token_id is not None:
            result["draft"] = run_cases(runner, vocab_size, args.mask_token_id)
        if args.resident_distillation:
            result["resident_distillation"] = run_resident_distillation(
                runner, args.mask_token_id, frozen_suffix=args.frozen_suffix,
            )
        result.update(passed=True, executed=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
