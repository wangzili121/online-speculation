#!/usr/bin/env python3
"""Replay the V4.1 token-ranking case through a target-only engine and native MRV2 n-gram."""

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from watch_dsv4_tp8 import check_port_available


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "docs/validation/dsv41/native-verifier-candidates.json"
SCORING_ROW = 434
OUTPUT_INDEX = 16
SPEC = {
    "method": "ngram_gpu",
    "num_speculative_tokens": 31,
    "prompt_lookup_min": 1,
    "prompt_lookup_max": 3,
    "enforce_eager": True,
}
PROMPTS = (
    "List the first ten positive integers, separated by commas.",
    "Read this context and answer with one sentence. "
    + "The sky is blue. " * 80 + "What color is the sky?",
)


def register():
    """vLLM test plugin. Replace candidate generation and capture one target logit row."""
    import torch
    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    state = Path(os.environ["DSV41_REPRO_STATE"])
    offers = json.loads(FIXTURE.read_text())

    def control():
        return state.read_text().strip()

    def write(value):
        if get_tp_group().rank_in_group == 0:
            with state.with_name("trace.jsonl").open("a") as stream:
                stream.write(json.dumps({"control": control(), **value}) + "\n")

    propose = NgramGPUSpeculator.propose
    sample = NPUModelRunner.sample
    if getattr(sample, "_dsv41_repro", False):
        return

    @wraps(propose)
    def fixed_proposals(self, input_batch, attn_metadata, slot_mappings,
                        last_hidden_states, aux_hidden_states, num_sampled,
                        num_rejected, last_sampled, next_prefill_tokens,
                        temperature, seeds, *args, dummy_run=False, **kwargs):
        if dummy_run or (len(args) > 1 and args[1]):
            return self.drafts[:input_batch.num_reqs]
        if input_batch.num_reqs != 1:
            raise RuntimeError("verifier reproduction requires one request")
        anchor = int(self.req_states.total_len.gpu[input_batch.idx_mapping[0]]) - 1
        ids = offers.get(str(anchor), [834] * self.num_speculative_steps)
        if anchor == SCORING_ROW - 1:
            if control().startswith("changed-suffix"):
                # Preserve the first two candidates and perturb only future rows.
                ids = ids[:2] + [263] * (len(ids) - 2)
            write({"anchor": anchor, "proposal": ids})
        self.drafts[0].copy_(torch.as_tensor(ids, device=self.device, dtype=self.drafts.dtype))
        return self.drafts[:1]

    @wraps(sample)
    def trace_sample(self, hidden, batch, grammar):
        result = sample(self, hidden, batch, grammar)
        if batch.num_reqs != 1 or not batch.logits_indices.numel():
            return result
        count = int(result[1][0])
        positions = batch.positions[batch.logits_indices][:count].tolist()
        if SCORING_ROW in positions:
            row = positions.index(SCORING_ROW)
            # Compute the same full set of LM-head rows as the sampler.
            logits = self.model.compute_logits(hidden[batch.logits_indices])
            if logits is not None:
                values, ids = logits[row].float().topk(4)
                write({"position": SCORING_ROW, "block_row": row,
                       "block_width": batch.logits_indices.numel(),
                       "input_token": int(batch.input_ids[batch.logits_indices[row]]),
                       "ids": ids.tolist(), "logits": values.tolist()})
        return result

    NgramGPUSpeculator.propose = fixed_proposals
    trace_sample._dsv41_repro = True
    NPUModelRunner.sample = trace_sample
    if os.environ.get("DSV41_REPRO_LAYER_TRACE") == "1":
        install_layer_trace(state)


def install_layer_trace(state):
    """Record one row before attention, after attention and after each FFN."""
    import torch
    from vllm.distributed import get_tp_group
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model

    forward = DeepseekV41Model.forward
    if getattr(forward, "_dsv41_trace", False):
        return

    @wraps(forward)
    def capture(model, input_ids, positions, *args, **kwargs):
        if get_tp_group().rank_in_group != 0 or positions.numel() > 32:
            return forward(model, input_ids, positions, *args, **kwargs)
        matches = positions.tolist()
        if SCORING_ROW not in matches:
            return forward(model, input_ids, positions, *args, **kwargs)
        if getattr(model, "use_sequence_parallel", False):
            raise RuntimeError("layer tracing requires unsharded token rows")
        row = matches.index(SCORING_ROW)
        saved = {}
        hooks = []

        def save_input(name, index):
            def hook(module, inputs):
                saved[name] = inputs[index][row].detach().clone()
            return hook

        def save_output(name):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                saved[name] = output[row].detach().clone()
            return hook

        for layer in model.layers:
            name = f"layer.{layer.layer_idx:02d}"
            hooks.extend((
                layer.register_forward_pre_hook(save_input(f"{name}.input", 1)),
                layer.self_attn.register_forward_pre_hook(save_input(f"{name}.attention_input", 1)),
                layer.self_attn.register_forward_hook(save_output(f"{name}.attention")),
                layer.mlp.register_forward_pre_hook(save_input(f"{name}.moe_input", 0)),
                layer.mlp.register_forward_hook(save_output(f"{name}.moe")),
                layer.register_forward_hook(save_output(f"{name}.output")),
            ))
        try:
            result = forward(model, input_ids, positions, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if not saved:
            raise RuntimeError("no eager decoder layers were captured")
        label = state.read_text().strip()
        output = state.with_name(f"layers-{label}.pt")
        # Preserve the raw BF16/FP32 activations for comparisons on the host.
        torch.save({"positions": matches, "input_token": int(input_ids[row]),
                    "activations": {name: value.cpu() for name, value in saved.items()}}, output)
        return result

    capture._dsv41_trace = True
    DeepseekV41Model.forward = capture

    attention = AscendDSAV41Impl._forward_attention

    @wraps(attention)
    def capture_swa(impl, layer, q, metadata, compressed_indices, **kwargs):
        output = attention(impl, layer, q, metadata, compressed_indices, **kwargs)
        if layer.layer_idx or q.shape[0] > 32 or get_tp_group().rank_in_group != 0:
            return output
        positions = metadata.positions[:q.shape[0]].tolist()
        if SCORING_ROW not in positions:
            return output
        row = positions.index(SCORING_ROW)
        swa = metadata.swa
        if (swa.num_reqs != 1 or swa.ori_sparse_indices is not None
                or swa.replay_block_table is not None or getattr(layer, "packed_cache_ops", None) is not None):
            raise RuntimeError("the SWA probe requires the A2 eager causal cache layout")
        begin = max(0, SCORING_ROW - swa.ori_win_left)
        indices = torch.arange(begin, SCORING_ROW + 1, device=q.device)
        block = swa.storage_block_size
        pages = swa.block_table[0, indices // block].long()
        cache = layer.dsa_attn.swa_cache_layer.kv_cache[0]
        first = output[row].detach().clone()
        duplicate = attention(impl, layer, q, metadata, compressed_indices, **kwargs)[row]
        label = state.read_text().strip()
        torch.save({"query": q[row].cpu(), "visible_kv": cache[pages, indices % block].cpu(),
                    "swa_output": first.cpu(), "identical_relaunch": torch.equal(first, duplicate)},
                   state.with_name(f"swa-{label}.pt"))
        return output

    AscendDSAV41Impl._forward_attention = capture_swa


def prepare_plugin(path):
    # A temporary entry point inherits across vLLM's worker spawn. The
    # pinned environment and its installed packages stay as they are.
    metadata = path / "dsv41_verifier_repro-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: dsv41-verifier-repro\nVersion: 1\n")
    (metadata / "entry_points.txt").write_text(
        "[vllm.general_plugins]\ndsv41_verifier_repro = repro_dsv41_native_verifier:register\n"
    )
    launcher = path / "vllm"
    launcher.write_text(
        '#!/bin/sh\n'
        'if [ -n "$DSV41_REPRO_SPEC" ]; then\n'
        '  exec "$DSV41_REPRO_VLLM" "$@" --speculative-config "$DSV41_REPRO_SPEC"\n'
        'fi\n'
        'exec "$DSV41_REPRO_VLLM" "$@"\n'
    )
    launcher.chmod(0o700)


def wait_ready(proc, url, log):
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urlopen(url + "/health", timeout=3):
                return
        except (OSError, URLError):
            time.sleep(5)
    raise RuntimeError(f"service did not become ready, see {log}")


def complete(url, text):
    data = json.dumps({
        "model": "dsv41-target-single-a2",
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 40, "min_tokens": 40, "temperature": 0, "return_token_ids": True,
    }).encode()
    req = Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=1200) as result:
        ids = json.load(result)["choices"][0]["token_ids"]
    if len(ids) != 40:
        raise RuntimeError(f"expected 40 tokens, got {len(ids)}")
    return ids


def run_server(args, env, state, *controls):
    url = f"http://127.0.0.1:{args.port}"
    check_port_available(args.port)
    log = state.parent / f"{controls[0]}.log"
    results = {}
    state.write_text(controls[0])
    with log.open("w") as output:
        proc = subprocess.Popen(
            ["bash", str(REPO / "scripts/serve_dsv41_single_a2.sh")],
            env=env, start_new_session=True, stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(proc, url, log)
            for control in controls:
                state.write_text(control)
                complete(url, PROMPTS[0])
                results[control] = complete(url, PROMPTS[1])
                with state.with_name("outputs.jsonl").open("a") as output_tokens:
                    output_tokens.write(json.dumps({"control": control, "ids": results[control]}) + "\n")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--port", type=int, default=19891)
    parser.add_argument("--layer-trace", action="store_true",
                        help="capture layer activations and repeat each control once")
    args = parser.parse_args()
    if not (args.model / "config.json").is_file() or not shutil.which("vllm"):
        parser.error("run inside the pinned Ascend image with the 1017 W4A8 checkpoint mounted")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = args.output_dir.resolve() / "control.txt"
    with tempfile.TemporaryDirectory(prefix="dsv41-verifier-") as folder:
        plugin = Path(folder)
        prepare_plugin(plugin)
        python_path = os.pathsep.join(filter(None, (folder, str(Path(__file__).resolve().parent),
                                                     os.environ.get("PYTHONPATH"))))
        env = os.environ | {
            "DSV41_REPRO_VLLM": shutil.which("vllm"),
            "DSV41_REPRO_STATE": str(state),
            "DSV41_REPRO_LAYER_TRACE": str(int(args.layer_trace)),
            "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,dsv41_verifier_repro",
            "PYTHONPATH": python_path,
            "PATH": folder + os.pathsep + os.environ["PATH"],
            "MODE": "target", "MODEL": str(args.model.resolve()),
            "HOST": "127.0.0.1", "PORT": str(args.port), "MAX_MODEL_LEN": "16384",
            "MAX_NUM_BATCHED_TOKENS": "512", "MAX_NUM_SEQS": "1", "MEMORY": "0.88",
            "LOCAL_IP": args.local_ip, "NIC_NAME": args.nic, "DRY_RUN": "0",
        }
        serial = ("serial", "serial-repeat") if args.layer_trace else ("serial",)
        native = ("original", "original-repeat", "changed-suffix", "changed-suffix-repeat") if args.layer_trace else (
            "original", "changed-suffix",
        )
        tokens = run_server(args, env | {"DSV41_REPRO_SPEC": ""}, state, *serial)
        tokens |= run_server(args, env | {"DSV41_REPRO_SPEC": json.dumps(SPEC)}, state, *native)
    trace = [json.loads(line) for line in state.with_name("trace.jsonl").read_text().splitlines()]
    scores = {row["control"]: row for row in trace if "logits" in row}
    widths = dict.fromkeys(serial, 1) | dict.fromkeys(native, 32)
    if any(scores.get(key, {}).get("block_width") != width for key, width in widths.items()):
        raise RuntimeError("the controls did not capture the same input row at the required widths")
    if len({row["input_token"] for row in scores.values()}) != 1:
        raise RuntimeError("the recorded scoring rows have different input tokens")
    same_prefix = all(ids[:OUTPUT_INDEX] == tokens["serial"][:OUTPUT_INDEX] for ids in tokens.values())
    first_differences = {
        key: next((i for i, pair in enumerate(zip(tokens["serial"], ids), start=1)
                   if pair[0] != pair[1]), None)
        for key, ids in tokens.items() if key != "serial"
    }
    report = {"same_prefix": same_prefix,
              "token_17": {key: ids[OUTPUT_INDEX] for key, ids in tokens.items()},
              "first_different_output_token": first_differences,
              "target_logits": scores}
    report["reproduced"] = same_prefix and report["token_17"]["serial"] != report["token_17"]["original"]
    state.with_name("report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
