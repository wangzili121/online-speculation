#!/usr/bin/env python3
"""Compare the same causal token in native DSV4 prefill and decode.

Diagnostic only: hooks copy selected activations to CPU and must not be used
for latency measurements. The target model, weights and attention stay intact.
"""

import argparse
import functools
import json
from pathlib import Path

import torch


def tensor_leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, f"{prefix}/{key}")


def select_trace_rows(positions, position, starts, lengths, prompt_length=None, block_start=None):
    matches = []
    for request, length in enumerate(lengths):
        if prompt_length is not None and length != prompt_length:
            continue
        start, end = int(starts[request]), int(starts[request + 1])
        if start == end:
            continue
        if block_start is not None:
            if positions[start] == block_start:
                matches.append((start, end, request))
        else:
            matches.extend((row, row + 1, request) for row in range(start, end)
                           if positions[row] == position)
    return matches[0] if len(matches) == 1 else None


def compare_traces(left, right):
    if not left or not right:
        raise ValueError("both traces must contain captured activations")
    rows = []
    for key, a in left.items():
        if key not in right:
            continue
        b = right[key]
        if a.shape != b.shape:
            rows.append({"stage": key, "shapes": [list(a.shape), list(b.shape)]})
            continue
        if a.dtype != b.dtype:
            rows.append({"stage": key, "dtypes": [str(a.dtype), str(b.dtype)]})
            continue
        error = (a.double() - b.double()).abs()
        rows.append({"stage": key, "equal": torch.equal(a, b),
                     "max_abs": error.max().item() if error.numel() else 0.0,
                     "mean_abs": error.mean().item() if error.numel() else 0.0,
                     "different": torch.count_nonzero(a != b).item(),
                     "elements": a.numel()})
    return {"stages": rows, "only_decode": sorted(left.keys() - right.keys()),
            "only_prefill": sorted(right.keys() - left.keys())}


class LayerTraceWorker:
    """Opt-in worker extension; no deployment patch or production hook."""

    def benchmark_prolog(self):
        self.trace_disable()
        from benchmark_dsv4_prolog_npu import benchmark_on_worker
        return benchmark_on_worker(self)

    def trace_disable(self):
        state = self._layer_trace
        state["active"] = False
        for handle in state["handles"]:
            handle.remove()
        for obj, method, original in state["wrapped"]:
            setattr(obj, method, original)
        del self._layer_trace

    def trace_arm(self, position, prompt_length=None, block_start=None):
        if not hasattr(self, "_layer_trace"):
            self._layer_trace = {"active": False, "tensors": {}, "shapes": {},
                                 "position": position, "layer": "outside", "counts": {},
                                 "handles": [], "wrapped": []}
            self._install_trace()
        self._layer_trace.update(position=position, prompt_length=prompt_length, block_start=block_start,
                                 active=False, tensors={}, shapes={}, counts={})

    def _save_trace(self, name, value):
        state = self._layer_trace
        if not state["active"]:
            return
        for suffix, tensor in tensor_leaves(value):
            # Only token-major activations, never weights or persistent cache.
            if tensor.ndim and tensor.shape[0] == state["tokens"]:
                key = name + suffix
                state["tensors"][key] = tensor[state["selector"]].detach().cpu().clone()
                state["shapes"][key] = list(tensor.shape)

    def _install_trace(self):
        model = self.model_runner.model

        def start(module, args, kwargs):
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            state = self._layer_trace
            state["active"] = False
            if positions is None:
                return
            runner = self.model_runner
            lengths = [runner.requests[req].num_prompt_tokens for req in runner.input_batch.req_ids]
            values = positions.cpu().tolist()
            selected = select_trace_rows(values, state["position"], runner.query_start_loc.np,
                                         lengths, state["prompt_length"], state["block_start"])
            if selected is not None:
                start, end, request = selected
                selector = start if state["block_start"] is None else slice(start, end)
                state.update(active=True, row=start, selector=selector, request_row=request,
                             first_position=values[start], last_position=values[end - 1],
                             tokens=positions.numel(), counts={})
                self._save_trace("model.input", kwargs)

        def finish(module, args, kwargs, output):
            if self._layer_trace["block_start"] is not None:
                self._layer_trace["active"] = False

        handles = self._layer_trace["handles"]
        handles.append(model.register_forward_pre_hook(start, with_kwargs=True))
        handles.append(model.register_forward_hook(finish, with_kwargs=True))
        for name, module in model.named_modules():
            if not name or ".dsa_attn." in name:
                continue
            # Record module boundaries plus fused Q/KV prolog and O projection.
            if name.startswith("model.layers.") or name in {"model.embed_tokens", "model.norm", "lm_head"}:
                def before(mod, args, kwargs, name=name):
                    if name.count(".") == 2:
                        self._layer_trace["layer"] = name
                    self._save_trace(name + ".input", (args, kwargs))

                def after(mod, args, kwargs, output, name=name):
                    self._save_trace(name + ".output", output)

                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True))

        for index, layer in enumerate(model.model.layers):
            for method in ("hc_pre", "hc_post"):
                self._wrap_trace(layer, method, f"model.layers.{index}.{method}")
            impl = layer.self_attn.dsa_attn.dsa_attn.impl
            for method in ("_mla_prolog_multistream", "_forward_o_proj"):
                self._wrap_trace(impl, method, f"model.layers.{index}.{method}")
        import torch_npu

        for method in ("npu_dynamic_quant", "npu_quant_matmul"):
            self._wrap_trace(torch_npu, method, f"op.{method}")
        for method in ("npu_rms_norm_dynamic_quant", "npu_sparse_attn_sharedkv"):
            self._wrap_trace(torch.ops._C_ascend, method, f"op.{method}")

    def _wrap_trace(self, obj, method, name):
        original = getattr(obj, method)
        self._layer_trace["wrapped"].append((obj, method, original))

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            state = self._layer_trace
            label = f"{state['layer']}.{name}" if name.startswith("op.") else name
            count = state["counts"].get(label, 0)
            state["counts"][label] = count + 1
            key = f"{label}.{count}"
            # Method inputs can contain weights: record only the activation.
            self._save_trace(key + ".input", args[:1])
            if method == "npu_sparse_attn_sharedkv" and state["active"]:
                self._save_history(key, kwargs)
            result = original(*args, **kwargs)
            self._save_trace(key + ".output", result)
            if method == "npu_sparse_attn_sharedkv" and state["active"] and state.get("repeat_attention"):
                # Diagnostic replay with exactly the same live inputs and metadata.
                self._save_trace(key + ".repeat_output", original(*args, **kwargs))
            return result

        setattr(obj, method, wrapped)

    def _save_history(self, key, kwargs):
        state = self._layer_trace
        position = state["last_position"]
        for kind in ("ori", "cmp"):
            cache, table = kwargs.get(kind + "_kv"), kwargs.get(kind + "_block_table")
            if cache is None:
                continue
            end = position + 1 if kind == "ori" else (position + 1) // kwargs["cmp_ratio"]
            start = max(0, state["first_position"] + 1 - 128) if kind == "ori" else 0
            logical = torch.arange(start, end, device=cache.device)
            physical = table[state["request_row"], logical // cache.shape[1]].long()
            value = cache[physical, logical % cache.shape[1]].detach().cpu().clone()
            state["tensors"][key + "." + kind + "_history"] = value
            state["shapes"][key + "." + kind + "_history"] = list(value.shape)
        indices = kwargs.get("cmp_sparse_indices")
        if indices is not None:
            self._save_trace(key + ".cmp_indices", indices)
        sink = kwargs.get("sinks")
        if sink is not None:
            state["tensors"][key + ".sink"] = sink.detach().cpu().clone()

    def trace_flush(self, directory, label):
        state = self._layer_trace
        state["active"] = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{label}-rank{self.rank}.pt"
        torch.save(state["tensors"], path)
        path.with_suffix(".json").write_text(json.dumps(state["shapes"], indent=2) + "\n")
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dsa-overlap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--benchmark-prolog", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    from vllm import LLM, SamplingParams

    additional = {"ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": False},
                  "enable_cpu_binding": True, "multistream_overlap_shared_expert": True}
    if args.dsa_overlap is not None:
        additional["multistream_dsv4_dsa_overlap"] = args.dsa_overlap
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enable_expert_parallel=True,
              quantization="ascend", tokenizer_mode="deepseek_v4", enforce_eager=True,
              compilation_config={"cudagraph_mode": "NONE"}, additional_config=additional,
              max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=1024,
              gpu_memory_utilization=0.50, enable_prefix_caching=False, block_size=128,
              model_loader_extra_config={"enable_multithread_load": True, "num_threads": 128},
              worker_extension_cls="trace_dsv4_native_layers.LayerTraceWorker")
    baseline = json.loads(args.baseline.read_text())["greedy"]
    candidate = json.loads(args.candidate.read_text())["greedy"]
    results = []
    for case_index, (a, b) in enumerate(zip(baseline, candidate, strict=True)):
        offset = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y), None)
        if offset is None:
            continue
        prompt = a["case"]["prompt"] * int(a["case"].get("prompt_repeat", 1))
        ids = llm.get_tokenizer().encode(prompt)
        assert len(ids) == a["usage"]["prompt_tokens"]
        prefix = ids + a["token_ids"][:offset]
        paths, tokens = {}, {}
        for phase, inputs, length in (("decode", ids, offset + 1), ("prefill", prefix, 1)):
            if phase == "prefill":
                # A runtime option can change earlier output tokens. Always
                # compare the actual decode prefix, not an old snapshot's.
                inputs = ids + tokens["decode"][:-1]
            llm.collective_rpc("trace_arm", args=(len(prefix) - 1,))
            output = llm.generate([{"prompt_token_ids": inputs}],
                                  SamplingParams(temperature=0, max_tokens=length, ignore_eos=True),
                                  use_tqdm=False)
            tokens[phase] = list(output[0].outputs[0].token_ids)
            paths[phase] = llm.collective_rpc("trace_flush", args=(str(args.output), f"case{case_index}-{phase}"))
        comparisons = []
        for left, right in zip(paths["decode"], paths["prefill"], strict=True):
            comparisons.append(compare_traces(torch.load(left, weights_only=True), torch.load(right, weights_only=True)))
        row = {"case_index": case_index, "position": len(prefix) - 1, "tokens": tokens,
               "reproduces_baseline": tokens["decode"] == a["token_ids"][:offset + 1],
               "same_prefix": tokens["decode"][:-1] == a["token_ids"][:offset],
               "prefill_matches_decode": tokens["prefill"] == tokens["decode"][-1:],
               "ranks": comparisons}
        results.append(row)
        (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "ranks"}), flush=True)
    if args.benchmark_prolog:
        result = llm.collective_rpc("benchmark_prolog", timeout=600)
        (args.output / "prolog-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
