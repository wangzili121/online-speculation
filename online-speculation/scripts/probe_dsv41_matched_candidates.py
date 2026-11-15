#!/usr/bin/env python3
"""Compare V4.1 Orthrus with the native verifier using the same candidate blocks."""

import argparse
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from urllib.request import Request, urlopen

from repro_dsv41_native_verifier import PROMPTS, SPEC, wait_ready
from watch_dsv41_single_a2 import spec_token_count


def register():
    import torch
    from vllm.distributed import get_tp_group

    mode = os.environ["DSV41_REPLAY_ARM"]
    schedule = Path(os.environ["DSV41_REPLAY_SCHEDULE"])
    if mode == "record":
        from blockspec.vllm_ascend.dsv41_speculator import AscendOrthrusV41Speculator

        cls = AscendOrthrusV41Speculator
        original = cls.propose
        if getattr(original, "_dsv41_schedule_replay", False):
            return
        signature = inspect.signature(original)

        @wraps(original)
        def capture(self, input_batch, *args, **kwargs):
            result = original(self, input_batch, *args, **kwargs)
            call = signature.bind(self, input_batch, *args, **kwargs).arguments
            if not call.get("dummy_run") and get_tp_group().rank_in_group == 0:
                if input_batch.num_reqs != 1:
                    raise RuntimeError("candidate replay expects one request")
                anchor = int(self.input_buffers.positions[0])
                row = {"anchor": anchor, "drafts": result[0].tolist()}
                with schedule.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            return result

        capture._dsv41_schedule_replay = True
        cls.propose = capture
        return

    from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator

    cls = NgramGPUSpeculator
    original = cls.propose
    if getattr(original, "_dsv41_schedule_replay", False):
        return
    rows = [json.loads(line) for line in schedule.read_text().splitlines()]
    candidates = {row["anchor"]: row["drafts"] for row in rows}
    if len(candidates) != len(rows):
        raise RuntimeError("candidate replay prompts must have disjoint anchor positions")
    signature = inspect.signature(original)

    @wraps(original)
    def replay(self, input_batch, *args, **kwargs):
        call = signature.bind(self, input_batch, *args, **kwargs).arguments
        if call.get("dummy_run"):
            return self.drafts[:input_batch.num_reqs]
        if input_batch.num_reqs != 1:
            raise RuntimeError("candidate replay expects one request")
        anchor = int(self.req_states.total_len.gpu[input_batch.idx_mapping[0]]) - 1
        try:
            tokens = candidates[anchor]
        except KeyError as e:
            raise RuntimeError(f"native verifier did not reach a recorded anchor: {anchor}") from e
        self.drafts[0].copy_(torch.as_tensor(tokens, device=self.device, dtype=self.drafts.dtype))
        if get_tp_group().rank_in_group == 0:
            with schedule.with_name("replayed.jsonl").open("a") as stream:
                stream.write(json.dumps({"anchor": anchor, "drafts": tokens}) + "\n")
        return self.drafts[:1]

    replay._dsv41_schedule_replay = True
    cls.propose = replay


def initialize_helper(path):
    dist = path / "dsv41_schedule_replay-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: dsv41-schedule-replay\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[vllm.general_plugins]\ndsv41_schedule_replay = probe_dsv41_matched_candidates:register\n"
    )
    command = path / "vllm"
    command.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$DSV41_REPLAY_ARM" == replay ]]; then\n'
        '  exec "$DSV41_REAL_VLLM" "$@" --no-async-scheduling --speculative-config "$DSV41_NATIVE_SPEC"\n'
        'fi\n'
        'exec "$DSV41_REAL_VLLM" "$@" --no-async-scheduling\n'
    )
    command.chmod(0o700)


def complete(api, mode, prompt, length=40):
    body = json.dumps({
        "model": f"dsv41-{mode}-single-a2",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": length, "min_tokens": length, "temperature": 0, "return_token_ids": True,
    }).encode()
    request = Request(api + "/v1/chat/completions", data=body,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=1200) as result:
        ids = json.load(result)["choices"][0]["token_ids"]
    if len(ids) != length:
        raise RuntimeError(f"expected {length} completion rows, got {len(ids)}")
    return ids


def run(args, env, arm):
    mode = "orthrus" if arm == "record" else "target"
    app = "http://127.0.0.1:19891"
    out = args.output_dir / f"{arm}.log"
    with out.open("w") as log:
        child = subprocess.Popen(["bash", "scripts/serve_dsv41_single_a2.sh"],
                                 env=env | {"MODE": mode, "DSV41_REPLAY_ARM": arm},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(child, app, out)
            tokens = [complete(app, mode, prompt) for prompt in PROMPTS]
            with urlopen(app + "/metrics", timeout=10) as response:
                data = response.read().decode()
            pair = {"token_ids": tokens, "drafted": spec_token_count(data, "draft"),
                    "accepted": spec_token_count(data, "accepted")}
            (args.output_dir / f"{arm}.json").write_text(json.dumps(pair, indent=2) + "\n")
            print(json.dumps({"arm": arm, **pair}), flush=True)
            return pair
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--nic", default="enp67s0f5")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = args.output_dir.resolve() / "orth-candidates.jsonl"
    with tempfile.TemporaryDirectory(prefix="dsv41-replay-") as root:
        initialize_helper(Path(root))
        env = os.environ | {
            "DSV41_REAL_VLLM": shutil.which("vllm"),
            "DSV41_NATIVE_SPEC": json.dumps(SPEC),
            "DSV41_REPLAY_SCHEDULE": str(schedule),
            "PYTHONPATH": os.pathsep.join(filter(None, (root, str(Path(__file__).resolve().parent),
                                                        os.environ.get("PYTHONPATH")))),
            "PATH": root + os.pathsep + os.environ["PATH"],
            "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,blockspec_orthrus,dsv41_schedule_replay",
            "MODEL": args.model, "HEAD": args.head, "LOCAL_IP": args.local_ip, "NIC_NAME": args.nic,
            "HOST": "127.0.0.1", "PORT": "19891", "REQUIRE_TRAINED_HEAD": "1", "ORTHRUS_DSV41_AUDIT_CACHE": "1",
            "HCCL_DETERMINISTIC": "strict",
        }
        print("starting Orthrus with the offline-updated head", flush=True)
        orth = run(args, env, "record")
        print("replaying Orthrus proposals through the stock native verifier", flush=True)
        native = run(args, env, "replay")
    checked = schedule.read_text().splitlines()
    observed = schedule.with_name("replayed.jsonl").read_text().splitlines()
    result = {"same_schedule": checked == observed, "scheduled_rounds": len(checked),
              "same_outputs_and_metrics": orth == native, "orth": orth, "native": native}
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not (checked and checked == observed and orth == native and orth["drafted"]):
        raise SystemExit("native verifier did not match the Orthrus schedule or result")


if __name__ == "__main__":
    main()
