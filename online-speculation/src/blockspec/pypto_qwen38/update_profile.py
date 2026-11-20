"""诊断当前 TP2 在线更新的同步调用窗口，不改计算、发布或返回值。"""

import atexit
from contextlib import contextmanager
from functools import wraps
from itertools import count
import json
from pathlib import Path
from threading import local
from time import monotonic_ns


class UpdateProfile:
    def __init__(self, output, *, clock=monotonic_ns):
        self.output, self.clock = Path(output), clock
        self.events, self.originals = [], []
        self.thread, self.ids = local(), count()

    def stack(self):
        if not hasattr(self.thread, "stack"):
            self.thread.stack = []
        return self.thread.stack

    @contextmanager
    def span(self, label):
        stack = self.stack()
        event = dict(id=next(self.ids), parent=stack[-1]["id"] if stack else None,
                     label=label, start_ns=self.clock(), child_ns=0, metadata={})
        self.events.append(event)
        stack.append(event)
        try:
            yield event
        except BaseException as error:
            event["error"] = repr(error)[:1024]
            raise
        finally:
            event["inclusive_ns"] = self.clock() - event["start_ns"]
            event["exclusive_ns"] = event["inclusive_ns"] - event["child_ns"]
            stack.pop()
            if stack:
                stack[-1]["child_ns"] += event["inclusive_ns"]

    def wrap(self, cls, method, label, *, root=False):
        original = getattr(cls, method)

        @wraps(original)
        def call(*args, **kwargs):
            # 初始化、导出与冻结推理不属于一次在线更新，不计入这里。
            if not root and not self.stack():
                return original(*args, **kwargs)
            with self.span(label) as event:
                if label == "train":
                    replay = args[1] if len(args) > 1 else kwargs["replay"]
                    event["metadata"].update(feedback_blocks=len(replay),
                        replay_audit=any("served_logits" in item for item in replay))
                value = original(*args, **kwargs)
                result = value if label == "train" else (
                    (args[1] if len(args) > 1 else kwargs["result"]) if label == "commit" else None)
                if isinstance(result, dict):
                    event["metadata"].update({k: result[k] for k in
                        ("version", "round", "supervised", "loss", "publish_read_ns",
                         "publish_pack_ns", "publish_upload_ns", "published_at_round") if k in result})
                return value

        self.originals.append((cls, method, original))
        setattr(cls, method, call)

    def restore(self):
        while self.originals:
            cls, method, original = self.originals.pop()
            setattr(cls, method, original)

    def write(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        complete = all("inclusive_ns" in row for row in self.events)
        result = dict(completed=complete, scope="Diagnostic synchronous host call windows, including worker "
                      "and device waits; not isolated kernel durations or throughput qualification. "
                      "Use exclusive spans when summing nested phases. Replay audits are labeled separately.",
                      events=self.events)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.output)


def install(output):
    """仅在独立诊断进程启动时安装；常规 serving/benchmark 不调用。"""
    from .oneserving_tp2_online import TP2Online
    from .oneserving_tp2_lora import TP2LoRAOnline
    from .online_suffix_optimizer import SuffixOptimizer
    from .training_executor import TrainingWorkspace

    if getattr(TP2Online.train, "_update_profile", False):
        raise RuntimeError("update profiler already installed")
    if Path(output).exists():
        raise ValueError("use a fresh update-profile result path")
    profile = UpdateProfile(output)
    profile.wrap(TP2Online, "train", "train", root=True)
    profile.wrap(TP2Online, "commit_update", "commit", root=True)
    TP2Online.train._update_profile = True
    for cls, method, label in [
        (TrainingWorkspace, "run", "forward_backward"),
        (SuffixOptimizer, "zero_grad", "zero_grad"),
        (SuffixOptimizer, "accumulate", "gradient_accumulate"),
        (SuffixOptimizer, "statistics", "finite_norm_checks"),
        (SuffixOptimizer, "step", "adam_and_publish"),
        (SuffixOptimizer, "render", "execution_render"),
        (TP2Online, "publish_flat", "publication_prepare"),
        (TP2LoRAOnline, "render_merged", "lora_merge_and_check"),
    ]:
        profile.wrap(cls, method, label)
    atexit.register(profile.write)
    return profile
