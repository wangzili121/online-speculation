"""Private, reproducible coding-workload manifests for serving experiments."""

import hashlib
import json
from pathlib import Path
import random


CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".go", ".h", ".hpp", ".java", ".js", ".jsx",
    ".py", ".rs", ".ts", ".tsx",
}


def _template_length(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return len(encoded)


def collect_code(roots):
    paths = sorted({path for root in roots for path in Path(root).rglob("*")
                    if path.is_file() and path.suffix.lower() in CODE_SUFFIXES})
    chunks = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if text.strip():
            chunks.append(f"\n\n# source-unit {len(chunks)}\n{text}")
    if not chunks:
        raise ValueError("no readable source files found")
    return "".join(chunks), len(chunks)


def _bucket_targets(count, buckets, rng):
    assigned = [round(count * fraction) for _, _, fraction in buckets[:-1]]
    assigned.append(count - sum(assigned))
    targets = []
    for bucket_index, ((lower, upper, _), size) in enumerate(
            zip(buckets, assigned, strict=True)):
        targets.extend((bucket_index, rng.randint(lower, upper))
                       for _ in range(size))
    rng.shuffle(targets)
    return targets


def build_records(tokenizer, source_text, *, count, buckets, max_new_tokens,
                  seed):
    rng = random.Random(seed)
    source_ids = tokenizer(source_text, add_special_tokens=False).input_ids
    if len(source_ids) < max(upper for _, upper, _ in buckets):
        raise ValueError("source corpus is shorter than the largest prompt bucket")
    records = []
    for request_index, (bucket_index, target) in enumerate(
            _bucket_targets(count, buckets, rng)):
        lower, upper, _ = buckets[bucket_index]
        prefix = (
            f"You are reviewing coding task {request_index}. Analyze the repository "
            "context below, identify the most important defect, and propose a precise fix.\n\n"
        )
        base_messages = [{"role": "user", "content": prefix}]
        overhead = _template_length(tokenizer, base_messages)
        maximum_content = min(len(source_ids), max(1, upper - overhead + 512))
        start = rng.randrange(0, len(source_ids) - maximum_content + 1)

        def render(length):
            content_ids = source_ids[start:start + length]
            content = tokenizer.decode(content_ids, skip_special_tokens=False)
            messages = [{"role": "user", "content": prefix + content}]
            actual = _template_length(tokenizer, messages)
            return actual, messages

        initial = min(maximum_content, max(1, target - overhead))
        actual, messages = render(initial)
        best = (abs(actual - target), actual, messages)
        low, high = 1, maximum_content
        while not lower <= actual <= upper and low <= high:
            length = (low + high) // 2
            actual, messages = render(length)
            candidate = (abs(actual - target), actual, messages)
            if lower <= actual <= upper:
                best = candidate
                break
            if candidate[0] < best[0]:
                best = candidate
            if actual < target:
                low = length + 1
            else:
                high = length - 1
        _, actual, messages = best
        if not lower <= actual <= upper:
            raise RuntimeError(
                "tokenizer round trip moved a prompt outside its bucket: "
                f"request={request_index}, target={target}, actual={actual}, "
                f"bucket=[{lower},{upper}], overhead={overhead}")
        records.append({
            "request_id": f"coding-{request_index:04d}",
            "bucket": bucket_index,
            "target_prompt_tokens": target,
            "actual_prompt_tokens": actual,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "seed": seed + request_index,
        })
    return records


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    return digest.hexdigest()


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
