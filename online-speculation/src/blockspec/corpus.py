"""Bounded public-corpus preparation with question-grouped, disjoint splits."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata


def question_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for_question(text, *, seed=314159):
    group = question_hash(text)
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if fraction < .70 else ("validation" if fraction < .85 else "test")


def convert_row(entry, tokenizer, *, seed=314159, max_tokens=8192, min_tokens=256):
    if entry.get("truncated_cells"):
        return None, "viewer_truncated"
    conversation = entry["row"].get("conversations", [])
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    if not conversation or any(x.get("from") not in roles or not isinstance(x.get("value"), str)
                               for x in conversation):
        return None, "unsupported_conversation"
    messages = [{"role": roles[x["from"]], "content": x["value"]} for x in conversation]
    questions = [m["content"] for m in messages if m["role"] == "user"]
    if not questions or messages[-1]["role"] != "assistant":
        return None, "missing_question_or_answer"
    question = "\n\n".join(questions)
    rendered = tokenizer.render_chat(messages)
    all_ids = tokenizer.encode(rendered, add_special_tokens=False)
    ids = all_ids[:max_tokens]
    if len(ids) < min_tokens:
        return None, "too_short"
    return {"input_ids": ids, "group_sha256": question_hash(question),
            "split": split_for_question(question, seed=seed),
            "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "source_row": entry["row_idx"], "domain": entry["row"].get("domain", "unknown"),
            "original_tokens": len(all_ids)}, None


def assert_disjoint(records):
    owners = {}
    for record in records:
        group, split = record["group_sha256"], record["split"]
        if group in owners and owners[group] != split:
            raise ValueError("the same question occurs in multiple splits")
        owners[group] = split


def prepare_snapshot(output, tokenizer, *, dataset, offsets, dataset_config="default", source_split="train", page_size=8,
                     seed=314159, max_tokens=8192, progress=None):
    """Fetch caller-selected conversation rows; keep split integrity metadata locally."""
    import requests

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("an explicit dataset identifier is required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"snapshot already exists: {output}")
    if not 1 <= page_size <= 100 or not offsets or any(n < 0 for n in offsets) or max_tokens < 256:
        raise ValueError("invalid bounded download configuration")
    session = requests.Session()
    revision_url = f"https://huggingface.co/api/datasets/{dataset}"
    response = session.get(revision_url, timeout=45)
    response.raise_for_status()
    revision_before = response.json()["sha"]
    records, page_hashes, skipped, seen_rows, seen_text = [], [], Counter(), set(), set()
    for offset in offsets:
        response = session.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": dataset, "config": dataset_config, "split": source_split, "offset": offset,
            "length": page_size}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        page_hashes.append({"offset": offset, "bytes_sha256": hashlib.sha256(response.content).hexdigest()})
        for entry in payload["rows"]:
            if entry["row_idx"] in seen_rows:
                skipped["duplicate_row"] += 1
                continue
            seen_rows.add(entry["row_idx"])
            record, reason = convert_row(entry, tokenizer, seed=seed, max_tokens=max_tokens)
            if record is None:
                skipped[reason] += 1
                continue
            if record["text_sha256"] in seen_text:
                skipped["duplicate_text"] += 1
                continue
            seen_text.add(record["text_sha256"])
            records.append(record)
        if progress is not None:
            progress({"offset": offset, "records_kept": len(records), "skipped": dict(skipped)})
    response = session.get(revision_url, timeout=45)
    response.raise_for_status()
    revision_after = response.json()["sha"]
    if revision_after != revision_before:
        raise RuntimeError("source revision changed while fetching; no snapshot written")
    assert_disjoint(records)
    counts = Counter(r["split"] for r in records)
    if any(counts[s] == 0 for s in ("train", "validation", "test")):
        raise ValueError("not all splits populated; choose more pages before training")
    manifest = {"dataset": dataset, "config": dataset_config, "source_split": source_split, "observed_revision": revision_before,
                "viewer_revision_pinned": False, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "page_size": page_size, "pages": page_hashes, "seed": seed, "max_tokens": max_tokens,
                "format": "local_chat_template", "splits": dict(counts), "skipped": dict(skipped),
                "unique_questions": len({r["group_sha256"] for r in records}),
                "domains": dict(Counter(r["domain"] for r in records)),
                "source_rows": [r["source_row"] for r in records]}
    output.mkdir(parents=True, exist_ok=False)
    manifest["files"] = {}
    for split in ("train", "validation", "test"):
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records if r["split"] == split)
        with (output / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        manifest["files"][split] = hashlib.sha256(body.encode()).hexdigest()
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
