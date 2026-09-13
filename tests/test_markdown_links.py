from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "chatgpt-conversation:")


def test_all_local_markdown_links_resolve():
    root = Path(__file__).resolve().parents[1]
    markdown = [root / "README.md"]
    markdown.extend((root / "docs").rglob("*.md"))
    markdown.extend((root / "results").rglob("*.md"))
    missing = []
    for source in markdown:
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("#", *EXTERNAL_SCHEMES)):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (source.parent / relative).resolve().exists():
                missing.append(f"{source.relative_to(root)} -> {target}")
    assert not missing, "missing local Markdown targets:\n" + "\n".join(missing)
