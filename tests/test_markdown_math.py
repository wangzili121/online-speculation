from __future__ import annotations

from pathlib import Path
import re


DISPLAY_MATH = re.compile(r"(?ms)^```math[ \t]*\n(.*?)^```[ \t]*$")
INLINE_MATH = re.compile(r"\$`([^`]*)`\$")
FORBIDDEN_GITHUB_MACROS = (r"\operatorname",)


def _markdown_files(root: Path) -> list[Path]:
    files = [root / "README.md"]
    files.extend((root / "docs").rglob("*.md"))
    files.extend((root / "results").rglob("*.md"))
    return files


def test_github_math_uses_supported_macros_and_escaped_comparisons() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        expressions = [match.group(1) for match in DISPLAY_MATH.finditer(text)]
        expressions.extend(match.group(1) for match in INLINE_MATH.finditer(text))
        for expression in expressions:
            for macro in FORBIDDEN_GITHUB_MACROS:
                if macro in expression:
                    failures.append(
                        f"{path.relative_to(root)}: unsupported GitHub macro {macro}"
                    )
            if "<" in expression or ">" in expression:
                failures.append(
                    f"{path.relative_to(root)}: use \\lt or \\gt inside math"
                )
    assert not failures, "GitHub-incompatible math:\n" + "\n".join(failures)


def test_inline_math_uses_github_code_delimiters() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in _markdown_files(root):
        inside_fence = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                continue
            without_supported_math = INLINE_MATH.sub("", line)
            if "$" in without_supported_math:
                failures.append(f"{path.relative_to(root)}:{line_number}")
    assert not failures, (
        "inline math must use GitHub's $`...`$ delimiters:\n"
        + "\n".join(failures)
    )
