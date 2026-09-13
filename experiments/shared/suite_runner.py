"""Shared execution and manifest handling for resumable experiment suites."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from experiments.shared.artifacts import json_fingerprint


def command_text(command: Sequence[str]) -> str:
    """Render a command using the quoting rules of the current platform."""

    values = list(command)
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


def repository_environment(
    root: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment in which repository modules are importable."""

    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root), existing)
    ).rstrip(os.pathsep)
    return environment


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_manifested_commands(
    *,
    commands: Sequence[Sequence[str]],
    root: Path,
    manifest_path: Path,
    metadata: Mapping[str, Any],
    dry_run: bool,
    environment: Mapping[str, str] | None = None,
    restart: bool = False,
) -> dict[str, Any]:
    """Execute a stable command plan and resume after its last completed command."""

    command_lists = [list(command) for command in commands]
    rendered_commands = [command_text(command) for command in command_lists]
    plan = {
        "metadata": dict(metadata),
        "commands": command_lists,
    }
    plan_fingerprint = json_fingerprint(plan)
    initial: dict[str, Any] = {
        "schema_version": 2,
        **metadata,
        "commands": rendered_commands,
        "command_argv": command_lists,
        "command_sha256": [json_fingerprint(command) for command in command_lists],
        "plan_fingerprint": plan_fingerprint,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_commands": 0,
        "status": "dry_run" if dry_run else "running",
    }
    manifest = initial
    if manifest_path.is_file() and not restart:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("plan_fingerprint") != plan_fingerprint:
            raise ValueError(
                f"existing suite manifest has a different command plan: {manifest_path}"
            )
        completed = int(previous.get("completed_commands", 0))
        if not 0 <= completed <= len(commands):
            raise ValueError("suite manifest has an invalid completed command count")
        manifest = previous
        manifest["status"] = "dry_run" if dry_run else "running"
        manifest.pop("finished_at_utc", None)
        manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        completed = 0
    _write_manifest(manifest_path, manifest)
    for index, command in enumerate(commands):
        status = "SKIP" if index < completed else "RUN"
        print(f"{status} {command_text(command)}", flush=True)
    if dry_run:
        return manifest

    process_environment = repository_environment(root, base=environment)
    try:
        for index, command in enumerate(commands[completed:], start=completed + 1):
            subprocess.run(
                list(command),
                cwd=root,
                env=process_environment,
                check=True,
            )
            manifest["completed_commands"] = index
            _write_manifest(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    return manifest
