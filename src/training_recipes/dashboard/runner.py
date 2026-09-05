"""Subprocess runner and AgenticML journal/code asset discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training_recipes.models import (
    UNSUPPORTED_DUMMY_ARCHITECTURES,
    UNSUPPORTED_DUMMY_LIBRARIES,
)

TEXT_CAP_BYTES = 1_048_576
BINARY_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".safetensors", ".so", ".dylib", ".whl"}
SOURCE_ORDER = (
    "model",
    "loss",
    "dataloader",
    "optimizer",
    "runtime_import",
    "runtime_read",
    "extra",
    "unknown",
)


class AssetPathError(ValueError):
    """Raised when an asset path is missing, binary-only, or escapes the run directory."""


@dataclass
class AssetFile:
    path: str
    source: str
    content_type: str | None = None
    logical_path: str | None = None
    size: int | None = None


@dataclass
class RunState:
    id: str
    task: str
    architecture: str
    run_dir: Path
    status: str = "running"
    logs: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None
    returncode: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    updated: threading.Event = field(default_factory=threading.Event)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            self.updated.set()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "task": self.task,
                "architecture": self.architecture,
                "status": self.status,
                "summary": self.summary,
                "logs": "\n".join(self.logs),
                "error": self.error,
                "run_dir": str(self.run_dir),
            }

    def wait_for_update(self, timeout: float = 0.5) -> None:
        self.updated.wait(timeout=timeout)
        self.updated.clear()


def is_dummy_supported(task: str, architecture: str, spec: dict[str, Any]) -> bool:
    library = str((spec.get("source") or {}).get("library") or "")
    return library not in UNSUPPORTED_DUMMY_LIBRARIES and (
        task,
        architecture,
    ) not in UNSUPPORTED_DUMMY_ARCHITECTURES


def serialize_architecture(task: str, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    source = spec.get("source") or {}
    return {
        "name": name,
        "family": spec.get("family"),
        "hub_id": source.get("hub_id"),
        "class": source.get("class"),
        "losses": list(spec.get("losses") or []),
        "dummy_supported": is_dummy_supported(task, name, spec),
        "library": source.get("library"),
    }


def discover_offline_dir(run_dir: Path) -> Path | None:
    matches = sorted(path for path in run_dir.glob("offline-*") if path.is_dir())
    if not matches:
        return None
    with_journal = [path for path in matches if (path / "journal.jsonl").is_file()]
    return (with_journal or matches)[-1]


def _source_from_code_path(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) >= 2 and parts[0] == "code":
        top = parts[1]
        if top in {"model", "loss", "dataloader", "optimizer"}:
            return top
        if top == "agenticml":
            return "runtime_import"
    return "unknown"


def parse_journal_assets(journal_path: Path) -> list[AssetFile]:
    sources: dict[str, dict[str, Any]] = {}
    stored: dict[str, AssetFile] = {}
    for raw in journal_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "upload_code":
            continue
        payload = event.get("payload") or {}
        manifest = payload.get("manifest") or {}
        for item in manifest.get("files") or []:
            path = item.get("path")
            if not path:
                continue
            sources[path] = item
        for item in payload.get("files") or []:
            logical = item.get("path")
            stored_at = item.get("stored_at") or (
                f"code/{logical}" if logical else None
            )
            if not stored_at:
                continue
            meta = sources.get(logical or "", {})
            stored[stored_at] = AssetFile(
                path=stored_at,
                source=str(meta.get("source") or _source_from_code_path(stored_at)),
                content_type=item.get("content_type") or meta.get("content_type"),
                logical_path=logical,
                size=meta.get("size"),
            )
    return list(stored.values())


def walk_code_fallback(offline_dir: Path) -> list[AssetFile]:
    code_dir = offline_dir / "code"
    if not code_dir.is_dir():
        return []
    files: list[AssetFile] = []
    for path in sorted(code_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(offline_dir).as_posix()
        suffix = path.suffix.lower()
        content_type = "text/x-python" if suffix == ".py" else None
        files.append(
            AssetFile(
                path=relative,
                source=_source_from_code_path(relative),
                content_type=content_type,
                size=path.stat().st_size,
            )
        )
    return files


def collect_assets(run_dir: Path) -> tuple[Path | None, list[AssetFile]]:
    offline_dir = discover_offline_dir(run_dir)
    if offline_dir is None:
        return None, []
    journal = offline_dir / "journal.jsonl"
    files = parse_journal_assets(journal) if journal.is_file() else []
    if not files:
        files = walk_code_fallback(offline_dir)
    return offline_dir, files


def group_assets(files: list[AssetFile]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in files:
        groups.setdefault(item.source, []).append(
            {
                "path": item.path,
                "source": item.source,
                "content_type": item.content_type,
                "logical_path": item.logical_path,
                "size": item.size,
            }
        )
    ordered: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_ORDER:
        if source in groups:
            ordered[source] = groups.pop(source)
    for source in sorted(groups):
        ordered[source] = groups[source]
    return ordered


def summary_from_journal(journal_path: Path) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for raw in journal_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "finalize":
            continue
        payload = event.get("payload") or {}
        summary = payload.get("summary")
        if isinstance(summary, dict):
            last = summary
    return last


def parse_summary_from_logs(lines: list[str]) -> dict[str, Any] | None:
    text = "\n".join(lines)
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    index = 0
    while True:
        brace = text.find("{", index)
        if brace < 0:
            break
        try:
            value, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(value, dict) and (
            "steps" in value or "final_loss" in value or "model_class" in value
        ):
            last = value
        index = end
    return last


def resolve_asset_path(run_dir: Path, requested: str) -> Path:
    if not requested or requested.strip() != requested:
        raise AssetPathError("path is required")
    candidate = Path(requested)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AssetPathError("path escapes the run directory")
    run_root = run_dir.resolve()
    offline_dir = discover_offline_dir(run_root)
    options = [run_root / candidate]
    if offline_dir is not None:
        options.append(offline_dir / candidate)
    resolved: Path | None = None
    for option in options:
        try:
            path = option.resolve()
        except OSError as exc:
            raise AssetPathError("invalid asset path") from exc
        if path.exists() and path.is_relative_to(run_root):
            resolved = path
            break
    if resolved is None:
        probe = (run_root / candidate).resolve()
        if not probe.is_relative_to(run_root):
            raise AssetPathError("path escapes the run directory")
        raise AssetPathError("asset not found")
    if not resolved.is_file():
        raise AssetPathError("asset not found")
    return resolved


def read_asset_file(run_dir: Path, requested: str) -> dict[str, Any]:
    path = resolve_asset_path(run_dir, requested)
    relative = path.relative_to(run_dir.resolve()).as_posix()
    suffix = path.suffix.lower()
    size = path.stat().st_size
    if suffix in BINARY_SUFFIXES:
        return {
            "path": relative,
            "binary": True,
            "truncated": False,
            "content": None,
            "note": f"binary file skipped ({suffix})",
            "size": size,
        }
    data = path.read_bytes()[:TEXT_CAP_BYTES]
    truncated = size > TEXT_CAP_BYTES
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "path": relative,
            "binary": True,
            "truncated": False,
            "content": None,
            "note": "binary file skipped (not valid UTF-8)",
            "size": size,
        }
    payload: dict[str, Any] = {
        "path": relative,
        "binary": False,
        "truncated": truncated,
        "content": content,
        "size": size,
    }
    if truncated:
        payload["note"] = f"truncated to {TEXT_CAP_BYTES} bytes"
    return payload


def _offline_command(task: str, architecture: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training_recipes.dashboard._offline_cli",
        "--task",
        task,
        "--architecture",
        architecture,
        "--profile",
        "smoke",
        "--tracker-mode",
        "offline",
        "--agenticml-asset-dir",
        str(run_dir),
    ]


class RunManager:
    def __init__(self, asset_dir: Path):
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}
        self._active_id: str | None = None

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def active_run(self) -> RunState | None:
        with self._lock:
            return self._active_locked()

    def start(self, task: str, architecture: str) -> RunState:
        with self._lock:
            if self._active_locked() is not None:
                raise RuntimeError("a run is already in progress")
            run_id = uuid.uuid4().hex
            run_dir = self.asset_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run = RunState(
                id=run_id,
                task=task,
                architecture=architecture,
                run_dir=run_dir,
            )
            self._runs[run_id] = run
            self._active_id = run_id
        thread = threading.Thread(target=self._run_process, args=(run,), daemon=True)
        thread.start()
        return run

    def register_finished(
        self,
        *,
        task: str,
        architecture: str,
        run_dir: Path | None = None,
        run_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> RunState:
        run_id = run_id or uuid.uuid4().hex
        dest = Path(run_dir) if run_dir is not None else self.asset_dir / run_id
        dest.mkdir(parents=True, exist_ok=True)
        run = RunState(
            id=run_id,
            task=task,
            architecture=architecture,
            run_dir=dest,
            status="finished",
            summary=summary,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def _active_locked(self) -> RunState | None:
        if self._active_id is None:
            return None
        run = self._runs.get(self._active_id)
        if run is None or run.status != "running":
            self._active_id = None
            return None
        return run

    def iter_events(self, run: RunState) -> Iterator[str]:
        index = 0
        while True:
            with run.lock:
                while index < len(run.logs):
                    line = run.logs[index]
                    index += 1
                    yield f"event: log\ndata: {json.dumps(line)}\n\n"
                done = run.status in {"finished", "failed"}
                summary = run.summary
                status = run.status
                error = run.error
            if done:
                payload = {"status": status, "summary": summary, "error": error}
                yield f"event: complete\ndata: {json.dumps(payload)}\n\n"
                return
            run.wait_for_update()

    def _run_process(self, run: RunState) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        command = _offline_command(run.task, run.architecture, run.run_dir)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            with run.lock:
                run.status = "failed"
                run.error = str(exc)
                run.updated.set()
            self._clear_active(run.id)
            return
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                run.append_log(line.rstrip("\n"))
            returncode = process.wait()
        except OSError as exc:
            with run.lock:
                run.status = "failed"
                run.error = str(exc)
                run.updated.set()
            self._clear_active(run.id)
            return
        summary = parse_summary_from_logs(run.logs)
        offline_dir = discover_offline_dir(run.run_dir)
        journal = None if offline_dir is None else offline_dir / "journal.jsonl"
        if summary is None and journal is not None and journal.is_file():
            summary = summary_from_journal(journal)
        with run.lock:
            run.returncode = returncode
            run.summary = summary
            if returncode == 0:
                run.status = "finished"
            else:
                run.status = "failed"
                run.error = f"process exited with code {returncode}"
            run.updated.set()
        self._clear_active(run.id)

    def _clear_active(self, run_id: str) -> None:
        with self._lock:
            if self._active_id == run_id:
                self._active_id = None
