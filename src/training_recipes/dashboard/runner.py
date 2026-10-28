"""Subprocess runner for smoke training recipes."""

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


def _smoke_command(task: str, architecture: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training_recipes.cli",
        "--task",
        task,
        "--architecture",
        architecture,
        "--profile",
        "smoke",
        "--output-dir",
        str(run_dir),
    ]


class RunManager:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
            run_dir = self.output_dir / run_id
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
        dest = Path(run_dir) if run_dir is not None else self.output_dir / run_id
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
        command = _smoke_command(run.task, run.architecture, run.run_dir)
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
