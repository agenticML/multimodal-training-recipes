"""Task and architecture discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = PROJECT_ROOT / "tasks"


def task_names() -> list[str]:
    return sorted(
        path.name for path in TASKS_ROOT.iterdir() if (path / "architectures.yaml").is_file()
    )


def task_dir(task: str) -> Path:
    path = TASKS_ROOT / task
    if not (path / "architectures.yaml").is_file():
        raise ValueError(f"unknown task {task!r}; expected one of {task_names()}")
    return path


def architecture_catalog(task: str) -> dict[str, dict[str, Any]]:
    document = load_document(task_dir(task) / "architectures.yaml")
    architectures = document.get("architectures", [])
    return {str(item["name"]): item for item in architectures}


def architecture_spec(task: str, architecture: str) -> dict[str, Any]:
    catalog = architecture_catalog(task)
    try:
        return catalog[architecture]
    except KeyError as exc:
        raise ValueError(
            f"unknown architecture {architecture!r} for {task}; expected one of {sorted(catalog)}"
        ) from exc


def recipe_path(task: str, architecture: str) -> Path:
    architecture_spec(task, architecture)
    return task_dir(task) / "configs" / f"{architecture}.yaml"
