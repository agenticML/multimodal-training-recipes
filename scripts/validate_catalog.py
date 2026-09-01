#!/usr/bin/env python3
"""Validate task topology, source locks, and recipe compatibility."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training_recipes.config import load_document, resolve_config
from training_recipes.registry import (
    architecture_catalog,
    recipe_path,
    task_dir,
    task_names,
)

EXPECTED_TASKS = {
    "tts",
    "asr",
    "speech_to_speech",
    "translation",
    "image_generation",
    "image_inpainting",
    "video_generation",
    "text_to_image",
    "text_to_video",
    "image_captioning",
    "video_captioning",
    "llm",
    "biomedical_signal_processing",
    "video_classification",
    "image_classification",
    "audio_super_resolution",
}
REQUIRED_TASK_FILES = {
    "architectures.yaml",
    "models.py",
    "losses.py",
    "optimizers.py",
    "datasets.py",
    "train.py",
    "train.sh",
    "README.md",
}


def validate(online: bool = False) -> list[str]:
    errors: list[str] = []
    discovered = set(task_names())
    if discovered != EXPECTED_TASKS:
        errors.append(
            f"task mismatch: missing={sorted(EXPECTED_TASKS - discovered)} extra={sorted(discovered - EXPECTED_TASKS)}"
        )
    for task in sorted(discovered):
        directory = task_dir(task)
        missing_files = sorted(
            name for name in REQUIRED_TASK_FILES if not (directory / name).is_file()
        )
        if missing_files:
            errors.append(f"{task}: missing files {missing_files}")
        catalog = architecture_catalog(task)
        if len(catalog) < 5:
            errors.append(f"{task}: expected at least 5 architectures, found {len(catalog)}")
        task_losses = {loss for spec in catalog.values() for loss in spec.get("losses", [])}
        if len(task_losses) < 4:
            errors.append(f"{task}: expected at least 4 losses, found {len(task_losses)}")
        for name, spec in catalog.items():
            source = spec.get("source", {})
            revision = str(source.get("revision", ""))
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                errors.append(f"{task}/{name}: revision is not an immutable SHA")
            if not source.get("source_file"):
                errors.append(f"{task}/{name}: source_file is missing")
            for field in ("license", "gated", "minimum_hardware"):
                if field not in source:
                    errors.append(f"{task}/{name}: source metadata is missing {field}")
            path = recipe_path(task, name)
            if not path.is_file():
                errors.append(f"{task}/{name}: recipe file is missing")
                continue
            recipe = load_document(path)
            for profile in ("smoke", "full"):
                try:
                    resolved = resolve_config(recipe, profile)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{task}/{name}/{profile}: {exc}")
                    continue
                configured = {entry["name"] for entry in resolved["loss"]}
                if not configured.issubset(set(spec["losses"])):
                    errors.append(f"{task}/{name}/{profile}: configured incompatible loss")
            if online:
                try:
                    with urllib.request.urlopen(
                        f"https://huggingface.co/api/models/{source['hub_id']}", timeout=20
                    ) as response:
                        current = json.load(response)["sha"]
                    if current != revision:
                        errors.append(f"{task}/{name}: Hub head differs from pinned revision")
                except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
                    errors.append(f"{task}/{name}: Hub source check failed: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    errors = validate(args.online)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    architecture_count = sum(len(architecture_catalog(task)) for task in task_names())
    print(f"catalog valid: {len(task_names())} tasks, {architecture_count} architectures")


if __name__ == "__main__":
    main()
