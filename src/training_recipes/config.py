"""Configuration loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_document(path: str | Path) -> Config:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {source}")
    return value


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_config(recipe: Config, profile: str, overrides: Config | None = None) -> Config:
    profiles = recipe.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(profiles)}")
    resolved = deepcopy(profiles[profile])
    resolved["schema_version"] = int(recipe.get("schema_version", 1))
    resolved["task"] = recipe["task"]
    resolved["architecture"] = recipe["architecture"]
    resolved["profile"] = profile
    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value
    validate_resolved_config(resolved)
    return resolved


def validate_resolved_config(config: Config) -> None:
    required = {"task", "architecture", "profile", "data", "loss", "optimizer", "training"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing configuration keys: {', '.join(missing)}")
    training = config["training"]
    if int(training.get("max_steps", 0)) <= 0:
        raise ValueError("training.max_steps must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    losses = config["loss"]
    if not isinstance(losses, list) or not losses:
        raise ValueError("loss must be a non-empty list")
    for item in losses:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each loss entry needs a name")
        if float(item.get("weight", 1.0)) < 0:
            raise ValueError("loss weights cannot be negative")
