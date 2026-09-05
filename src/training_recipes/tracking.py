"""AgenticML experiment tracking helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_HYPERPARAMETER_KEYS = (
    "task",
    "architecture",
    "profile",
    "model",
    "data",
    "loss",
    "optimizer",
    "training",
    "source",
)
_MAX_STRING = 4096
_MAX_SEQUENCE = 256
_active = False


def _load_agenticml():
    try:
        import agenticml
    except ImportError as exc:
        raise ImportError(
            "Tracking is enabled but agenticml is not installed. "
            "Install it with: pip install 'agenticml-py>=0.0.3'"
        ) from exc
    return agenticml


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _flatten_value(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_value(child, item, out)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_SEQUENCE:
            return
        for index, item in enumerate(value):
            _flatten_value(f"{prefix}.{index}", item, out)
        return
    if isinstance(value, Path):
        out[prefix] = str(value)
        return
    if not _is_json_scalar(value):
        return
    if isinstance(value, str) and len(value) > _MAX_STRING:
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    out[prefix] = value


def flatten_hyperparameters(config: dict) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in _HYPERPARAMETER_KEYS:
        if key in config:
            _flatten_value(key, config[key], flattened)
    return flattened


def _resolved_tracking(config: dict, output_dir: Path) -> dict[str, Any]:
    tracking = dict(config.get("tracking") or {})
    mode = str(tracking.get("mode") or "offline").lower()
    name = tracking.get("name") or "{task}-{architecture}-{profile}".format(
        task=config.get("task", "task"),
        architecture=config.get("architecture", "architecture"),
        profile=config.get("profile", "profile"),
    )
    offline_dir = tracking.get("offline_dir")
    if offline_dir is None:
        offline_dir = str(Path(output_dir) / "agenticml")
    else:
        offline_dir = str(offline_dir)
    enabled = tracking.get("enabled", True)
    if mode == "disabled":
        enabled = False
    return {
        "enabled": bool(enabled),
        "mode": mode,
        "project": tracking.get("project") or "multimodal-training-recipes",
        "name": name,
        "offline_dir": offline_dir,
        "group": tracking.get("group") or config.get("task"),
        "tags": tracking.get("tags")
        or [
            value
            for value in (config.get("task"), config.get("architecture"), config.get("profile"))
            if value
        ],
    }


def start_run(config: dict, output_dir: str | Path) -> None:
    global _active
    _active = False
    tracking = _resolved_tracking(config, Path(output_dir))
    if not tracking["enabled"]:
        return
    agenticml = _load_agenticml()
    init_kwargs: dict[str, Any] = {
        "project": tracking["project"],
        "name": tracking["name"],
        "config": flatten_hyperparameters(config),
        "tags": tracking["tags"],
        "group": tracking["group"],
        "job_type": config.get("profile"),
        "mode": tracking["mode"],
        "track_source": "runtime",
    }
    if tracking["mode"] == "offline":
        init_kwargs["offline_dir"] = tracking["offline_dir"]
    agenticml.init(**init_kwargs)
    _active = True


def track_components(
    model,
    optimizer,
    loss_fn,
    dataset,
    loader,
    extra_files: Iterable[str | Path] | None = None,
) -> None:
    if not _active:
        return
    agenticml = _load_agenticml()
    agenticml.track_model(model)
    agenticml.track_optimizer(optimizer)
    agenticml.track_loss(loss_fn)
    agenticml.track_dataloader(dataset, name="train_dataset")
    agenticml.track_dataloader(loader, name="train")
    paths = [str(path) for path in extra_files or ()]
    if paths:
        agenticml.track_files(*paths)


def log_step(metrics: dict, step: int) -> None:
    if not _active:
        return
    _load_agenticml().log(metrics, step=step)


def write_summary(summary: dict) -> None:
    if not _active:
        return
    _load_agenticml().summary.update(summary)


def log_run_artifacts(output_dir: str | Path) -> None:
    if not _active:
        return
    agenticml = _load_agenticml()
    run_dir = Path(output_dir)
    checkpoint = run_dir / "checkpoint.pt"
    summary_path = run_dir / "summary.json"
    if checkpoint.is_file():
        agenticml.log_artifact(checkpoint, type="model")
    if summary_path.is_file():
        agenticml.log_artifact(summary_path)


def finish_run(status: str = "finished") -> None:
    global _active
    if not _active:
        return
    agenticml = _load_agenticml()
    try:
        agenticml.finish(status=status)
    finally:
        _active = False
