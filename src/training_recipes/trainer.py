"""Shared deterministic training loop."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import save_json
from .datasets import DummyHubDataset, create_dataset
from .forward import forward_and_loss
from .models import create_model
from .optimizers import create_optimizer
from .registry import architecture_spec


def _ensure_catalog_metadata(config: dict) -> dict:
    if config.get("source") and config.get("task_kind"):
        return config
    spec = architecture_spec(config["task"], config["architecture"])
    config.setdefault("source", spec["source"])
    config.setdefault("task_kind", spec.get("task_kind"))
    config.setdefault("family", spec.get("family"))
    return config


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train(config: dict) -> dict:
    config = _ensure_catalog_metadata(config)
    training = config["training"]
    seed = int(training.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(training.get("device", "cpu")))
    load_mode = str(config.get("model", {}).get("load_mode", "tiny_config"))
    if load_mode == "tiny_config":
        dataset = DummyHubDataset(config)
    else:
        dataset = create_dataset(config["task"], config["data"], config["data"].get("dataset_path"))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True)
    model = create_model(config).to(device)
    optimizer = create_optimizer(model.parameters(), config["optimizer"])
    output_dir = Path(config.get("output_dir", "outputs")) / config["task"] / config["architecture"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
    config_obj = getattr(model, "config", None)
    save_json(
        output_dir / "model_config.json",
        config_obj.to_dict() if hasattr(config_obj, "to_dict") else {},
    )
    first_batch = next(iter(loader))
    save_json(
        output_dir / "batch_shapes.json",
        {
            "batch_size": int(training["batch_size"]),
            **{
                name: list(value.shape)
                for name, value in first_batch.items()
                if torch.is_tensor(value)
            },
        },
    )
    metrics_path = output_dir / "metrics.jsonl"
    step = 0
    resume_from = training.get("resume_from")
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
    last_loss = 0.0
    started = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < int(training["max_steps"]):
            for raw_batch in loader:
                batch = _to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = forward_and_loss(model, batch, config["loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}")
                loss.backward()
                if training.get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training["gradient_clip_norm"])
                    )
                optimizer.step()
                step += 1
                last_loss = float(loss.detach())
                record = {"step": step, "train/loss": last_loss, **components}
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"{config['task']}/{config['architecture']} step={step} loss={last_loss:.6f}")
                if step >= int(training["max_steps"]):
                    break
    summary = {
        "task": config["task"],
        "architecture": config["architecture"],
        "profile": config["profile"],
        "steps": step,
        "final_loss": last_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "model_class": type(model).__name__,
    }
    save_json(output_dir / "summary.json", summary)
    if bool(training.get("save_checkpoint", True)):
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            output_dir / "checkpoint.pt",
        )
    return summary
