"""Shared deterministic training loop."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import save_json
from .datasets import create_dataset
from .losses import compute_loss
from .models import create_smoke_model
from .optimizers import create_optimizer


def train(config: dict) -> dict:
    training = config["training"]
    seed = int(training.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(training.get("device", "cpu")))
    dataset = create_dataset(config["task"], config["data"], config["data"].get("dataset_path"))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True)
    model = create_smoke_model(config["task"], config).to(device)
    optimizer = create_optimizer(model.parameters(), config["optimizer"])
    output_dir = Path(config.get("output_dir", "outputs")) / config["task"] / config["architecture"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
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
                batch = {key: value.to(device) for key, value in raw_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss, components = compute_loss(outputs, batch, config["loss"])
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
    }
    save_json(output_dir / "summary.json", summary)
    if bool(training.get("save_checkpoint", True)):
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            output_dir / "checkpoint.pt",
        )
    return summary
