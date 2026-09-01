"""Deterministic canonical datasets for smoke and local-manifest training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .models import CLASSIFICATION_TASKS, IMAGE_TASKS, TOKEN_TASKS, VIDEO_TASKS


class SyntheticDataset(Dataset):
    def __init__(self, task: str, config: dict):
        self.task = task
        self.length = int(config.get("length", 8))
        self.sequence_length = int(config.get("sequence_length", 12))
        self.vocab_size = int(config.get("vocab_size", 32))
        self.num_classes = int(config.get("num_classes", 4))
        self.seed = int(config.get("seed", 42))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        if self.task in TOKEN_TASKS:
            ids = torch.randint(0, self.vocab_size, (self.sequence_length,), generator=generator)
            return {"input_ids": ids, "targets": torch.roll(ids, shifts=-1)}
        if self.task in IMAGE_TASKS:
            target = torch.randn(3, 16, 16, generator=generator)
            inputs = target + 0.2 * torch.randn(3, 16, 16, generator=generator)
            return {"inputs": inputs, "targets": target}
        if self.task in VIDEO_TASKS:
            target = torch.randn(3, 4, 12, 12, generator=generator)
            inputs = target + 0.2 * torch.randn(3, 4, 12, 12, generator=generator)
            return {"inputs": inputs, "targets": target}
        if self.task in CLASSIFICATION_TASKS:
            if self.task == "image_classification":
                inputs = torch.randn(3, 16, 16, generator=generator)
            elif self.task == "video_classification":
                inputs = torch.randn(3, 4, 12, 12, generator=generator)
            else:
                inputs = torch.randn(1, 64, generator=generator)
            return {"inputs": inputs, "targets": torch.tensor(index % self.num_classes)}
        target = torch.randn(1, 128, generator=generator)
        low_resolution = torch.repeat_interleave(target[:, ::2], 2, dim=-1)
        return {"inputs": low_resolution, "targets": target}


class ManifestDataset(SyntheticDataset):
    """Manifest-backed contract validator with deterministic tensorization fallback."""

    def __init__(self, task: str, config: dict, dataset_path: str):
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"dataset manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"dataset manifest is empty: {path}")
        self.manifest_dir = path.parent
        super().__init__(task, {**config, "length": len(self.records)})

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if "tensor_path" in record:
            tensor_path = self.manifest_dir / str(record["tensor_path"])
            value = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not isinstance(value, dict) or not {"inputs", "targets"}.issubset(value):
                raise ValueError(f"{tensor_path} must contain inputs and targets tensors")
            return {"inputs": value["inputs"].float(), "targets": value["targets"]}
        if "input_ids" in record and "targets" in record:
            return {
                "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
                "targets": torch.tensor(record["targets"], dtype=torch.long),
            }
        if "inputs" in record and "targets" in record:
            target_dtype = torch.long if self.task in CLASSIFICATION_TASKS else torch.float
            return {
                "inputs": torch.tensor(record["inputs"], dtype=torch.float),
                "targets": torch.tensor(record["targets"], dtype=target_dtype),
            }
        raise ValueError("manifest records need tensor_path, input_ids+targets, or inputs+targets")


def create_dataset(task: str, config: dict, dataset_path: str | None = None) -> Dataset:
    if dataset_path:
        return ManifestDataset(task, config, dataset_path)
    return SyntheticDataset(task, config)
