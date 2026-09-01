"""Canonical smoke models and Hugging Face source loader."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

TOKEN_TASKS = {"asr", "translation", "image_captioning", "video_captioning", "llm"}
IMAGE_TASKS = {"image_generation", "image_inpainting", "text_to_image"}
VIDEO_TASKS = {"video_generation", "text_to_video"}
CLASSIFICATION_TASKS = {
    "biomedical_signal_processing",
    "video_classification",
    "image_classification",
}


class TokenModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "input_ids" in batch:
            features = self.embedding(batch["input_ids"])
        else:
            values = batch["inputs"]
            if values.ndim == 2:
                values = values.unsqueeze(-1)
            pooled = (
                values.mean(dim=tuple(range(1, values.ndim))).long().abs()
                % self.embedding.num_embeddings
            )
            features = self.embedding(pooled[:, None]).repeat(1, batch["targets"].shape[1], 1)
        hidden, _ = self.rnn(features)
        return {"logits": self.head(hidden)}


class DenseRegressor(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        if dimensions == 1:
            self.network = nn.Sequential(
                nn.Conv1d(1, 12, 5, padding=2), nn.SiLU(), nn.Conv1d(12, 1, 3, padding=1)
            )
        elif dimensions == 2:
            self.network = nn.Sequential(
                nn.Conv2d(3, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 3, 3, padding=1)
            )
        else:
            self.network = nn.Sequential(
                nn.Conv3d(3, 8, 3, padding=1), nn.SiLU(), nn.Conv3d(8, 3, 3, padding=1)
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"prediction": self.network(batch["inputs"])}


class Classifier(nn.Module):
    def __init__(self, task: str, num_classes: int = 4):
        super().__init__()
        dimensions = (
            3 if task == "video_classification" else (2 if task == "image_classification" else 1)
        )
        convolution = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimensions]
        pooling = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[
            dimensions
        ]
        output_size = (1,) * dimensions
        self.features = nn.Sequential(
            convolution(3 if dimensions > 1 else 1, 12, 3, padding=1),
            nn.SiLU(),
            pooling(output_size),
        )
        self.head = nn.Linear(12, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.features(batch["inputs"]).flatten(1))}


def create_smoke_model(task: str, config: dict[str, Any]) -> nn.Module:
    if task in TOKEN_TASKS:
        return TokenModel(vocab_size=int(config["data"].get("vocab_size", 32)))
    if task in IMAGE_TASKS:
        return DenseRegressor(2)
    if task in VIDEO_TASKS:
        return DenseRegressor(3)
    if task in CLASSIFICATION_TASKS:
        return Classifier(task, num_classes=int(config["data"].get("num_classes", 4)))
    return DenseRegressor(1)


def load_huggingface_model(source: dict[str, Any], **kwargs):
    """Load a pinned upstream class for a full-profile adapter."""
    module = importlib.import_module(str(source["library"]))
    model_class = getattr(module, str(source["class"]))
    return model_class.from_pretrained(
        str(source["hub_id"]),
        revision=str(source["revision"]),
        trust_remote_code=bool(source.get("trust_remote_code", False)),
        **kwargs,
    )
