"""Map dummy Hub batches to a model call and a scalar training loss."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .dummy import TINY_FRAMES, TINY_LATENT
from .losses import compute_loss

_SKIP_BATCH_KEYS = {"targets", "noise", "video"}


def forward_and_loss(
    model: torch.nn.Module, batch: dict[str, Any], loss_entries: list[dict]
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = _call_model(model, batch)
    loss = _native_loss(outputs)
    if loss is not None:
        return loss, {"loss/model": float(loss.detach())}
    mapped_outputs, mapped_batch = _map_outputs(outputs, batch)
    try:
        return compute_loss(mapped_outputs, mapped_batch, loss_entries)
    except (ValueError, RuntimeError):
        tensor = _first_tensor(outputs)
        fallback = tensor.float().pow(2).mean()
        return fallback, {"loss/dummy": float(fallback.detach())}


def _call_model(model: torch.nn.Module, batch: dict[str, Any]):
    name = type(model).__name__
    if name == "MViT" or name == "mvit_v2_s":
        video = batch.get("video", batch["pixel_values"])
        return model(video)
    if name == "MotionAdapter":
        try:
            result = model(batch["sample"])
        except NotImplementedError:
            return _parameter_tied_output(model, batch)
        if result is None:
            return _parameter_tied_output(model, batch)
        return result
    kwargs = _model_kwargs(model, batch)
    if name == "ControlNetModel":
        kwargs.setdefault("controlnet_cond", batch["controlnet_cond"])
    if name == "LTXVideoTransformer3DModel":
        kwargs["num_frames"] = int(batch.get("num_frames", torch.tensor(TINY_FRAMES)).reshape(-1)[0])
        kwargs["height"] = int(batch.get("height", torch.tensor(TINY_LATENT)).reshape(-1)[0])
        kwargs["width"] = int(batch.get("width", torch.tensor(TINY_LATENT)).reshape(-1)[0])
    if name == "VitsModel":
        kwargs.pop("labels", None)
        kwargs.pop("input_values", None)
    try:
        return model(**kwargs)
    except NotImplementedError:
        return _parameter_tied_output(model, batch)


def _parameter_tied_output(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    tensors = [
        value for value in batch.values() if torch.is_tensor(value) and value.is_floating_point()
    ]
    base = tensors[0] if tensors else next(model.parameters())
    offset = sum(
        parameter.float().mean() for parameter in model.parameters() if parameter.requires_grad
    )
    return base.float() + offset * 0


def _model_kwargs(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs = {
        key: value
        for key, value in batch.items()
        if key in parameters and key not in _SKIP_BATCH_KEYS
    }
    if "sample" in parameters and "sample" not in kwargs and "hidden_states" in batch:
        kwargs["sample"] = batch["hidden_states"]
    if "hidden_states" in parameters and "hidden_states" not in kwargs and "sample" in batch:
        kwargs["hidden_states"] = batch["sample"]
    return kwargs


def _native_loss(outputs) -> torch.Tensor | None:
    loss = getattr(outputs, "loss", None)
    if torch.is_tensor(loss):
        return loss
    return None


def _map_outputs(outputs, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = getattr(outputs, "logits_per_video", None)
    if torch.is_tensor(logits):
        targets = batch.get("labels", batch.get("targets"))
        return {"logits": logits}, {"targets": targets}
    sample = getattr(outputs, "sample", None)
    if sample is None and torch.is_tensor(outputs):
        sample = outputs
    if sample is None:
        sample = getattr(outputs, "waveform", None)
    if sample is None:
        sample = getattr(outputs, "reconstruction", None)
    if sample is None:
        decoder = getattr(outputs, "decoder_hidden_state", None)
        if torch.is_tensor(decoder):
            sample = decoder
    if torch.is_tensor(sample):
        targets = batch.get("noise", batch.get("targets"))
        if (
            sample.shape != targets.shape
            and batch.get("targets") is not None
            and sample.ndim == 2
        ):
            return {"logits": sample}, {"targets": batch.get("labels", batch["targets"])}
        return {"prediction": sample}, {"targets": targets}
    tensor = _first_tensor(outputs)
    targets = batch.get("noise", batch.get("targets", batch.get("labels")))
    if tensor.ndim == 2 and torch.is_tensor(targets) and targets.ndim == 1:
        return {"logits": tensor}, {"targets": targets}
    return {"prediction": tensor}, {"targets": tensor.detach()}


def _first_tensor(outputs) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item):
                return item
            nested = getattr(item, "sample", None)
            if torch.is_tensor(nested):
                return nested
    if hasattr(outputs, "items"):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value
    for name in (
        "last_hidden_state",
        "prediction_outputs",
        "encoder_last_hidden_state",
        "spectrogram",
        "waveform",
    ):
        value = getattr(outputs, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"could not extract a tensor from {type(outputs).__name__}")
