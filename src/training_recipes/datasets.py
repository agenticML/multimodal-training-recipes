"""Deterministic canonical datasets for smoke and local-manifest training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dummy import (
    TINY_AUDIO_LEN,
    TINY_CROSS_DIM,
    TINY_FRAMES,
    TINY_HIDDEN,
    TINY_IMAGE,
    TINY_IN_CHANNELS,
    TINY_LATENT,
    TINY_MEL_BINS,
    TINY_MEL_TIME,
    TINY_SEQ,
    TINY_TIME,
    TINY_TS_CHANNELS,
    TINY_VOCAB,
)
from .models import CLASSIFICATION_TASKS, IMAGE_TASKS, TOKEN_TASKS, VIDEO_TASKS

_CTC_CLASSES = {"Wav2Vec2ForCTC", "HubertForCTC", "Data2VecAudioForCTC"}
_SPEECH2TEXT_CLASSES = {"Speech2TextForConditionalGeneration"}


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


class DummyHubDataset(Dataset):
    """HF-shaped dummy tensors keyed by task / Hub class, not the surrogate contract."""

    def __init__(self, config: dict):
        self.config = config
        self.task = str(config["task"])
        source = config.get("source") or {}
        self.class_name = str(source.get("class", ""))
        data = config.get("data") or {}
        self.length = int(data.get("length", 8))
        self.sequence_length = int(data.get("sequence_length", TINY_SEQ))
        self.vocab_size = int(data.get("vocab_size", TINY_VOCAB))
        self.num_classes = int(data.get("num_classes", 4))
        self.seed = int(data.get("seed", 42))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        return dummy_hub_item(
            self.task,
            self.class_name,
            index,
            generator,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
        )


def dummy_hub_item(
    task: str,
    class_name: str,
    index: int,
    generator: torch.Generator,
    *,
    sequence_length: int = TINY_SEQ,
    vocab_size: int = TINY_VOCAB,
    num_classes: int = 4,
) -> dict[str, torch.Tensor]:
    if task == "asr":
        return _asr_item(class_name, generator, sequence_length, vocab_size)
    if task in {"translation", "llm"}:
        return _text_item(task, generator, sequence_length, vocab_size)
    if task == "image_classification":
        return _image_classification_item(generator, index, num_classes)
    if task == "video_classification":
        return _video_classification_item(class_name, generator, index, num_classes, vocab_size)
    if task == "image_captioning":
        return _image_caption_item(class_name, generator, sequence_length, vocab_size)
    if task == "video_captioning":
        return _video_caption_item(class_name, generator, sequence_length, vocab_size)
    if task == "biomedical_signal_processing":
        return _biomedical_item(class_name, generator, index, num_classes)
    if task in {"tts", "speech_to_speech"}:
        return _speech_item(task, class_name, generator, sequence_length, vocab_size)
    if task in {"image_generation", "image_inpainting", "text_to_image"}:
        return _image_diffusion_item(task, class_name, generator)
    if task in {"video_generation", "text_to_video"}:
        return _video_diffusion_item(class_name, generator)
    raise ValueError(f"no dummy Hub tensors for task {task!r}")


def _ids(generator: torch.Generator, length: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(0, max(2, vocab_size), (length,), generator=generator)


def _asr_item(
    class_name: str, generator: torch.Generator, sequence_length: int, vocab_size: int
) -> dict[str, torch.Tensor]:
    labels = _ids(generator, sequence_length, vocab_size)
    if class_name in _CTC_CLASSES:
        values = torch.randn(TINY_AUDIO_LEN, generator=generator)
        return {"input_values": values, "labels": labels, "targets": labels}
    time_steps = 256 if "Whisper" in class_name else TINY_MEL_TIME
    features = torch.randn(TINY_MEL_BINS, time_steps, generator=generator)
    if class_name in _SPEECH2TEXT_CLASSES:
        features = features.transpose(0, 1).contiguous()
    return {"input_features": features, "labels": labels, "targets": labels}


def _text_item(
    task: str, generator: torch.Generator, sequence_length: int, vocab_size: int
) -> dict[str, torch.Tensor]:
    input_ids = _ids(generator, sequence_length, vocab_size)
    labels = torch.roll(input_ids, shifts=-1) if task == "llm" else _ids(
        generator, sequence_length, vocab_size
    )
    attention_mask = torch.ones_like(input_ids)
    decoder_ids = _ids(generator, sequence_length, vocab_size)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_ids,
        "labels": labels,
        "targets": labels,
    }


def _image_classification_item(
    generator: torch.Generator, index: int, num_classes: int
) -> dict[str, torch.Tensor]:
    pixels = torch.randn(3, TINY_IMAGE, TINY_IMAGE, generator=generator)
    labels = torch.tensor(index % num_classes)
    return {"pixel_values": pixels, "labels": labels, "targets": labels}


def _video_classification_item(
    class_name: str,
    generator: torch.Generator,
    index: int,
    num_classes: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    labels = torch.tensor(index % num_classes)
    if class_name == "mvit_v2_s":
        video = torch.randn(3, TINY_FRAMES, TINY_IMAGE, TINY_IMAGE, generator=generator)
        return {"pixel_values": video, "video": video, "labels": labels, "targets": labels}
    pixels = torch.randn(TINY_FRAMES, 3, TINY_IMAGE, TINY_IMAGE, generator=generator)
    item = {"pixel_values": pixels, "labels": labels, "targets": labels}
    if class_name == "XCLIPModel":
        text = _ids(generator, TINY_SEQ, vocab_size)
        item["input_ids"] = text
        item["attention_mask"] = torch.ones_like(text)
    return item


def _image_caption_item(
    class_name: str, generator: torch.Generator, sequence_length: int, vocab_size: int
) -> dict[str, torch.Tensor]:
    pixels = torch.randn(3, TINY_IMAGE, TINY_IMAGE, generator=generator)
    labels = _ids(generator, sequence_length, vocab_size)
    item = {
        "pixel_values": pixels,
        "input_ids": labels,
        "decoder_input_ids": labels,
        "labels": labels,
        "targets": labels,
        "attention_mask": torch.ones_like(labels),
    }
    if "InstructBlip" in class_name or "Blip2" in class_name:
        qformer = _ids(generator, sequence_length, vocab_size)
        item["qformer_input_ids"] = qformer
        item["qformer_attention_mask"] = torch.ones_like(qformer)
    return item


def _video_caption_item(
    class_name: str, generator: torch.Generator, sequence_length: int, vocab_size: int
) -> dict[str, torch.Tensor]:
    frames = torch.randn(TINY_FRAMES, 3, TINY_IMAGE, TINY_IMAGE, generator=generator)
    labels = _ids(generator, sequence_length, vocab_size)
    item = {
        "pixel_values": frames[0],
        "pixel_values_videos": frames,
        "input_ids": labels,
        "decoder_input_ids": labels,
        "labels": labels,
        "targets": labels,
        "attention_mask": torch.ones_like(labels),
    }
    if class_name == "GitForCausalLM":
        item["pixel_values"] = frames
    if "InstructBlip" in class_name:
        qformer = _ids(generator, sequence_length, vocab_size)
        item["qformer_input_ids"] = qformer
        item["qformer_attention_mask"] = torch.ones_like(qformer)
        item["pixel_values"] = frames[0]
    return item


def _biomedical_item(
    class_name: str, generator: torch.Generator, index: int, num_classes: int
) -> dict[str, torch.Tensor]:
    labels = torch.tensor(index % num_classes)
    if class_name in {"PatchTSTForClassification", "PatchTSMixerForPrediction"}:
        past_values = torch.randn(TINY_TIME, TINY_TS_CHANNELS, generator=generator)
        item = {
            "past_values": past_values,
            "labels": labels,
            "targets": labels,
        }
        if class_name == "PatchTSTForClassification":
            item["target_values"] = labels
        return item
    history = 40
    past_values = torch.randn(history, 1, generator=generator)
    observed = torch.ones_like(past_values)
    time_features = torch.zeros(history, 1)
    future_time = torch.zeros(4, 1)
    future_values = torch.randn(4, 1, generator=generator)
    return {
        "past_values": past_values,
        "past_observed_mask": observed,
        "past_time_features": time_features,
        "future_time_features": future_time,
        "future_values": future_values,
        "labels": labels,
        "targets": labels,
    }


def _speech_item(
    task: str,
    class_name: str,
    generator: torch.Generator,
    sequence_length: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    input_ids = _ids(generator, sequence_length, vocab_size)
    spectrogram = torch.randn(TINY_MEL_TIME, TINY_MEL_BINS, generator=generator)
    audio = torch.randn(TINY_AUDIO_LEN, generator=generator)
    labels = spectrogram if task == "tts" else audio
    duration = torch.ones(sequence_length, dtype=torch.long)
    pitch = torch.zeros(sequence_length)
    energy = torch.zeros(sequence_length)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "input_values": audio,
        "decoder_input_values": spectrogram,
        "labels": labels,
        "targets": labels,
        "spectrogram_labels": spectrogram,
        "duration_labels": duration,
        "pitch_labels": pitch,
        "energy_labels": energy,
        "speaker_embeddings": torch.randn(TINY_HIDDEN, generator=generator),
    }


def _image_diffusion_item(
    task: str, class_name: str, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    channels = (
        3
        if class_name
        in {"UNet2DModel", "DiTTransformer2DModel", "VQModel", "RePaintPipeline"}
        else TINY_IN_CHANNELS
    )
    if "Inpaint" in class_name:
        channels = 9
    if class_name == "ControlNetModel":
        channels = TINY_IN_CHANNELS
    sample = torch.randn(channels, TINY_LATENT, TINY_LATENT, generator=generator)
    noise = torch.randn_like(sample)
    timestep = torch.tensor(1, dtype=torch.long)
    text = torch.randn(TINY_SEQ, TINY_CROSS_DIM, generator=generator)
    item = {
        "sample": sample,
        "hidden_states": sample,
        "timestep": timestep,
        "noise": noise,
        "targets": noise,
    }
    if class_name not in {"UNet2DModel", "VQModel", "RePaintPipeline"}:
        item["encoder_hidden_states"] = text
    if class_name == "DiTTransformer2DModel":
        item["class_labels"] = torch.tensor(0, dtype=torch.long)
    if class_name == "ControlNetModel" or "ControlNet" in class_name:
        item["controlnet_cond"] = torch.randn(3, TINY_LATENT, TINY_LATENT, generator=generator)
        item["encoder_hidden_states"] = text
    if "Cascade" in class_name:
        item["timestep_ratio"] = torch.tensor(0.5)
        item["clip_text_pooled"] = torch.randn(1, TINY_CROSS_DIM, generator=generator)
        item["encoder_hidden_states"] = text
    if class_name == "VQModel":
        item["targets"] = sample
        item["noise"] = sample
    return item


def _video_diffusion_item(class_name: str, generator: torch.Generator) -> dict[str, torch.Tensor]:
    bcthw = torch.randn(
        TINY_IN_CHANNELS, TINY_FRAMES, TINY_LATENT, TINY_LATENT, generator=generator
    )
    bfchw = bcthw.permute(1, 0, 2, 3).contiguous()
    noise = torch.randn_like(bcthw)
    timestep = torch.tensor(1, dtype=torch.long)
    text = torch.randn(TINY_SEQ, TINY_CROSS_DIM, generator=generator)
    return {
        "sample": bcthw,
        "hidden_states": bfchw,
        "timestep": timestep,
        "encoder_hidden_states": text,
        "encoder_attention_mask": torch.ones(TINY_SEQ, dtype=torch.long),
        "noise": noise,
        "targets": noise,
        "num_frames": torch.tensor(TINY_FRAMES, dtype=torch.long),
        "height": torch.tensor(TINY_LATENT, dtype=torch.long),
        "width": torch.tensor(TINY_LATENT, dtype=torch.long),
    }


def create_dataset(task: str, config: dict, dataset_path: str | None = None) -> Dataset:
    if dataset_path:
        return ManifestDataset(task, config, dataset_path)
    return SyntheticDataset(task, config)
