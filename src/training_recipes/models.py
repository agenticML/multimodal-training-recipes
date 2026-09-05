"""Canonical smoke models and Hugging Face source loader."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

from .dummy import (
    TINY_CROSS_DIM,
    TINY_FFN,
    TINY_FRAMES,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_IMAGE,
    TINY_IN_CHANNELS,
    TINY_LATENT,
    TINY_LAYERS,
    TINY_MEL_TIME,
    TINY_PATCH,
    TINY_TIME,
    TINY_TS_CHANNELS,
    TINY_VOCAB,
    TINY_WINDOW,
)

TOKEN_TASKS = {"asr", "translation", "image_captioning", "video_captioning", "llm"}
IMAGE_TASKS = {"image_generation", "image_inpainting", "text_to_image"}
VIDEO_TASKS = {"video_generation", "text_to_video"}
CLASSIFICATION_TASKS = {
    "biomedical_signal_processing",
    "video_classification",
    "image_classification",
}

UNSUPPORTED_DUMMY_LIBRARIES = frozenset({"custom", "mlx"})
UNSUPPORTED_DUMMY_ARCHITECTURES = frozenset(
    {
        ("audio_super_resolution", "audiosr"),
        ("audio_super_resolution", "flashsr"),
        ("audio_super_resolution", "nuwave2"),
        ("audio_super_resolution", "aero"),
        ("audio_super_resolution", "universr"),
        ("image_generation", "stylegan2_ada"),
    }
)

_PIPELINE_CLASSES = frozenset(
    {
        "StableDiffusionPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionXLPipeline",
        "StableDiffusionXLInpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22InpaintPipeline",
        "PixArtAlphaPipeline",
        "StableCascadeCombinedPipeline",
        "RePaintPipeline",
    }
)

_NESTED_CONFIG_ATTRS = (
    "text_config",
    "vision_config",
    "qformer_config",
    "audio_config",
    "encoder",
    "decoder",
    "encoder_config",
    "decoder_config",
    "language_config",
    "aligner_config",
    "codec_config",
    "semantic_config",
    "coarse_acoustics_config",
    "fine_acoustics_config",
    "thinker_config",
    "talker_config",
    "token2wav_config",
    "composer_config",
    "unet_config",
    "vqgan_config",
)



class UnsupportedDummyArchitecture(RuntimeError):
    """Raised when a recipe has no importable Hub class for tiny dummy training."""


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
    """Surrogate modules retained for loss unit tests, not the trainer path."""
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


def _resolve_source(config: dict[str, Any]) -> dict[str, Any]:
    source = config.get("source")
    if source:
        return source
    from .registry import architecture_spec

    spec = architecture_spec(config["task"], config["architecture"])
    config["source"] = spec["source"]
    config.setdefault("task_kind", spec.get("task_kind"))
    config.setdefault("family", spec.get("family"))
    return spec["source"]


def create_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate the catalog Hub class from a tiny config (no weight download)."""
    source = _resolve_source(config)
    library = str(source.get("library", ""))
    class_name = str(source.get("class", ""))
    task = str(config["task"])
    architecture = str(config["architecture"])
    if (
        library in UNSUPPORTED_DUMMY_LIBRARIES
        or (task, architecture) in UNSUPPORTED_DUMMY_ARCHITECTURES
    ):
        raise UnsupportedDummyArchitecture(
            f"{task}/{architecture} uses library {library!r} ({class_name}); "
            "dummy runs skip custom/mlx recipes"
        )
    load_mode = str(config.get("model", {}).get("load_mode", "tiny_config"))
    if load_mode == "pretrained":
        return load_huggingface_model(source)
    return _tiny_from_config(library, class_name, config)


def _tiny_from_config(library: str, class_name: str, config: dict[str, Any]) -> nn.Module:
    if library == "torchvision" or class_name == "mvit_v2_s":
        return _tiny_torchvision(class_name, config)
    cls = _resolve_hub_class(library, class_name)
    module_name = getattr(cls, "__module__", "")
    if "diffusers" in module_name or library == "diffusers":
        return _tiny_diffusers(cls, class_name)
    return _tiny_transformers(cls, config)


def _resolve_hub_class(library: str, class_name: str):
    try:
        module = importlib.import_module(library)
    except ImportError as exc:
        raise ImportError(
            f"Dummy architecture {class_name} requires {library}. "
            "Install transformers and diffusers (main project dependencies)."
        ) from exc
    if hasattr(module, class_name):
        return getattr(module, class_name)
    if library == "transformers":
        diffusers = importlib.import_module("diffusers")
        if hasattr(diffusers, class_name):
            return getattr(diffusers, class_name)
    raise AttributeError(f"{library} has no class {class_name}")


def _tiny_torchvision(class_name: str, config: dict[str, Any]) -> nn.Module:
    try:
        from torchvision.models.video import mvit_v2_s
    except ImportError as exc:
        raise ImportError("video_classification/mvit requires torchvision") from exc
    if class_name != "mvit_v2_s":
        raise AttributeError(f"unsupported torchvision architecture {class_name}")
    num_classes = int(config.get("data", {}).get("num_classes", 4))
    return mvit_v2_s(weights=None, num_classes=num_classes)


def _tiny_transformers(cls, config: dict[str, Any]) -> nn.Module:
    data = config.get("data") or {}
    cfg = _tiny_transformers_config(cls, data, task=str(config.get("task", "")))
    if hasattr(cls, "from_config"):
        model = cls.from_config(cfg)
    elif hasattr(cls, "_from_config"):
        model = cls._from_config(cfg)
    else:
        model = cls(cfg)
    model.train()
    return model


def _tiny_transformers_config(cls, data: dict[str, Any], task: str = ""):
    name = cls.__name__
    if name == "VisionEncoderDecoderModel":
        return _vision_encoder_decoder_config(data)
    if name == "Blip2ForConditionalGeneration":
        return _blip2_config(data)
    if name == "InstructBlipForConditionalGeneration":
        return _instructblip_config(data)
    config_cls = getattr(cls, "config_class", None)
    if config_cls is None:
        raise TypeError(f"{name} has no config_class for tiny from_config construction")
    cfg = config_cls()
    _shrink_config(cfg, data, seen=set())
    _apply_class_specific_config(name, cfg, data, task=task)
    return cfg


def _vocab(data: dict[str, Any]) -> int:
    return int(data.get("vocab_size", TINY_VOCAB))


def _num_classes(data: dict[str, Any]) -> int:
    return int(data.get("num_classes", 4))


def _vision_encoder_decoder_config(data: dict[str, Any]):
    transformers = importlib.import_module("transformers")
    vocab = _vocab(data)
    encoder = transformers.ViTConfig(
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        intermediate_size=TINY_FFN,
        image_size=TINY_IMAGE,
        patch_size=TINY_PATCH,
        num_channels=3,
        encoder_stride=TINY_PATCH,
    )
    decoder = transformers.GPT2Config(
        vocab_size=vocab,
        n_embd=TINY_HIDDEN,
        n_layer=TINY_LAYERS,
        n_head=TINY_HEADS,
        n_inner=TINY_FFN,
        n_positions=32,
        n_ctx=32,
        add_cross_attention=True,
        is_decoder=True,
    )
    return transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)


def _blip2_config(data: dict[str, Any]):
    transformers = importlib.import_module("transformers")
    if not hasattr(transformers.Blip2Config, "from_vision_qformer_text_configs"):
        cfg = transformers.Blip2Config()
        _shrink_config(cfg, data, seen=set())
        return cfg
    vocab = _vocab(data)
    vision = transformers.Blip2VisionConfig(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_FFN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        image_size=TINY_IMAGE,
        patch_size=TINY_PATCH,
    )
    qformer = transformers.Blip2QFormerConfig(
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        intermediate_size=TINY_FFN,
        vocab_size=vocab,
    )
    text = transformers.OPTConfig(
        vocab_size=vocab,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        ffn_dim=TINY_FFN,
        num_attention_heads=TINY_HEADS,
        word_embed_proj_dim=TINY_HIDDEN,
    )
    return transformers.Blip2Config.from_vision_qformer_text_configs(vision, qformer, text)


def _instructblip_config(data: dict[str, Any]):
    transformers = importlib.import_module("transformers")
    if not hasattr(transformers.InstructBlipConfig, "from_vision_qformer_text_configs"):
        cfg = transformers.InstructBlipConfig()
        _shrink_config(cfg, data, seen=set())
        return cfg
    vocab = _vocab(data)
    vision = transformers.InstructBlipVisionConfig(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_FFN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        image_size=TINY_IMAGE,
        patch_size=TINY_PATCH,
    )
    qformer = transformers.InstructBlipQFormerConfig(
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        intermediate_size=TINY_FFN,
        vocab_size=vocab,
    )
    text = transformers.T5Config(
        vocab_size=vocab,
        d_model=TINY_HIDDEN,
        d_kv=TINY_HEAD_DIM,
        d_ff=TINY_FFN,
        num_layers=TINY_LAYERS,
        num_decoder_layers=TINY_LAYERS,
        num_heads=TINY_HEADS,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return transformers.InstructBlipConfig.from_vision_qformer_text_configs(vision, qformer, text)


def _is_config(obj: Any) -> bool:
    return hasattr(obj, "to_dict") and hasattr(obj, "model_type") and not isinstance(obj, type)


def _shrink_config(cfg: Any, data: dict[str, Any], seen: set[int]) -> None:
    if cfg is None or not _is_config(cfg):
        return
    ident = id(cfg)
    if ident in seen:
        return
    seen.add(ident)
    for attr in _NESTED_CONFIG_ATTRS:
        child = getattr(cfg, attr, None)
        if _is_config(child):
            _shrink_config(child, data, seen)
    _assign_tiny_fields(cfg, data)
    _shrink_feature_extractor(cfg)
    _align_attention(cfg)
    _clamp_token_ids(cfg)


def _assign_tiny_fields(cfg: Any, data: dict[str, Any]) -> None:
    vocab = _vocab(data)
    classes = _num_classes(data)
    scalars: dict[str, Any] = {
        "vocab_size": vocab,
        "decoder_vocab_size": vocab,
        "encoder_vocab_size": vocab,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 0,
        "num_hidden_layers": TINY_LAYERS,
        "num_layers": TINY_LAYERS,
        "n_layer": TINY_LAYERS,
        "n_layers": TINY_LAYERS,
        "encoder_layers": TINY_LAYERS,
        "decoder_layers": TINY_LAYERS,
        "num_encoder_layers": TINY_LAYERS,
        "num_decoder_layers": TINY_LAYERS,
        "num_hidden_groups": TINY_LAYERS,
        "num_attention_heads": TINY_HEADS,
        "num_heads": TINY_HEADS,
        "n_head": TINY_HEADS,
        "n_heads": TINY_HEADS,
        "encoder_attention_heads": TINY_HEADS,
        "decoder_attention_heads": TINY_HEADS,
        "num_key_value_heads": TINY_HEADS,
        "hidden_size": TINY_HIDDEN,
        "d_model": TINY_HIDDEN,
        "n_embd": TINY_HIDDEN,
        "encoder_hidden_size": TINY_HIDDEN,
        "decoder_hidden_size": TINY_HIDDEN,
        "word_embed_proj_dim": TINY_HIDDEN,
        "intermediate_size": TINY_FFN,
        "encoder_ffn_dim": TINY_FFN,
        "decoder_ffn_dim": TINY_FFN,
        "ffn_dim": TINY_FFN,
        "d_ff": TINY_FFN,
        "n_inner": TINY_FFN,
        "filter_size": TINY_FFN,
        "d_kv": TINY_HEAD_DIM,
        "head_dim": TINY_HEAD_DIM,
        "query_pre_attn_scalar": TINY_HEAD_DIM,
        "n_positions": 32,
        "n_ctx": 32,
        "max_position_embeddings": 64,
        "max_target_positions": 64,
        "max_source_positions": 64,
        "max_mel_frames": TINY_MEL_TIME,
        "num_labels": classes,
        "num_targets": classes,
        "id2label": {i: str(i) for i in range(classes)},
        "label2id": {str(i): i for i in range(classes)},
        "image_size": TINY_IMAGE,
        "patch_size": TINY_PATCH,
        "encoder_stride": TINY_PATCH,
        "num_channels": 3,
        "num_image_channels": 3,
        "num_frames": TINY_FRAMES,
        "tubelet_size": 2,
        "num_mel_bins": 80,
        "num_feat": 80,
        "input_feat_per_channel": 80,
        "num_conv_pos_embeddings": 16,
        "mask_time_prob": 0.0,
        "mask_feature_prob": 0.0,
        "context_length": TINY_TIME,
        "prediction_length": 4,
        "patch_length": 4,
        "patch_stride": 4,
        "num_input_channels": TINY_TS_CHANNELS,
        "speaker_embedding_dim": TINY_HIDDEN,
        "sliding_window": 8,
        "window_size": TINY_WINDOW,
    }
    for key, value in scalars.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        if isinstance(current, bool):
            continue
        if current is None:
            if key in {"context_length", "prediction_length", "speaker_embedding_dim"}:
                setattr(cfg, key, value)
            continue
        if isinstance(current, (list, tuple)):
            fill = _list_fill(key)
            if fill is None:
                continue
            setattr(cfg, key, type(current)(fill for _ in current))
            continue
        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
            setattr(cfg, key, type(current)(value))
            continue
        if isinstance(value, dict):
            setattr(cfg, key, value)


def _list_fill(key: str) -> int | None:
    if key in {"depths", "num_hidden_layers"}:
        return TINY_LAYERS
    if key in {"hidden_sizes", "embed_dims"}:
        return TINY_HIDDEN
    if key in {"num_heads", "num_attention_heads"}:
        return 2
    return None


def _shrink_feature_extractor(cfg: Any) -> None:
    if not hasattr(cfg, "conv_dim"):
        return
    cfg.conv_dim = (TINY_HIDDEN, TINY_HIDDEN)
    if hasattr(cfg, "conv_stride"):
        cfg.conv_stride = (2, 2)
    if hasattr(cfg, "conv_kernel"):
        cfg.conv_kernel = (3, 3)
    if hasattr(cfg, "conv_bias") and isinstance(cfg.conv_bias, (list, tuple)):
        cfg.conv_bias = (False, False)
    if hasattr(cfg, "num_feat_extract_layers"):
        cfg.num_feat_extract_layers = len(cfg.conv_dim)


def _clamp_token_ids(cfg: Any) -> None:
    vocab = getattr(cfg, "vocab_size", None)
    if not isinstance(vocab, int) or vocab <= 0:
        return
    for key in (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        "sep_token_id",
        "cls_token_id",
    ):
        value = getattr(cfg, key, None)
        if isinstance(value, int) and value >= vocab:
            setattr(cfg, key, 0)


def _align_attention(cfg: Any) -> None:
    hidden = None
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            hidden = value
            break
    if hidden is None:
        return
    for head_key in (
        "num_attention_heads",
        "num_heads",
        "n_head",
        "encoder_attention_heads",
        "decoder_attention_heads",
    ):
        heads = getattr(cfg, head_key, None)
        if isinstance(heads, int) and heads > 0 and hidden % heads != 0:
            setattr(cfg, head_key, next(h for h in (4, 2, 1) if hidden % h == 0))
    heads = getattr(cfg, "num_attention_heads", None)
    kv = getattr(cfg, "num_key_value_heads", None)
    if isinstance(heads, int) and isinstance(kv, int) and heads % kv != 0:
        cfg.num_key_value_heads = heads
    head_dim = getattr(cfg, "head_dim", None)
    if (
        isinstance(head_dim, int)
        and isinstance(heads, int)
        and heads
        and heads * head_dim != hidden
        and hasattr(cfg, "head_dim")
    ):
        cfg.head_dim = max(1, hidden // heads)


def _apply_class_specific_config(name: str, cfg: Any, data: dict[str, Any], task: str = "") -> None:
    if name == "SwinForImageClassification":
        cfg.image_size = TINY_IMAGE
        cfg.patch_size = 4
        cfg.window_size = TINY_WINDOW
        cfg.embed_dim = TINY_HIDDEN
        cfg.depths = (1, 1)
        cfg.num_heads = (2, 2)
    elif name == "ConvNextForImageClassification":
        cfg.num_channels = 3
        if hasattr(cfg, "hidden_sizes") and isinstance(cfg.hidden_sizes, (list, tuple)):
            cfg.hidden_sizes = type(cfg.hidden_sizes)(TINY_HIDDEN for _ in cfg.hidden_sizes)
        if hasattr(cfg, "depths") and isinstance(cfg.depths, (list, tuple)):
            cfg.depths = type(cfg.depths)(1 for _ in cfg.depths)
        cfg.image_size = TINY_IMAGE
    elif name == "GitForCausalLM":
        if task == "video_captioning" and hasattr(cfg, "num_image_with_embedding"):
            cfg.num_image_with_embedding = TINY_FRAMES
        vision = getattr(cfg, "vision_config", None)
        if vision is not None:
            vision.image_size = TINY_IMAGE
            vision.patch_size = TINY_PATCH
            vision.hidden_size = TINY_HIDDEN
            vision.num_hidden_layers = TINY_LAYERS
            vision.num_attention_heads = TINY_HEADS
    elif name == "WhisperForConditionalGeneration":
        cfg.max_source_positions = 128
        cfg.d_model = TINY_HIDDEN
        cfg.encoder_layers = TINY_LAYERS
        cfg.decoder_layers = TINY_LAYERS
        cfg.encoder_attention_heads = TINY_HEADS
        cfg.decoder_attention_heads = TINY_HEADS
        cfg.encoder_ffn_dim = TINY_FFN
        cfg.decoder_ffn_dim = TINY_FFN
        cfg.num_mel_bins = 80
    elif name in {"TimeSeriesTransformerModel", "InformerModel", "AutoformerModel"}:
        cfg.prediction_length = 4
        cfg.context_length = 32
        cfg.input_size = 1
        cfg.lags_sequence = [1]
        cfg.num_time_features = 1
        cfg.num_static_categorical_features = 0
        cfg.num_static_real_features = 0
        cfg.num_dynamic_real_features = 0
        cfg.d_model = TINY_HIDDEN
        cfg.encoder_layers = TINY_LAYERS
        cfg.decoder_layers = TINY_LAYERS
        cfg.encoder_attention_heads = TINY_HEADS
        cfg.decoder_attention_heads = TINY_HEADS
        cfg.encoder_ffn_dim = TINY_FFN
        cfg.decoder_ffn_dim = TINY_FFN
        if hasattr(cfg, "cardinality"):
            cfg.cardinality = []
        if hasattr(cfg, "embedding_dimension"):
            cfg.embedding_dimension = []
    elif name in {"PatchTSTForClassification", "PatchTSMixerForPrediction"}:
        cfg.num_input_channels = TINY_TS_CHANNELS
        cfg.context_length = TINY_TIME
        cfg.patch_length = 4
        cfg.patch_stride = 4
        cfg.d_model = TINY_HIDDEN
        cfg.num_hidden_layers = TINY_LAYERS
        cfg.num_attention_heads = TINY_HEADS
        cfg.ffn_dim = TINY_FFN
        if hasattr(cfg, "num_targets"):
            cfg.num_targets = _num_classes(data)
        if hasattr(cfg, "prediction_length"):
            cfg.prediction_length = 4
    elif name == "XCLIPModel":
        text = getattr(cfg, "text_config", None)
        vision = getattr(cfg, "vision_config", None)
        if text is not None:
            text.hidden_size = TINY_HIDDEN
            text.num_hidden_layers = TINY_LAYERS
            text.num_attention_heads = TINY_HEADS
            text.intermediate_size = TINY_FFN
            text.vocab_size = _vocab(data)
            text.max_position_embeddings = 32
        if vision is not None:
            vision.hidden_size = TINY_HIDDEN
            vision.num_hidden_layers = TINY_LAYERS
            vision.num_attention_heads = TINY_HEADS
            vision.intermediate_size = TINY_FFN
            vision.image_size = TINY_IMAGE
            vision.patch_size = TINY_PATCH
            vision.num_frames = TINY_FRAMES


def _tiny_diffusers(cls, class_name: str) -> nn.Module:
    import diffusers

    if class_name in _PIPELINE_CLASSES:
        return _tiny_pipeline_trainable(diffusers, class_name)
    kwargs = _diffusers_tiny_kwargs(class_name)
    model = _instantiate_diffusers(cls, kwargs)
    model.train()
    return model


def _instantiate_diffusers(cls, kwargs: dict[str, Any]):
    try:
        return cls(**kwargs)
    except TypeError:
        return cls.from_config(kwargs)


def _tiny_pipeline_trainable(diffusers, class_name: str) -> nn.Module:
    if class_name in {
        "PixArtAlphaPipeline",
    }:
        model = _instantiate_diffusers(diffusers.Transformer2DModel, _transformer2d_kwargs())
    elif class_name == "StableCascadeCombinedPipeline":
        model = _instantiate_diffusers(diffusers.StableCascadeUNet, _cascade_kwargs())
    elif class_name == "RePaintPipeline":
        model = _instantiate_diffusers(diffusers.UNet2DModel, _unet2d_kwargs())
    elif "Inpaint" in class_name:
        model = _instantiate_diffusers(
            diffusers.UNet2DConditionModel, _unet2d_cond_kwargs(in_channels=9)
        )
    else:
        model = _instantiate_diffusers(diffusers.UNet2DConditionModel, _unet2d_cond_kwargs())
    model.train()
    return model


def _unet2d_kwargs() -> dict[str, Any]:
    return {
        "sample_size": TINY_LATENT,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 1,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
        "attention_head_dim": TINY_HEAD_DIM,
    }


def _unet2d_cond_kwargs(in_channels: int = TINY_IN_CHANNELS) -> dict[str, Any]:
    return {
        "sample_size": TINY_LATENT,
        "in_channels": in_channels,
        "out_channels": TINY_IN_CHANNELS,
        "layers_per_block": 1,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": TINY_CROSS_DIM,
        "attention_head_dim": TINY_HEAD_DIM,
        "norm_num_groups": 8,
    }


def _transformer2d_kwargs() -> dict[str, Any]:
    return {
        "sample_size": TINY_LATENT,
        "num_layers": TINY_LAYERS,
        "attention_head_dim": TINY_HEAD_DIM,
        "num_attention_heads": TINY_HEADS,
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "patch_size": 2,
        "num_embeds_ada_norm": 1000,
        "cross_attention_dim": TINY_CROSS_DIM,
        "activation_fn": "gelu-approximate",
        "caption_channels": TINY_CROSS_DIM,
        "norm_num_groups": 8,
        "norm_type": "ada_norm_single",
    }


def _dit_kwargs() -> dict[str, Any]:
    return {
        "sample_size": TINY_LATENT,
        "num_layers": TINY_LAYERS,
        "attention_head_dim": TINY_HEAD_DIM,
        "num_attention_heads": TINY_HEADS,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": 2,
        "num_embeds_ada_norm": 1000,
        "activation_fn": "gelu-approximate",
        "norm_type": "ada_norm_zero",
    }


def _unet3d_kwargs() -> dict[str, Any]:
    return {
        "sample_size": TINY_LATENT,
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "layers_per_block": 1,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "down_block_types": ("CrossAttnDownBlock3D", "DownBlock3D"),
        "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
        "cross_attention_dim": TINY_CROSS_DIM,
        "attention_head_dim": TINY_HEAD_DIM,
        "norm_num_groups": 8,
    }


def _motion_adapter_kwargs() -> dict[str, Any]:
    return {
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "motion_layers_per_block": 1,
        "motion_mid_block_layers_per_block": 1,
        "motion_num_attention_heads": 2,
        "use_motion_mid_block": False,
        "conv_in_channels": None,
    }


def _vq_kwargs() -> dict[str, Any]:
    return {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": TINY_IN_CHANNELS,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        "layers_per_block": 1,
        "num_vq_embeddings": 32,
        "vq_embed_dim": TINY_HIDDEN,
        "norm_num_groups": 8,
        "sample_size": TINY_LATENT,
    }


def _controlnet_kwargs() -> dict[str, Any]:
    return {
        "in_channels": TINY_IN_CHANNELS,
        "conditioning_channels": 3,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "layers_per_block": 1,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "cross_attention_dim": TINY_CROSS_DIM,
        "attention_head_dim": TINY_HEAD_DIM,
        "conditioning_embedding_out_channels": (16, 32),
        "norm_num_groups": 8,
    }


def _cascade_kwargs() -> dict[str, Any]:
    return {
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "conditioning_dim": TINY_CROSS_DIM,
        "block_out_channels": (TINY_HIDDEN, TINY_HIDDEN),
        "num_attention_heads": (TINY_HEADS, TINY_HEADS),
        "down_num_layers_per_block": (1, 1),
        "up_num_layers_per_block": (1, 1),
        "clip_text_pooled_in_channels": TINY_CROSS_DIM,
        "timestep_ratio_embedding_dim": TINY_HIDDEN,
        "timestep_conditioning_type": ("sca",),
    }


def _cogvideox_kwargs() -> dict[str, Any]:
    return {
        "num_attention_heads": TINY_HEADS,
        "attention_head_dim": TINY_HEAD_DIM,
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "time_embed_dim": TINY_HIDDEN,
        "text_embed_dim": TINY_CROSS_DIM,
        "num_layers": TINY_LAYERS,
        "sample_width": TINY_LATENT,
        "sample_height": TINY_LATENT,
        "sample_frames": TINY_FRAMES,
        "patch_size": 2,
        "max_text_seq_length": 8,
        "timestep_activation_fn": "silu",
    }


def _ltx_kwargs() -> dict[str, Any]:
    return {
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": TINY_HEADS,
        "attention_head_dim": TINY_HEAD_DIM,
        "cross_attention_dim": TINY_CROSS_DIM,
        "num_layers": TINY_LAYERS,
        "caption_channels": TINY_CROSS_DIM,
    }


def _latte_kwargs() -> dict[str, Any]:
    return {
        "num_attention_heads": TINY_HEADS,
        "attention_head_dim": TINY_HEAD_DIM,
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "num_layers": TINY_LAYERS,
        "sample_size": TINY_LATENT,
        "patch_size": 2,
        "num_embeds_ada_norm": 1000,
        "caption_channels": TINY_CROSS_DIM,
        "video_length": TINY_FRAMES,
        "cross_attention_dim": TINY_CROSS_DIM,
        "norm_type": "ada_norm_single",
    }


def _mochi_kwargs() -> dict[str, Any]:
    return {
        "patch_size": 2,
        "num_attention_heads": TINY_HEADS,
        "attention_head_dim": TINY_HEAD_DIM,
        "num_layers": TINY_LAYERS,
        "pooled_projection_dim": TINY_CROSS_DIM,
        "in_channels": TINY_IN_CHANNELS,
        "out_channels": TINY_IN_CHANNELS,
        "text_embed_dim": TINY_CROSS_DIM,
        "time_embed_dim": TINY_HIDDEN,
        "max_sequence_length": 8,
    }


def _diffusers_tiny_kwargs(class_name: str) -> dict[str, Any]:
    mapping = {
        "UNet2DModel": _unet2d_kwargs,
        "UNet2DConditionModel": _unet2d_cond_kwargs,
        "DiTTransformer2DModel": _dit_kwargs,
        "UNet3DConditionModel": _unet3d_kwargs,
        "MotionAdapter": _motion_adapter_kwargs,
        "ControlNetModel": _controlnet_kwargs,
        "VQModel": _vq_kwargs,
        "Transformer2DModel": _transformer2d_kwargs,
        "LatteTransformer3DModel": _latte_kwargs,
        "CogVideoXTransformer3DModel": _cogvideox_kwargs,
        "LTXVideoTransformer3DModel": _ltx_kwargs,
        "MochiTransformer3DModel": _mochi_kwargs,
        "StableCascadeUNet": _cascade_kwargs,
    }
    factory = mapping.get(class_name)
    if factory is None:
        return _unet2d_cond_kwargs()
    return factory()
