import pytest

from training_recipes.config import load_document, resolve_config
from training_recipes.models import UnsupportedDummyArchitecture, create_model
from training_recipes.registry import architecture_catalog, architecture_spec, recipe_path
from training_recipes.trainer import train

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

_FAMILY_SMOKES = (
    ("asr", "wav2vec2", "Wav2Vec2ForCTC"),
    ("translation", "marian", "MarianMTModel"),
    ("llm", "gpt2", "GPT2LMHeadModel"),
    ("image_classification", "vit", "ViTForImageClassification"),
    ("image_generation", "ddpm_unet2d", "UNet2DModel"),
)


def _smoke_config(tmp_path, task: str, architecture: str) -> dict:
    spec = architecture_spec(task, architecture)
    config = resolve_config(load_document(recipe_path(task, architecture)), "smoke")
    config["source"] = spec["source"]
    config["task_kind"] = spec.get("task_kind")
    config["training"]["max_steps"] = 1
    config["training"]["save_checkpoint"] = False
    config["output_dir"] = str(tmp_path)
    config.setdefault("tracking", {})["enabled"] = False
    return config


@pytest.mark.parametrize(("task", "architecture", "class_name"), _FAMILY_SMOKES)
def test_representative_family_smoke_uses_hub_class(tmp_path, task, architecture, class_name):
    config = _smoke_config(tmp_path, task, architecture)
    model = create_model(config)
    assert type(model).__name__ == class_name
    assert type(model).__name__ != "TokenModel"
    summary = train(config)
    assert summary["steps"] == 1
    assert summary["final_loss"] == summary["final_loss"]
    assert summary["model_class"] == class_name


def test_custom_dummy_architecture_is_skipped(tmp_path):
    architecture = next(iter(architecture_catalog("audio_super_resolution")))
    config = _smoke_config(tmp_path, "audio_super_resolution", architecture)
    with pytest.raises(UnsupportedDummyArchitecture, match="custom/mlx"):
        create_model(config)
