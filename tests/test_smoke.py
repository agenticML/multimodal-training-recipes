from training_recipes.config import load_document, resolve_config
from training_recipes.registry import architecture_catalog, recipe_path
from training_recipes.trainer import train


def test_representative_smoke_runs(tmp_path):
    for task in ("asr", "image_inpainting", "video_classification", "audio_super_resolution"):
        architecture = next(iter(architecture_catalog(task)))
        config = resolve_config(load_document(recipe_path(task, architecture)), "smoke")
        config["training"]["max_steps"] = 1
        config["training"]["save_checkpoint"] = False
        config["output_dir"] = str(tmp_path)
        summary = train(config)
        assert summary["steps"] == 1
        assert summary["final_loss"] == summary["final_loss"]
