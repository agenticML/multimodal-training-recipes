from pathlib import Path

from training_recipes.config import load_document, resolve_config
from training_recipes.registry import architecture_catalog, recipe_path, task_names


def test_catalog_has_all_tasks_and_minimums():
    assert len(task_names()) == 16
    assert sum(len(architecture_catalog(task)) for task in task_names()) == 80
    for task in task_names():
        catalog = architecture_catalog(task)
        assert len(catalog) >= 5
        assert len({loss for spec in catalog.values() for loss in spec["losses"]}) >= 4
        for name, spec in catalog.items():
            assert len(spec["source"]["revision"]) == 40
            assert Path(recipe_path(task, name)).is_file()


def test_every_recipe_resolves_both_profiles():
    for task in task_names():
        for architecture in architecture_catalog(task):
            recipe = load_document(recipe_path(task, architecture))
            assert resolve_config(recipe, "smoke")["profile"] == "smoke"
            assert resolve_config(recipe, "full")["profile"] == "full"
