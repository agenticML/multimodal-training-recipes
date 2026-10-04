import json
import os
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("transformers")

from training_recipes.config import load_document, resolve_config
from training_recipes.registry import architecture_catalog, recipe_path
from training_recipes.tracking import flatten_hyperparameters, start_run
from training_recipes.trainer import train


def _smoke_config(tmp_path, *, save_checkpoint=True):
    task = "asr"
    architecture = next(iter(architecture_catalog(task)))
    config = resolve_config(load_document(recipe_path(task, architecture)), "smoke")
    config["training"]["max_steps"] = 1
    config["training"]["save_checkpoint"] = save_checkpoint
    config["output_dir"] = str(tmp_path)
    return config, task, architecture


class _RunWithOfflineDir:
    def __init__(self, project, name, mode="online", offline_dir=None, auto_track=True, **kwargs):
        del project, name, mode, offline_dir, auto_track, kwargs


class _RunWithoutOfflineDir:
    def __init__(self, project, name, mode="online", **kwargs):
        del project, name, mode, kwargs


def test_trainer_records_agenticml_run(tmp_path, monkeypatch):
    mock = MagicMock()
    mock.Run = _RunWithOfflineDir
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, task, architecture = _smoke_config(tmp_path)
    expected_offline = str(tmp_path / task / architecture / "agenticml")

    summary = train(config)

    assert summary["steps"] == 1
    mock.init.assert_called_once()
    init_kwargs = mock.init.call_args.kwargs
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["project"] == "multimodal-training-recipes"
    assert init_kwargs["name"] == f"{task}-{architecture}-smoke"
    assert init_kwargs["group"] == task
    assert init_kwargs["job_type"] == "smoke"
    assert init_kwargs["track_source"] == "runtime"
    assert init_kwargs["auto_track"] is False
    assert init_kwargs["offline_dir"] == expected_offline
    assert os.environ["AGENTICML_OFFLINE_DIR"] == expected_offline
    mock.track_model.assert_called_once()
    assert type(mock.track_model.call_args.args[0]).__name__ == "Wav2Vec2ForCTC"
    mock.track_optimizer.assert_called_once()
    mock.track_loss.assert_called_once()
    assert mock.track_dataloader.call_count == 2
    dataloader_names = [call.kwargs["name"] for call in mock.track_dataloader.call_args_list]
    assert dataloader_names == ["train_dataset", "train"]
    mock.track_files.assert_called_once()
    tracked_files = mock.track_files.call_args.args
    run_dir = tmp_path / task / architecture
    assert str(run_dir / "resolved_config.json") in tracked_files
    assert str(run_dir / "model_config.json") in tracked_files
    assert str(run_dir / "batch_shapes.json") in tracked_files
    model_config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    assert model_config["model_type"] == "wav2vec2"
    batch_shapes = json.loads((run_dir / "batch_shapes.json").read_text(encoding="utf-8"))
    assert batch_shapes["batch_size"] == config["training"]["batch_size"]
    assert batch_shapes["input_values"][0] == config["training"]["batch_size"]
    assert all(isinstance(dim, int) for name, shape in batch_shapes.items() if name != "batch_size" for dim in shape)
    mock.log.assert_called()
    logged = mock.log.call_args
    assert logged.kwargs["step"] == 1
    assert "train/loss" in logged.args[0]
    mock.summary.update.assert_called_once_with(summary)
    artifact_types = [call.kwargs.get("type") for call in mock.log_artifact.call_args_list]
    artifact_paths = [str(call.args[0]) for call in mock.log_artifact.call_args_list]
    assert "model" in artifact_types
    assert any(path.endswith("checkpoint.pt") for path in artifact_paths)
    assert any(path.endswith("summary.json") for path in artifact_paths)
    mock.finish.assert_called_once_with(status="finished")


def test_disabled_tracking_skips_agenticml(tmp_path, monkeypatch):
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, _, _ = _smoke_config(tmp_path, save_checkpoint=False)
    config.setdefault("tracking", {})["enabled"] = False

    summary = train(config)

    assert summary["steps"] == 1
    mock.init.assert_not_called()
    mock.finish.assert_not_called()


def test_online_mode_omits_offline_dir(tmp_path, monkeypatch):
    mock = MagicMock()
    mock.Run = _RunWithOfflineDir
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, _, _ = _smoke_config(tmp_path, save_checkpoint=False)
    config.setdefault("tracking", {})["mode"] = "online"

    train(config)

    assert "offline_dir" not in mock.init.call_args.kwargs
    assert mock.init.call_args.kwargs["mode"] == "online"


def test_offline_dir_uses_env_when_sdk_rejects_kwarg(tmp_path, monkeypatch):
    mock = MagicMock()
    mock.Run = _RunWithoutOfflineDir
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, task, architecture = _smoke_config(tmp_path, save_checkpoint=False)
    expected_offline = str(tmp_path / task / architecture / "agenticml")

    train(config)

    assert os.environ["AGENTICML_OFFLINE_DIR"] == expected_offline
    assert "offline_dir" not in mock.init.call_args.kwargs
    assert "auto_track" not in mock.init.call_args.kwargs
    assert mock.init.call_args.kwargs["mode"] == "offline"
    mock.track_model.assert_called_once()
    mock.track_optimizer.assert_called_once()
    mock.track_loss.assert_called_once()
    assert mock.track_dataloader.call_count == 2
    mock.track_files.assert_called_once()


def test_auto_tracking_skips_explicit_component_calls(tmp_path, monkeypatch):
    mock = MagicMock()
    mock.Run = _RunWithOfflineDir
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, _, _ = _smoke_config(tmp_path, save_checkpoint=False)
    config.setdefault("tracking", {})["auto_tracking"] = True

    train(config)

    assert mock.init.call_args.kwargs["auto_track"] is True
    mock.track_model.assert_not_called()
    mock.track_optimizer.assert_not_called()
    mock.track_loss.assert_not_called()
    mock.track_dataloader.assert_not_called()
    mock.track_files.assert_called_once()


def test_auto_tracking_requires_sdk_support(tmp_path, monkeypatch):
    mock = MagicMock()
    mock.Run = _RunWithoutOfflineDir
    monkeypatch.setitem(sys.modules, "agenticml", mock)
    config, _, _ = _smoke_config(tmp_path, save_checkpoint=False)
    config.setdefault("tracking", {})["auto_tracking"] = True

    with pytest.raises(RuntimeError, match="auto_track"):
        train(config)

    mock.init.assert_not_called()


def test_missing_agenticml_raises_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "agenticml", None)
    with pytest.raises(ImportError, match="agenticml-py"):
        start_run(
            {
                "task": "asr",
                "architecture": "wav2vec2",
                "profile": "smoke",
                "tracking": {"enabled": True, "mode": "offline"},
            },
            tmp_path,
        )


def test_flatten_hyperparameters_skips_non_serializable_values():
    flattened = flatten_hyperparameters(
        {
            "task": "asr",
            "architecture": "wav2vec2",
            "profile": "smoke",
            "model": {"backend": "surrogate", "module": object()},
            "data": {"length": 8, "dataset_path": None},
            "loss": [{"name": "ctc", "weight": 1.0}],
            "optimizer": {"learning_rate": 0.001},
            "training": {"max_steps": 2},
            "source": {"hub_id": "facebook/wav2vec2-base"},
            "tracking": {"enabled": True},
        }
    )
    assert flattened["task"] == "asr"
    assert flattened["model.backend"] == "surrogate"
    assert "model.module" not in flattened
    assert flattened["data.dataset_path"] is None
    assert flattened["loss.0.name"] == "ctc"
    assert flattened["optimizer.learning_rate"] == 0.001
    assert "tracking.enabled" not in flattened
