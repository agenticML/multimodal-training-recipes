from unittest.mock import patch

import pytest

from training_recipes.cli import apply_cli_overrides, build_parser, main_offline


def test_offline_entrypoint_requires_agenticml_asset_dir():
    with pytest.raises(ValueError, match="--agenticml-asset-dir is required"):
        main_offline(["--task", "asr", "--architecture", "wav2vec2"])


def test_offline_entrypoint_sets_offline_mode_and_asset_dir(tmp_path):
    parser = build_parser(offline_tracking=True)
    args = parser.parse_args(
        [
            "--task",
            "asr",
            "--architecture",
            "wav2vec2",
            "--tracker-mode",
            "offline",
            "--agenticml-asset-dir",
            str(tmp_path / "assets"),
        ]
    )
    assert args.tracker_mode == "offline"
    assert args.agenticml_asset_dir == str(tmp_path / "assets")
    config = {
        "task": "asr",
        "architecture": "wav2vec2",
        "profile": "smoke",
        "data": {},
        "loss": [{"name": "ctc", "weight": 1.0}],
        "optimizer": {"name": "adamw"},
        "training": {},
    }
    apply_cli_overrides(args, config)
    assert config["tracking"]["mode"] == "offline"
    assert config["tracking"]["offline_dir"] == str(tmp_path / "assets")
    assert config["tracking"]["group"] == "asr"


def test_offline_entrypoint_rejects_online_mode():
    parser = build_parser(offline_tracking=True)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--task",
                "asr",
                "--architecture",
                "wav2vec2",
                "--tracker-mode",
                "online",
                "--agenticml-asset-dir",
                "/tmp/assets",
            ]
        )


def test_main_cli_accepts_agenticml_asset_dir_alias(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task",
            "asr",
            "--architecture",
            "wav2vec2",
            "--agenticml-asset-dir",
            str(tmp_path / "assets"),
        ]
    )
    config = {
        "task": "asr",
        "data": {},
        "loss": [{"name": "ctc"}],
        "optimizer": {},
        "training": {},
    }
    apply_cli_overrides(args, config)
    assert config["tracking"]["offline_dir"] == str(tmp_path / "assets")


def test_main_offline_invokes_train(tmp_path):
    with patch("training_recipes.cli.train", return_value={"steps": 1}) as train:
        main_offline(
            [
                "--task",
                "asr",
                "--architecture",
                "wav2vec2",
                "--profile",
                "smoke",
                "--max-steps",
                "1",
                "--output-dir",
                str(tmp_path),
                "--tracker-mode",
                "offline",
                "--agenticml-asset-dir",
                str(tmp_path / "assets"),
            ]
        )
    train.assert_called_once()
    config = train.call_args.args[0]
    assert config["tracking"]["mode"] == "offline"
    assert config["tracking"]["offline_dir"] == str(tmp_path / "assets")
