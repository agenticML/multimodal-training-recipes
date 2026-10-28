import os
from unittest.mock import patch

from training_recipes.cli import apply_cli_overrides, build_parser, main


def test_main_invokes_train(tmp_path):
    with patch("training_recipes.cli.train", return_value={"steps": 1}) as train:
        main(
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
            ]
        )
    train.assert_called_once()
    config = train.call_args.args[0]
    assert config["task"] == "asr"
    assert config["architecture"] == "wav2vec2"
    assert config["profile"] == "smoke"
    assert config["training"]["max_steps"] == 1
    assert config["output_dir"] == str(tmp_path)


def test_cli_overrides_training_fields(tmp_path):
    args = build_parser().parse_args(
        [
            "--task", "asr", "--architecture", "wav2vec2",
            "--dataset-path", "/data/asr.jsonl", "--loss", "ctc",
            "--optimizer", "adamw", "--learning-rate", "0.001",
            "--batch-size", "4", "--max-steps", "2", "--device", "cpu",
            "--precision", "fp32", "--resume-from", "/data/checkpoint.pt",
            "--output-dir", str(tmp_path),
        ]
    )
    config = {"data": {}, "optimizer": {}, "training": {}, "_compatible_losses": ["ctc"]}
    apply_cli_overrides(args, config)
    assert config == {
        "data": {"dataset_path": "/data/asr.jsonl"},
        "loss": [{"name": "ctc", "weight": 1.0}],
        "optimizer": {"name": "adamw", "learning_rate": 0.001},
        "training": {
            "batch_size": 4,
            "max_steps": 2,
            "device": "cpu",
            "precision": "fp32",
            "resume_from": "/data/checkpoint.pt",
        },
        "output_dir": str(tmp_path),
    }


def test_offline_flag_disables_hub_downloads(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with patch("training_recipes.cli.train", return_value={"steps": 1}):
        main(["--task", "asr", "--architecture", "wav2vec2", "--offline"])
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
