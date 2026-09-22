from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run-online.sh"


def test_run_online_requires_api_key():
    env = os.environ.copy()
    env.pop("AGENTICML_API_KEY", None)
    result = subprocess.run(
        [str(RUN_SCRIPT), "--task", "asr", "--architecture", "wav2vec2"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "AGENTICML_API_KEY is not set" in result.stderr


def test_run_online_exports_default_host_and_forces_online_mode(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf 'host=%s\\n' \"$AGENTICML_HOST\"\n"
        "printf 'mode=%s\\n' \"$AGENTICML_MODE\"\n"
        "printf 'args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({"AGENTICML_API_KEY": "test-key", "PYTHON": str(fake_python)})

    result = subprocess.run(
        [
            str(RUN_SCRIPT),
            "--task",
            "asr",
            "--architecture",
            "wav2vec2",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "host=https://api.agenticml.xyz" in result.stdout
    assert "mode=online" in result.stdout
    assert (
        "args=-m training_recipes.cli --tracker-mode online --task asr "
        "--architecture wav2vec2"
    ) in result.stdout


def test_run_online_accepts_host_override(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf 'host=%s\\n' \"$AGENTICML_HOST\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({"AGENTICML_API_KEY": "test-key", "PYTHON": str(fake_python)})

    result = subprocess.run(
        [
            str(RUN_SCRIPT),
            "--agenticml-host",
            "https://agenticml.example.com",
            "--task",
            "asr",
            "--architecture",
            "wav2vec2",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "host=https://agenticml.example.com" in result.stdout
