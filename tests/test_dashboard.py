import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from training_recipes.dashboard.__main__ import build_parser
from training_recipes.dashboard.app import create_app
from training_recipes.dashboard.runner import (
    AssetPathError,
    RunState,
    collect_assets,
    group_assets,
    parse_journal_assets,
    read_asset_file,
)
from training_recipes.registry import task_names

REPO_ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = REPO_ROOT / "scripts" / "start_dashboard.sh"


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(asset_dir=tmp_path, host="127.0.0.1"))


def _write_journal_fixture(run_dir: Path) -> Path:
    offline = run_dir / "offline-testfixture"
    (offline / "code" / "model").mkdir(parents=True)
    (offline / "code" / "loss").mkdir()
    (offline / "artifacts").mkdir()
    (offline / "code" / "model" / "models.py").write_text("class DummyModel:\n    pass\n")
    (offline / "code" / "loss" / "losses.py").write_text("def dummy_loss():\n    return 0\n")
    (offline / "artifacts" / "checkpoint.pt").write_bytes(b"\x00\x01\x02")
    journal = [
        {
            "kind": "upload_code",
            "payload": {
                "manifest": {
                    "files": [
                        {
                            "path": "model/models.py",
                            "content_type": "text/x-python",
                            "source": "model",
                            "size": 28,
                        },
                        {
                            "path": "loss/losses.py",
                            "content_type": "text/x-python",
                            "source": "loss",
                            "size": 32,
                        },
                        {
                            "path": "artifacts/checkpoint.pt",
                            "content_type": "application/octet-stream",
                            "source": "runtime_read",
                            "size": 3,
                        },
                    ]
                },
                "files": [
                    {
                        "path": "model/models.py",
                        "content_type": "text/x-python",
                        "stored_at": "code/model/models.py",
                    },
                    {
                        "path": "loss/losses.py",
                        "content_type": "text/x-python",
                        "stored_at": "code/loss/losses.py",
                    },
                    {
                        "path": "artifacts/checkpoint.pt",
                        "content_type": "application/octet-stream",
                        "stored_at": "artifacts/checkpoint.pt",
                    },
                ],
            },
        },
        {
            "kind": "finalize",
            "payload": {
                "summary": {
                    "steps": 1,
                    "final_loss": 0.5,
                    "model_class": "Wav2Vec2ForCTC",
                }
            },
        },
    ]
    (offline / "journal.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in journal),
        encoding="utf-8",
    )
    return offline


def test_meta_and_tasks(tmp_path):
    client = _client(tmp_path)
    meta = client.get("/api/meta").json()
    assert meta["profile"] == "smoke"
    assert meta["tracker_mode"] == "offline"
    assert meta["host"] == "127.0.0.1"
    assert Path(meta["asset_dir"]) == tmp_path.resolve()
    tasks = client.get("/api/tasks").json()["tasks"]
    assert tasks == task_names()
    assert len(tasks) == 16
    assert "asr" in tasks


def test_wav2vec2_is_dummy_supported(tmp_path):
    client = _client(tmp_path)
    payload = client.get("/api/tasks/asr").json()
    wav2vec2 = next(item for item in payload["architectures"] if item["name"] == "wav2vec2")
    assert wav2vec2["dummy_supported"] is True
    assert wav2vec2["class"] == "Wav2Vec2ForCTC"
    assert wav2vec2["family"]
    assert wav2vec2["hub_id"]
    assert "ctc" in wav2vec2["losses"]


def test_audiosr_is_not_dummy_supported(tmp_path):
    client = _client(tmp_path)
    payload = client.get("/api/tasks/audio_super_resolution").json()
    audiosr = next(item for item in payload["architectures"] if item["name"] == "audiosr")
    assert audiosr["dummy_supported"] is False
    response = client.post(
        "/api/runs",
        json={"task": "audio_super_resolution", "architecture": "audiosr"},
    )
    assert response.status_code == 400


def test_journal_asset_parsing(tmp_path):
    run_dir = tmp_path / "run"
    offline = _write_journal_fixture(run_dir)
    files = parse_journal_assets(offline / "journal.jsonl")
    by_path = {item.path: item for item in files}
    assert by_path["code/model/models.py"].source == "model"
    assert by_path["code/loss/losses.py"].source == "loss"
    discovered, collected = collect_assets(run_dir)
    assert discovered == offline
    groups = group_assets(collected)
    assert "model" in groups
    assert "loss" in groups


def test_code_walk_fallback_without_journal(tmp_path):
    run_dir = tmp_path / "run"
    offline = run_dir / "offline-fallback"
    target = offline / "code" / "model"
    target.mkdir(parents=True)
    (target / "models.py").write_text("print(1)\n")
    discovered, files = collect_assets(run_dir)
    assert discovered == offline
    assert files[0].path == "code/model/models.py"
    assert files[0].source == "model"


def test_asset_file_and_path_traversal(tmp_path):
    app = create_app(asset_dir=tmp_path, host="127.0.0.1")
    run_dir = tmp_path / "run-a"
    _write_journal_fixture(run_dir)
    secret = tmp_path / "secret.txt"
    secret.write_text("should-not-read")
    run = app.state.runner.register_finished(
        task="asr",
        architecture="wav2vec2",
        run_dir=run_dir,
        summary={"steps": 1, "final_loss": 0.5, "model_class": "Wav2Vec2ForCTC"},
    )
    client = TestClient(app)
    assets = client.get(f"/api/runs/{run.id}/assets").json()
    assert "model" in assets["groups"]
    text = client.get(
        f"/api/runs/{run.id}/assets/file",
        params={"path": "code/model/models.py"},
    )
    assert text.status_code == 200
    assert "DummyModel" in text.json()["content"]
    binary = client.get(
        f"/api/runs/{run.id}/assets/file",
        params={"path": "artifacts/checkpoint.pt"},
    )
    assert binary.status_code == 200
    assert binary.json()["binary"] is True
    assert binary.json()["content"] is None
    escaped = client.get(
        f"/api/runs/{run.id}/assets/file",
        params={"path": "../secret.txt"},
    )
    assert escaped.status_code == 400
    with pytest.raises(AssetPathError):
        read_asset_file(run_dir, "../secret.txt")
    absolute = client.get(
        f"/api/runs/{run.id}/assets/file",
        params={"path": str(secret)},
    )
    assert absolute.status_code == 400


def test_rejects_concurrent_run(tmp_path):
    app = create_app(asset_dir=tmp_path, host="127.0.0.1")
    runner = app.state.runner
    busy = RunState(
        id="busy",
        task="asr",
        architecture="wav2vec2",
        run_dir=tmp_path / "busy",
        status="running",
    )
    runner._runs[busy.id] = busy
    runner._active_id = busy.id
    client = TestClient(app)
    response = client.post("/api/runs", json={"task": "asr", "architecture": "wav2vec2"})
    assert response.status_code == 409


def test_unknown_task_is_404(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/tasks/not-a-task").status_code == 404
    assert client.get("/api/runs/missing").status_code == 404


def test_module_parses_port_and_asset_dir(tmp_path):
    args = build_parser().parse_args(
        ["--host", "127.0.0.1", "--port", "8080", "--asset-dir", str(tmp_path)]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.asset_dir == str(tmp_path)


def test_start_dashboard_script_exists():
    assert START_SCRIPT.is_file()
    text = START_SCRIPT.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "VIRTUAL_ENV" in text
    assert "-m training_recipes.dashboard" in text
    assert "--asset-dir" in text
    assert os.access(START_SCRIPT, os.X_OK)
