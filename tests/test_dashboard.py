import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from training_recipes.dashboard.__main__ import build_parser
from training_recipes.dashboard.app import create_app
from training_recipes.dashboard.runner import RunState
from training_recipes.registry import task_names

REPO_ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = REPO_ROOT / "scripts" / "start_dashboard.sh"


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, host="127.0.0.1"))


def test_meta_and_tasks(tmp_path):
    client = _client(tmp_path)
    meta = client.get("/api/meta").json()
    assert meta["profile"] == "smoke"
    assert meta["host"] == "127.0.0.1"
    assert Path(meta["output_dir"]) == tmp_path.resolve()
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


def test_rejects_concurrent_run(tmp_path):
    app = create_app(output_dir=tmp_path, host="127.0.0.1")
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


def test_module_parses_port_and_output_dir(tmp_path):
    args = build_parser().parse_args(
        ["--host", "127.0.0.1", "--port", "8080", "--output-dir", str(tmp_path)]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.output_dir == str(tmp_path)


def test_start_dashboard_script_exists():
    assert START_SCRIPT.is_file()
    text = START_SCRIPT.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "VIRTUAL_ENV" in text
    assert "-m training_recipes.dashboard" in text
    assert os.access(START_SCRIPT, os.X_OK)


def test_smoke_run_streams_logs_and_writes_outputs(tmp_path):
    client = _client(tmp_path)
    response = client.post("/api/runs", json={"task": "asr", "architecture": "wav2vec2"})
    assert response.status_code == 201
    run = response.json()
    events = client.get(f"/api/runs/{run['id']}/events")
    assert events.status_code == 200
    assert "event: log" in events.text
    assert "step=1 loss=" in events.text
    completion = events.text.split("event: complete\ndata: ", 1)[1].split("\n", 1)[0]
    payload = json.loads(completion)
    assert payload["status"] == "finished", events.text
    assert payload["summary"]["steps"] == 2
    assert payload["summary"]["model_class"] == "Wav2Vec2ForCTC"
    finished = client.get(f"/api/runs/{run['id']}").json()
    assert finished["status"] == "finished"
    assert "step=1 loss=" in finished["logs"]
    output = Path(run["run_dir"]) / "asr" / "wav2vec2"
    assert output.is_relative_to(tmp_path)
    assert json.loads((output / "summary.json").read_text()) == payload["summary"]
    for name in ("resolved_config.json", "model_config.json", "batch_shapes.json",
                 "metrics.jsonl", "checkpoint.pt"):
        assert (output / name).is_file()
