"""FastAPI app for the localhost smoke-run dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from training_recipes.registry import architecture_catalog, task_names

from .runner import (
    AssetPathError,
    RunManager,
    collect_assets,
    group_assets,
    read_asset_file,
    serialize_architecture,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunCreate(BaseModel):
    task: str = Field(min_length=1)
    architecture: str = Field(min_length=1)


def _architecture_or_404(task: str, architecture: str) -> dict:
    try:
        catalog = architecture_catalog(task)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    spec = catalog.get(architecture)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown architecture {architecture!r} for {task}",
        )
    return spec


def create_app(*, asset_dir: str | Path, host: str = "127.0.0.1") -> FastAPI:
    asset_root = Path(asset_dir).expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    runner = RunManager(asset_root)
    app = FastAPI(title="Smoke dashboard", docs_url=None, redoc_url=None)
    app.state.runner = runner
    app.state.asset_dir = asset_root
    app.state.host = host

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "profile": "smoke",
            "tracker_mode": "offline",
            "asset_dir": str(app.state.asset_dir),
            "host": app.state.host,
        }

    @app.get("/api/tasks")
    def list_tasks() -> dict:
        return {"tasks": task_names()}

    @app.get("/api/tasks/{task}")
    def list_architectures(task: str) -> dict:
        try:
            catalog = architecture_catalog(task)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        architectures = [
            serialize_architecture(task, name, spec) for name, spec in catalog.items()
        ]
        return {"task": task, "architectures": architectures}

    @app.post("/api/runs", status_code=201)
    def start_run(body: RunCreate) -> dict:
        spec = _architecture_or_404(body.task, body.architecture)
        payload = serialize_architecture(body.task, body.architecture, spec)
        if not payload["dummy_supported"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{body.task}/{body.architecture} is not dummy-supported "
                    "(custom/mlx recipes are skipped)"
                ),
            )
        try:
            run = runner.start(body.task, body.architecture)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return run.snapshot()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = runner.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return run.snapshot()

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        run = runner.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return StreamingResponse(
            runner.iter_events(run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/runs/{run_id}/assets")
    def run_assets(run_id: str) -> dict:
        run = runner.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        offline_dir, files = collect_assets(run.run_dir)
        return {
            "run_id": run.id,
            "status": run.status,
            "offline_dir": None if offline_dir is None else str(offline_dir),
            "groups": group_assets(files),
        }

    @app.get("/api/runs/{run_id}/assets/file")
    def run_asset_file(run_id: str, path: str = Query(...)) -> dict:
        run = runner.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        try:
            return read_asset_file(run.run_dir, path)
        except AssetPathError as exc:
            message = str(exc)
            status = 404 if "not found" in message else 400
            raise HTTPException(status_code=status, detail=message) from exc

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
