from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from litebench.config import DB_PATH, ensure_dirs
from litebench.core.storage import Storage
from litebench.tasks import get_task, list_tasks

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as e:  # pragma: no cover - only hits when extra isn't installed
    raise ImportError(
        "litebench[web] is required for the server. Install with: pip install 'litebench[web]'"
    ) from e

import aiosqlite

STATIC_DIR = Path(__file__).parent / "static"


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def create_app() -> FastAPI:
    ensure_dirs()
    storage = Storage(DB_PATH)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await storage.init()
        yield

    app = FastAPI(title="LiteBench", version="0.3.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/tasks")
    async def api_tasks() -> list[dict[str, str]]:
        out = []
        for name in list_tasks():
            task = get_task(name)
            out.append({"name": name, "description": task.description or ""})
        return out

    @app.get("/api/runs")
    async def api_runs(limit: int = 100, task: str | None = None) -> list[dict[str, Any]]:
        runs = await storage.list_runs(limit=limit, task=task)
        return [_run_to_api(r) for r in runs]

    @app.get("/api/runs/{run_prefix}")
    async def api_run_detail(run_prefix: str) -> dict[str, Any]:
        """Look up a run by an 8-char (or longer) prefix, same UX as the CLI."""
        runs = await storage.list_runs(limit=500)
        match = next((r for r in runs if r.run_id.startswith(run_prefix)), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_prefix}")

        async with aiosqlite.connect(storage.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT sample_id, input, target, prediction, score, correct,
                          latency_ms, prompt_tokens, completion_tokens, error, metadata
                   FROM samples WHERE run_id = ? ORDER BY sample_id""",
                (match.run_id,),
            )
            rows = await cursor.fetchall()
        samples = [_sample_row_to_api(r) for r in rows]
        return {"summary": _run_to_api(match), "samples": samples}

    @app.get("/api/compare")
    async def api_compare() -> dict[str, Any]:
        """Matrix view: accuracy grouped by (task, model) across all saved runs."""
        runs = await storage.list_runs(limit=500)
        tasks: set[str] = set()
        models: set[str] = set()
        cells: dict[tuple[str, str], dict[str, Any]] = {}
        for r in runs:
            tasks.add(r.task)
            models.add(r.model)
            key = (r.task, r.model)
            # Keep the most recent run per (task, model) pair.
            if key not in cells or r.finished_at > _parse_iso(cells[key]["finished_at"]):
                cells[key] = {
                    "task": r.task,
                    "model": r.model,
                    "accuracy": r.accuracy,
                    "n_samples": r.n_samples,
                    "run_id": r.run_id,
                    "finished_at": r.finished_at.isoformat(),
                }
        return {
            "tasks": sorted(tasks),
            "models": sorted(models),
            "cells": list(cells.values()),
        }

    return app


def _run_to_api(r: Any) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "task": r.task,
        "model": r.model,
        "n_samples": r.n_samples,
        "n_correct": r.n_correct,
        "accuracy": r.accuracy,
        "mean_latency_ms": r.mean_latency_ms,
        "total_prompt_tokens": r.total_prompt_tokens,
        "total_completion_tokens": r.total_completion_tokens,
        "started_at": r.started_at.isoformat(),
        "finished_at": r.finished_at.isoformat(),
        "config": r.config,
    }


def _sample_row_to_api(row: aiosqlite.Row) -> dict[str, Any]:
    meta = json.loads(row["metadata"])
    return {
        "sample_id": row["sample_id"],
        "input": row["input"],
        "target": row["target"],
        "prediction": row["prediction"],
        "score": row["score"],
        "correct": bool(row["correct"]),
        "latency_ms": row["latency_ms"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "error": row["error"],
        "tool_calls": meta.get("tool_calls") if isinstance(meta, dict) else None,
        "stop_reason": meta.get("stop_reason") if isinstance(meta, dict) else None,
    }
