from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import aiosqlite

from litebench.core.models import RunSummary, SampleResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    n_correct INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    mean_latency_ms REAL NOT NULL,
    total_prompt_tokens INTEGER NOT NULL,
    total_completion_tokens INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    config TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    input TEXT NOT NULL,
    target TEXT NOT NULL,
    prediction TEXT NOT NULL,
    score REAL NOT NULL,
    correct INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    error TEXT,
    metadata TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_task_model ON runs(task, model);
CREATE INDEX IF NOT EXISTS idx_runs_finished_at ON runs(finished_at);
"""


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def save_run(self, summary: RunSummary, results: list[SampleResult]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO runs
                (run_id, task, model, n_samples, n_correct, accuracy, mean_latency_ms,
                 total_prompt_tokens, total_completion_tokens, started_at, finished_at, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.run_id,
                    summary.task,
                    summary.model,
                    summary.n_samples,
                    summary.n_correct,
                    summary.accuracy,
                    summary.mean_latency_ms,
                    summary.total_prompt_tokens,
                    summary.total_completion_tokens,
                    summary.started_at.isoformat(),
                    summary.finished_at.isoformat(),
                    json.dumps(summary.config),
                ),
            )
            await db.executemany(
                """INSERT OR REPLACE INTO samples
                (run_id, sample_id, input, target, prediction, score, correct,
                 latency_ms, prompt_tokens, completion_tokens, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        summary.run_id,
                        r.sample_id,
                        r.input,
                        json.dumps(r.target) if isinstance(r.target, list) else r.target,
                        r.prediction,
                        r.score,
                        1 if r.correct else 0,
                        r.latency_ms,
                        r.prompt_tokens,
                        r.completion_tokens,
                        r.error,
                        json.dumps(r.metadata),
                    )
                    for r in results
                ],
            )
            await db.commit()

    async def list_runs(self, limit: int = 20, task: str | None = None) -> list[RunSummary]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if task:
                cursor = await db.execute(
                    "SELECT * FROM runs WHERE task = ? ORDER BY finished_at DESC LIMIT ?",
                    (task, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM runs ORDER BY finished_at DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        return [self._row_to_summary(r) for r in rows]

    async def get_run(self, run_id: str) -> RunSummary | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = await cursor.fetchone()
        return self._row_to_summary(row) if row else None

    @staticmethod
    def _row_to_summary(row: aiosqlite.Row) -> RunSummary:
        return RunSummary(
            run_id=row["run_id"],
            task=row["task"],
            model=row["model"],
            n_samples=row["n_samples"],
            n_correct=row["n_correct"],
            accuracy=row["accuracy"],
            mean_latency_ms=row["mean_latency_ms"],
            total_prompt_tokens=row["total_prompt_tokens"],
            total_completion_tokens=row["total_completion_tokens"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]),
            config=json.loads(row["config"]),
        )
