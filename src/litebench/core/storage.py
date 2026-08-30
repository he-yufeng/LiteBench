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

CREATE TABLE IF NOT EXISTS checkpoint_runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT NOT NULL,
    started_at TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS checkpoint_results (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id),
    FOREIGN KEY (run_id) REFERENCES checkpoint_runs(run_id)
);
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
            rows = []
            for r in results:
                # Fold agent-trace fields into metadata so the schema doesn't
                # need to grow dedicated columns; the server/UI reads them back
                # out of the parsed metadata dict.
                meta: dict = dict(r.metadata) if r.metadata else {}
                if r.tool_calls is not None:
                    meta["tool_calls"] = r.tool_calls
                if r.steps:
                    meta["steps"] = r.steps
                rows.append((
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
                    json.dumps(meta),
                ))
            await db.executemany(
                """INSERT OR REPLACE INTO samples
                (run_id, sample_id, input, target, prediction, score, correct,
                 latency_ms, prompt_tokens, completion_tokens, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
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

    async def start_checkpoint(
        self,
        run_id: str,
        task: str,
        model: str,
        config: dict,
        started_at: datetime,
        total_samples: int,
    ) -> None:
        """Register a resumable run before its first sample goes out."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO checkpoint_runs
                (run_id, task, model, config, started_at, total_samples, status)
                VALUES (?, ?, ?, ?, ?, ?, 'running')""",
                (run_id, task, model, json.dumps(config), started_at.isoformat(), total_samples),
            )
            await db.commit()

    async def save_checkpoint_result(self, run_id: str, result: SampleResult) -> None:
        """Persist one finished sample immediately, so an interruption only
        ever costs the in-flight samples."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO checkpoint_results (run_id, sample_id, result, created_at)
                VALUES (?, ?, ?, ?)""",
                (
                    run_id,
                    result.sample_id,
                    result.model_dump_json(),
                    datetime.now().isoformat(),
                ),
            )
            await db.commit()

    async def list_checkpoint_results(self, run_id: str) -> list[SampleResult]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT result FROM checkpoint_results WHERE run_id = ? ORDER BY sample_id",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [SampleResult.model_validate_json(r[0]) for r in rows]

    async def find_interrupted_checkpoint(self, task: str, model: str) -> dict | None:
        """Newest unfinished checkpoint for this task+model, if one exists."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT run_id, config, started_at, total_samples FROM checkpoint_runs
                WHERE task = ? AND model = ? AND status = 'running'
                ORDER BY started_at DESC LIMIT 1""",
                (task, model),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "config": json.loads(row["config"]),
            "started_at": datetime.fromisoformat(row["started_at"]),
            "total_samples": row["total_samples"],
        }

    async def get_checkpoint(self, run_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT run_id, config, started_at, total_samples, status
                FROM checkpoint_runs WHERE run_id = ?""",
                (run_id,),
            )
            row = await cursor.fetchone()
        if row is None or row["status"] != "running":
            return None
        return {
            "run_id": row["run_id"],
            "config": json.loads(row["config"]),
            "started_at": datetime.fromisoformat(row["started_at"]),
            "total_samples": row["total_samples"],
        }

    async def complete_checkpoint(self, run_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE checkpoint_runs SET status = 'completed' WHERE run_id = ?",
                (run_id,),
            )
            await db.commit()

    async def list_samples(self, run_id: str) -> list[SampleResult]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT sample_id, input, target, prediction, score, correct,
                          latency_ms, prompt_tokens, completion_tokens, error, metadata
                   FROM samples WHERE run_id = ? ORDER BY sample_id""",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [self._row_to_sample(r) for r in rows]

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

    @staticmethod
    def _row_to_sample(row: aiosqlite.Row) -> SampleResult:
        meta = json.loads(row["metadata"])
        if not isinstance(meta, dict):
            meta = {}
        target = _decode_target(row["target"])
        return SampleResult(
            sample_id=row["sample_id"],
            input=row["input"],
            target=target,
            prediction=row["prediction"],
            score=row["score"],
            correct=bool(row["correct"]),
            latency_ms=row["latency_ms"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            error=row["error"],
            tool_calls=meta.get("tool_calls"),
            steps=int(meta.get("steps") or 0),
            metadata={k: v for k, v in meta.items() if k not in {"tool_calls", "steps"}},
        )


def _decode_target(raw: str) -> str | list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return value if isinstance(value, list) else raw
