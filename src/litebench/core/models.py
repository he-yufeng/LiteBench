from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """One item from a benchmark dataset."""

    id: str
    input: str
    target: str | list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    """Outcome of running a single sample through the model."""

    sample_id: str
    input: str
    target: str | list[str]
    prediction: str
    score: float
    correct: bool
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    # Populated only for AgentTask rollouts. Each entry records one tool call's
    # {name, arguments, result, error}. None for single-turn tasks.
    tool_calls: list[dict[str, Any]] | None = None
    steps: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Aggregated results for one benchmark run."""

    run_id: str
    task: str
    model: str
    system_prompt: str | None
    n_samples: int
    n_correct: int
    accuracy: float
    mean_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    started_at: datetime
    finished_at: datetime
    config: dict[str, Any] = Field(default_factory=dict)
