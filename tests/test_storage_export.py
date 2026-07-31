from datetime import datetime

import pytest

from litebench.cli import _export_payload
from litebench.core.models import RunSummary, SampleResult
from litebench.core.storage import Storage


@pytest.mark.asyncio
async def test_storage_lists_samples_for_export(tmp_path):
    storage = Storage(tmp_path / "runs.sqlite")
    await storage.init()
    summary = RunSummary(
        run_id="abcdef123456",
        task="gsm8k",
        model="gpt-5",
        system_prompt="You are a helpful assistant.",
        n_samples=1,
        n_correct=1,
        accuracy=1.0,
        mean_latency_ms=123.0,
        total_prompt_tokens=10,
        total_completion_tokens=5,
        started_at=datetime(2026, 5, 12, 1, 0, 0),
        finished_at=datetime(2026, 5, 12, 1, 0, 1),
        config={"temperature": 0.0},
    )
    result = SampleResult(
        sample_id="gsm8k-1",
        input="1+1?",
        target=["2", "two"],
        prediction="2",
        score=1.0,
        correct=True,
        latency_ms=123,
        prompt_tokens=10,
        completion_tokens=5,
        tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}, "result": "2"}],
        steps=1,
        metadata={"stop_reason": "tool"},
    )

    await storage.save_run(summary, [result])

    samples = await storage.list_samples("abcdef123456")
    assert len(samples) == 1
    assert samples[0].target == ["2", "two"]
    assert samples[0].tool_calls == result.tool_calls
    assert samples[0].steps == 1
    assert samples[0].metadata == {"stop_reason": "tool"}


def test_export_payload_shape():
    summary = RunSummary(
        run_id="run-1",
        task="custom",
        model="deepseek",
        system_prompt="You are a code assistant.",
        n_samples=0,
        n_correct=0,
        accuracy=0.0,
        mean_latency_ms=0.0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        started_at=datetime(2026, 5, 12, 1, 0, 0),
        finished_at=datetime(2026, 5, 12, 1, 0, 0),
        config={},
    )

    payload = _export_payload(summary, [])

    assert payload["summary"]["run_id"] == "run-1"
    assert payload["results"] == []
