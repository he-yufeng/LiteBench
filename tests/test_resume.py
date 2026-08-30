"""Tests for resumable runs: checkpoints let an interrupted benchmark continue."""

import asyncio
from datetime import datetime

import pytest

from litebench.core.models import Sample
from litebench.core.runner import Runner
from litebench.core.storage import Storage
from litebench.llm.client import ChatResult


class _MiniTask:
    name = "mini"
    description = "mini task"

    def build_prompt(self, sample):
        return f"{sample.input}"

    def system_prompt(self):
        return None

    def score(self, sample, text):
        return (1.0, text.strip() == str(sample.target))


class _CountingClient:
    """Answers every sample correctly, recording each prompt it was called with."""

    model = "fake-model"
    temperature = 0.0
    max_tokens = 32

    def __init__(self):
        self.calls: list[str] = []

    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        return ChatResult(text=f"{prompt}-OK", prompt_tokens=1, completion_tokens=1, latency_ms=1)


SAMPLES = [Sample(id=f"s{i}", input=f"s{i}", target=f"s{i}-OK") for i in range(4)]


def _seed_interrupted_run(storage, run_id):
    async def seed():
        await storage.init()
        await storage.start_checkpoint(
            run_id, "mini", "fake-model", {"temperature": 0.0}, datetime(2026, 8, 29, 10, 0, 0), 4
        )
        client = _CountingClient()
        runner = Runner(task=_MiniTask(), client=client, run_id=run_id)
        results = []
        for sample in SAMPLES[:2]:
            res = await runner._eval_sample(sample)
            await storage.save_checkpoint_result(run_id, res)
            results.append(res)
        return results

    return asyncio.run(seed())


def test_checkpoint_roundtrip(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    run_id = "run-1"
    _seed_interrupted_run(storage, run_id)

    async def check():
        state = await storage.find_interrupted_checkpoint("mini", "fake-model")
        assert state is not None
        assert state["run_id"] == run_id
        prior = await storage.list_checkpoint_results(run_id)
        assert [r.sample_id for r in prior] == ["s0", "s1"]
        await storage.complete_checkpoint(run_id)
        assert await storage.find_interrupted_checkpoint("mini", "fake-model") is None

    asyncio.run(check())


def test_find_interrupted_returns_none_without_a_checkpoint(tmp_path):
    storage = Storage(tmp_path / "runs.db")

    async def check():
        await storage.init()
        assert await storage.find_interrupted_checkpoint("mini", "fake-model") is None

    asyncio.run(check())


def test_resume_skips_completed_samples_and_merges(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    run_id = "run-1"
    prior = _seed_interrupted_run(storage, run_id)
    assert len(prior) == 2

    client = _CountingClient()
    started = datetime(2026, 8, 29, 10, 0, 0)
    runner = Runner(
        task=_MiniTask(),
        client=client,
        prior_results=prior,
        skip_ids={r.sample_id for r in prior},
        run_id=run_id,
        started_at=started,
    )
    summary, results = asyncio.run(runner.run(SAMPLES))

    # Only the unfinished samples were re-run; nothing was paid for twice.
    assert client.calls == ["s2", "s3"]
    assert summary.run_id == run_id
    assert summary.started_at == started
    assert summary.n_samples == 4
    assert summary.n_correct == 4
    assert [r.sample_id for r in results] == ["s0", "s1", "s2", "s3"]


def test_resume_aggregate_matches_a_single_full_run(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    run_id = "run-2"
    prior = _seed_interrupted_run(storage, run_id)

    resumed_client = _CountingClient()
    resumed = Runner(
        task=_MiniTask(),
        client=resumed_client,
        prior_results=prior,
        skip_ids={r.sample_id for r in prior},
        run_id=run_id,
    )
    resumed_summary, resumed_results = asyncio.run(resumed.run(SAMPLES))

    full = Runner(task=_MiniTask(), client=_CountingClient())
    full_summary, full_results = asyncio.run(full.run(SAMPLES))

    assert resumed_summary.n_correct == full_summary.n_correct
    assert resumed_summary.accuracy == full_summary.accuracy
    assert resumed_summary.total_prompt_tokens == full_summary.total_prompt_tokens
    assert [r.sample_id for r in resumed_results] == [r.sample_id for r in full_results]
