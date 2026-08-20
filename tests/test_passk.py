"""Tests for pass@k: the estimator and the runner's repeat plumbing."""

import asyncio
import re

import pytest

from litebench.core.models import Sample
from litebench.core.passk import mean_pass_at_k, pass_at_k
from litebench.core.runner import Runner
from litebench.llm.client import ChatResult


def test_pass_at_k_estimator_known_values():
    # n=10, c=5, k=3 -> 1 - C(5,3)/C(10,3) = 1 - 10/120
    assert pass_at_k(10, 5, 3) == pytest.approx(0.9166667, abs=1e-4)
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6, abs=1e-6)
    assert pass_at_k(4, 4, 4) == 1.0
    assert pass_at_k(4, 0, 2) == 0.0
    assert pass_at_k(5, 3, 1) == pytest.approx(0.6)


def test_pass_at_k_rejects_bad_bounds():
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 5)  # k > n
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_mean_pass_at_k():
    assert mean_pass_at_k([4, 2], 4, 1) == pytest.approx(0.75)


class _MiniTask:
    name = "mini"
    description = "mini task"

    def build_prompt(self, sample):
        return f"{sample.input} (repeat {sample.id})"

    def system_prompt(self):
        return None

    def score(self, sample, text):
        return (1.0, text.strip() == str(sample.target))


class _FakeClient:
    """Answers correctly on the "a" sample always; on "b" only for the first
    half of its repeats, reading the repeat index the task embeds in the prompt."""

    model = "fake-model"
    temperature = 0.0
    max_tokens = 32

    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        repeat = int(re.search(r"#(\d+)", prompt).group(1))
        text = "A-OK" if prompt.startswith("a") else ("B-OK" if repeat < 2 else "B-NO")
        return ChatResult(text=text, prompt_tokens=1, completion_tokens=1, latency_ms=1)


def _run(samples, repeat, pass_k=None):
    runner = Runner(task=_MiniTask(), client=_FakeClient(), samples_per_task=repeat, pass_k=pass_k)
    return asyncio.run(runner.run(samples))


def test_runner_repeat_reports_pass_at_k():
    samples = [
        Sample(id="a", input="a in", target="A-OK"),
        Sample(id="b", input="b in", target="B-OK"),
    ]
    summary, results = _run(samples, repeat=4)

    assert summary.n_samples == 8
    assert summary.n_correct == 6  # a: 4/4, b: 2/4
    assert summary.samples_per_task == 4
    assert summary.pass_at_1 == pytest.approx(0.75)
    # k defaults to n: "at least one of all four attempts correct".
    assert summary.pass_k == 4
    assert summary.pass_at_k == pytest.approx(1.0)
    # repeats stay distinguishable in history
    ids = sorted(r.sample_id for r in results)
    assert ids == ["a#0", "a#1", "a#2", "a#3", "b#0", "b#1", "b#2", "b#3"]


def test_runner_repeat_with_explicit_k():
    samples = [
        Sample(id="a", input="a in", target="A-OK"),
        Sample(id="b", input="b in", target="B-OK"),
    ]
    summary, _ = _run(samples, repeat=4, pass_k=2)
    # a: pass@2 = 1; b: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert summary.pass_k == 2
    assert summary.pass_at_k == pytest.approx((1 + (1 - 1 / 6)) / 2, abs=1e-6)


def test_runner_single_run_keeps_pass_fields_empty():
    samples = [Sample(id="a", input="a in", target="A-OK")]
    summary, results = _run(samples, repeat=1)
    assert summary.pass_k is None
    assert summary.pass_at_k is None
    assert results[0].sample_id == "a"
