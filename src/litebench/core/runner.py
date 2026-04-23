from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime

from litebench.core.models import RunSummary, Sample, SampleResult
from litebench.llm.client import LLMClient
from litebench.tasks.base import Task


class Runner:
    def __init__(
        self,
        task: Task,
        client: LLMClient,
        concurrency: int = 8,
        on_progress=None,
    ):
        self.task = task
        self.client = client
        self.concurrency = concurrency
        self.on_progress = on_progress

    async def run(self, samples: list[Sample]) -> tuple[RunSummary, list[SampleResult]]:
        started_at = datetime.now()
        sem = asyncio.Semaphore(self.concurrency)
        results: list[SampleResult] = []
        done = 0
        total = len(samples)
        lock = asyncio.Lock()

        async def run_one(sample: Sample) -> SampleResult:
            async with sem:
                return await self._eval_sample(sample)

        async def track(sample: Sample) -> SampleResult:
            nonlocal done
            res = await run_one(sample)
            async with lock:
                done += 1
                if self.on_progress:
                    self.on_progress(done, total, res)
            return res

        tasks = [asyncio.create_task(track(s)) for s in samples]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

        finished_at = datetime.now()
        results.sort(key=lambda r: r.sample_id)

        n_correct = sum(1 for r in results if r.correct)
        accuracy = n_correct / total if total else 0.0
        mean_latency = sum(r.latency_ms for r in results) / total if total else 0.0
        total_prompt = sum(r.prompt_tokens for r in results)
        total_completion = sum(r.completion_tokens for r in results)

        summary = RunSummary(
            run_id=str(uuid.uuid4()),
            task=self.task.name,
            model=self.client.model,
            n_samples=total,
            n_correct=n_correct,
            accuracy=accuracy,
            mean_latency_ms=mean_latency,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            started_at=started_at,
            finished_at=finished_at,
            config={
                "temperature": self.client.temperature,
                "max_tokens": self.client.max_tokens,
                "concurrency": self.concurrency,
            },
        )
        return summary, results

    async def _eval_sample(self, sample: Sample) -> SampleResult:
        prompt = self.task.build_prompt(sample)
        messages: list[dict[str, str]] = []
        system = self.task.system_prompt()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            out = await self.client.chat(messages)
            score_result = self.task.score(sample, out.text)
            if inspect.iscoroutine(score_result):
                score_result = await score_result
            score, correct = score_result
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                target=sample.target,
                prediction=out.text,
                score=score,
                correct=correct,
                latency_ms=out.latency_ms,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                metadata=sample.metadata,
            )
        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                target=sample.target,
                prediction="",
                score=0.0,
                correct=False,
                latency_ms=0,
                error=f"{type(e).__name__}: {e}"[:500],
                metadata=sample.metadata,
            )
