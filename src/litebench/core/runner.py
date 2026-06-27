from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from datetime import datetime
from typing import Any

from litebench.agent.base import AgentTask, AgentTrace, Tool, ToolCall
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
        system_prompt_override: str | None = None,
    ):
        self.task = task
        self.client = client
        self.concurrency = concurrency
        self.on_progress = on_progress
        self.system_prompt_override = system_prompt_override

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
            system_prompt=self._get_system_prompt(),
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
        if isinstance(self.task, AgentTask):
            return await self._eval_agent_sample(sample, self.task)

        prompt = self.task.build_prompt(sample)
        messages: list[dict[str, Any]] = []
        system = self._get_system_prompt()
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

    async def _eval_agent_sample(self, sample: Sample, task: AgentTask) -> SampleResult:
        tools = task.tools()
        tools_by_name: dict[str, Tool] = {t.name: t for t in tools}
        tool_schemas = [t.to_openai_tool() for t in tools]

        prompt = task.build_prompt(sample)
        messages: list[dict[str, Any]] = []
        system = self._get_system_prompt()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        trace_calls: list[ToolCall] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0
        final_answer = ""
        stop_reason = "complete"
        steps = 0

        try:
            for step in range(task.max_steps):
                steps = step + 1
                out = await self.client.chat(messages, tools=tool_schemas)
                total_prompt_tokens += out.prompt_tokens
                total_completion_tokens += out.completion_tokens
                total_latency_ms += out.latency_ms

                if not out.tool_calls:
                    # Model gave up on tools and answered in text — accept that as the final answer.
                    final_answer = out.text
                    break

                # Record the assistant message with tool_calls verbatim so the next
                # turn's message list is shaped correctly for the API.
                messages.append({
                    "role": "assistant",
                    "content": out.text or None,
                    "tool_calls": out.tool_calls,
                })

                terminated = False
                for tc in out.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except json.JSONDecodeError:
                        args = {}

                    tool = tools_by_name.get(name)
                    if tool is None:
                        result_str = f"Error: unknown tool '{name}'"
                        err = result_str
                    else:
                        try:
                            result_str = tool.handler(**args)
                            err = None
                        except Exception as e:
                            result_str = f"Error: {type(e).__name__}: {e}"
                            err = str(e)

                    trace_calls.append(ToolCall(name=name, arguments=args, result=result_str, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result_str,
                    })

                    if name == "final_answer" and err is None:
                        final_answer = args.get("answer", result_str)
                        terminated = True

                if terminated:
                    break
            else:
                stop_reason = "max_steps"

            trace = AgentTrace(
                final_answer=final_answer,
                tool_calls=trace_calls,
                steps=steps,
                stop_reason=stop_reason,
            )
            score, correct = task.score_trace(sample, trace)

            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                target=sample.target,
                prediction=final_answer,
                score=score,
                correct=correct,
                latency_ms=total_latency_ms,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                tool_calls=[{"name": tc.name, "arguments": tc.arguments, "result": tc.result, "error": tc.error} for tc in trace_calls],
                steps=steps,
                metadata={**sample.metadata, "stop_reason": stop_reason},
            )
        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                target=sample.target,
                prediction=final_answer,
                score=0.0,
                correct=False,
                latency_ms=total_latency_ms,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                error=f"{type(e).__name__}: {e}"[:500],
                tool_calls=[{"name": tc.name, "arguments": tc.arguments, "result": tc.result, "error": tc.error} for tc in trace_calls],
                steps=steps,
                metadata={**sample.metadata, "stop_reason": "error"},
            )

    def _get_system_prompt(self) -> str | None:
        if self.system_prompt_override is None:
            return self.task.system_prompt()
        return self.system_prompt_override
