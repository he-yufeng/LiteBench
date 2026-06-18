"""GSM8K re-run as an agent task: model calls a calculator tool instead of reasoning inline.

This is the motivating example for LiteBench's agent mode. Stock GSM8K measures
chain-of-thought + arithmetic in one go; giving the model a calculator
separates the reasoning signal from the arithmetic signal and matches how
production agents are actually built.
"""

from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset

from litebench.agent.base import AgentTask, AgentTrace, Tool
from litebench.agent.tools import CALCULATOR, FINAL_ANSWER
from litebench.core.models import Sample
from litebench.scorers.extract_number import extract_number, numbers_equal


def _gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().replace(",", "")


class GSM8KAgentTask(AgentTask):
    name = "gsm8k-agent"
    description = "GSM8K with a calculator tool. Tests tool-use + reasoning."
    max_steps = 12

    def tools(self) -> list[Tool]:
        return [CALCULATOR, FINAL_ANSWER]

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
        taken = 0
        for i, row in enumerate(ds):
            if n is not None and taken >= n:
                break
            yield Sample(
                id=f"gsm8k-agent-{i}",
                input=row["question"],
                target=_gold(row["answer"]),
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "You are solving a math word problem. Use the calculator tool for every "
            "arithmetic step — do not compute numbers in your head. When you have "
            "the final answer, call final_answer with just the number."
        )

    def build_prompt(self, sample: Sample) -> str:
        return sample.input

    def score_trace(self, sample: Sample, trace: AgentTrace) -> tuple[float, bool]:
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        # Prefer the explicit final_answer tool submission; fall back to extracting
        # from whatever prose the model emitted.
        pred = extract_number(trace.final_answer)
        correct = numbers_equal(pred, target)
        return (1.0 if correct else 0.0), correct
