from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset

from litebench.core.models import Sample
from litebench.scorers.exec_code import extract_python_block, run_humaneval_test
from litebench.tasks.base import Task


class HumanEvalTask(Task):
    name = "humaneval"
    description = "OpenAI HumanEval. Execute generated Python against hidden tests."

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        ds = load_dataset("openai/openai_humaneval", split=split, streaming=True)
        taken = 0
        for row in ds:
            if n is not None and taken >= n:
                break
            yield Sample(
                id=row["task_id"],
                input=row["prompt"],
                target=row["canonical_solution"],
                metadata={
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "prompt": row["prompt"],
                },
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "You are a coding assistant. Complete the function body. "
            "Respond with a single Python code block containing the full function."
        )

    def build_prompt(self, sample: Sample) -> str:
        return (
            f"Complete the following Python function. Return ONLY the full function "
            f"(signature + body) inside a ```python code block.\n\n"
            f"{sample.input}"
        )

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        completion = extract_python_block(prediction)
        passed, _ = run_humaneval_test(
            prompt=sample.metadata["prompt"],
            completion=completion,
            test=sample.metadata["test"],
            entry_point=sample.metadata["entry_point"],
        )
        return (1.0 if passed else 0.0), passed
