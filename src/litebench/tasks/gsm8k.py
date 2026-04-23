from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset

from litebench.core.models import Sample
from litebench.scorers.extract_number import extract_number, numbers_equal
from litebench.tasks.base import Task


def _gold(answer: str) -> str:
    """GSM8K stores the final number after a ``####`` marker at the end of the answer."""
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().replace(",", "")


class GSM8KTask(Task):
    name = "gsm8k"
    description = "Grade school math word problems. Final-answer exact match."

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        ds = load_dataset("gsm8k", "main", split=split, streaming=True)
        taken = 0
        for i, row in enumerate(ds):
            if n is not None and taken >= n:
                break
            yield Sample(
                id=f"gsm8k-{i}",
                input=row["question"],
                target=_gold(row["answer"]),
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "You are solving a grade-school math problem. Reason step by step. "
            "End your response with 'The answer is <number>' on its own line."
        )

    def build_prompt(self, sample: Sample) -> str:
        return sample.input

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        pred = extract_number(prediction)
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        correct = numbers_equal(pred, target)
        return (1.0 if correct else 0.0), correct
