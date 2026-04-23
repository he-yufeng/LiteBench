from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset

from litebench.core.models import Sample
from litebench.scorers.multiple_choice import extract_letter, letters_equal
from litebench.tasks.base import Task

_LETTERS = ["A", "B", "C", "D"]


class MMLUTask(Task):
    name = "mmlu"
    description = "MMLU multiple-choice (57 subjects). Accepts optional --subject filter."

    def __init__(self, subject: str | None = None):
        self.subject = subject

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        # "all" is a convenience config that merges every subject.
        config = self.subject or "all"
        ds = load_dataset("cais/mmlu", config, split=split, streaming=True)
        taken = 0
        for i, row in enumerate(ds):
            if n is not None and taken >= n:
                break
            choices = row["choices"]
            question = row["question"]
            answer_idx = int(row["answer"])
            yield Sample(
                id=f"mmlu-{row.get('subject', 'all')}-{i}",
                input=question,
                target=_LETTERS[answer_idx],
                metadata={"choices": choices, "subject": row.get("subject", "all")},
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "Answer the multiple-choice question by stating a single letter "
            "(A, B, C, or D). End your response with 'The answer is X'."
        )

    def build_prompt(self, sample: Sample) -> str:
        choices = sample.metadata["choices"]
        lines = [f"Question: {sample.input}", "", "Choices:"]
        lines += [f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)]
        return "\n".join(lines)

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        pred = extract_letter(prediction)
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        correct = letters_equal(pred, target)
        return (1.0 if correct else 0.0), correct
