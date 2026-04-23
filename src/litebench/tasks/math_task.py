from __future__ import annotations

import re
from collections.abc import Iterable

from datasets import load_dataset

from litebench.core.models import Sample
from litebench.scorers.extract_number import extract_number, numbers_equal
from litebench.tasks.base import Task

_BOXED = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_gold_from_solution(solution: str) -> str | None:
    """MATH gold is whatever is inside the final \\boxed{...} in the official solution."""
    matches = _BOXED.findall(solution)
    if not matches:
        return None
    return matches[-1].strip()


class MATHTask(Task):
    name = "math"
    description = "MATH-500 (OpenAI's held-out subset of Hendrycks MATH). Answer in \\boxed{...}."

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        # HuggingFaceH4/MATH-500 is the clean, always-available mirror of the
        # original hendrycks/competition_math held-out set. The upstream
        # repository was taken down in early 2025.
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split, streaming=True)
        taken = 0
        for i, row in enumerate(ds):
            if n is not None and taken >= n:
                break
            gold = row.get("answer") or _extract_gold_from_solution(row.get("solution", ""))
            if not gold:
                continue
            yield Sample(
                id=row.get("unique_id") or f"math-{i}",
                input=row["problem"],
                target=gold,
                metadata={
                    "level": row.get("level"),
                    "subject": row.get("subject"),
                    "solution": row.get("solution", ""),
                },
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "Solve the math problem step by step. "
            "Put your final answer inside \\boxed{...} at the end."
        )

    def build_prompt(self, sample: Sample) -> str:
        return sample.input

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        # Try structural boxed-value match first (handles fractions / expressions);
        # fall back to plain number extraction when the gold is just a number.
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        boxed = _BOXED.findall(prediction)
        if boxed:
            pred_boxed = boxed[-1].strip()
            if _normalize_math(pred_boxed) == _normalize_math(target):
                return 1.0, True

        pred_num = extract_number(prediction)
        if numbers_equal(pred_num, target):
            return 1.0, True
        return 0.0, False


def _normalize_math(s: str) -> str:
    s = s.replace(" ", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    # Drop trailing periods/commas that sometimes end the gold in the dataset.
    return s.rstrip(".").rstrip(",")
