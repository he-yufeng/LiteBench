from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset

from litebench.core.models import Sample
from litebench.scorers.multiple_choice import extract_letter, letters_equal
from litebench.tasks.base import Task


class ARCTask(Task):
    """ARC-Challenge — science exam questions, multiple choice."""

    name = "arc"
    description = "ARC-Challenge (AI2 Reasoning Challenge). Multiple choice."

    def __init__(self, config: str = "ARC-Challenge"):
        self.config = config

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        ds = load_dataset("allenai/ai2_arc", self.config, split=split, streaming=True)
        taken = 0
        for i, row in enumerate(ds):
            if n is not None and taken >= n:
                break
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            target = row["answerKey"]
            # ARC mixes letter labels (A/B/C/D) and numeric labels (1/2/3/4); normalize to letters.
            letter_map = {}
            display_choices = []
            for j, lbl in enumerate(labels):
                letter = chr(ord("A") + j)
                letter_map[lbl] = letter
                display_choices.append((letter, texts[j]))
            if target not in letter_map:
                continue
            yield Sample(
                id=row.get("id", f"arc-{i}"),
                input=row["question"],
                target=letter_map[target],
                metadata={"display_choices": display_choices},
            )
            taken += 1

    def system_prompt(self) -> str | None:
        return (
            "Answer the science question with a single letter (A, B, C, ...). "
            "End with 'The answer is X'."
        )

    def build_prompt(self, sample: Sample) -> str:
        lines = [f"Question: {sample.input}", "", "Choices:"]
        for letter, text in sample.metadata["display_choices"]:
            lines.append(f"{letter}. {text}")
        return "\n".join(lines)

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        pred = extract_letter(prediction)
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        correct = letters_equal(pred, target)
        return (1.0 if correct else 0.0), correct
