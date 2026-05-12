"""Load a user-defined task from a YAML file.

Example ``my-task.yaml``::

    name: simple-math
    description: Three-term addition for smoke tests.
    scorer: number                      # number | string | regex | mc | llm-judge
    system_prompt: |
      Add the numbers and return just the integer result.
    samples:
      - input: "1 + 2 + 3"
        target: "6"
      - input: "10 + 20 + 30"
        target: "60"

Or, instead of inline ``samples``, point at a JSONL file::

    samples_jsonl: ./my-data.jsonl      # each line: {"input": "...", "target": "..."}

For ``scorer: llm-judge`` you can also set ``judge_model: gpt-4o-mini``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from litebench.core.models import Sample
from litebench.scorers.extract_number import extract_number, numbers_equal
from litebench.scorers.llm_judge import LLMJudge
from litebench.scorers.multiple_choice import extract_letter, letters_equal
from litebench.tasks.base import Task


class CustomTask(Task):
    def __init__(self, path: Path):
        self.path = Path(path)
        spec = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: YAML root must be a mapping.")
        self.name = spec.get("name", self.path.stem)
        self.description = spec.get("description", f"Custom task from {self.path.name}")
        self._system_prompt = spec.get("system_prompt")
        self._scorer_kind = spec.get("scorer", "string").lower()
        self._regex_pattern = spec.get("regex")
        self._judge_model = spec.get("judge_model", "gpt-4o-mini")
        self._judge: LLMJudge | None = None
        self._samples = self._load_samples_spec(spec)

    def _load_samples_spec(self, spec: dict[str, Any]) -> list[Sample]:
        raw: list[dict[str, Any]]
        if "samples" in spec:
            raw = spec["samples"]
        elif "samples_jsonl" in spec:
            jsonl_path = (self.path.parent / spec["samples_jsonl"]).resolve()
            raw = []
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
        else:
            raise ValueError(
                f"{self.path}: YAML must contain 'samples' or 'samples_jsonl'."
            )

        samples: list[Sample] = []
        for i, row in enumerate(raw):
            target = row["target"]
            samples.append(
                Sample(
                    id=row.get("id", f"{self.name}-{i}"),
                    input=row["input"],
                    target=target,
                    metadata={k: v for k, v in row.items() if k not in {"id", "input", "target"}},
                )
            )
        return samples

    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        # Custom tasks ignore ``split`` — the YAML is the only source of truth.
        yield from (self._samples[:n] if n is not None else self._samples)

    def system_prompt(self) -> str | None:
        return self._system_prompt

    def build_prompt(self, sample: Sample) -> str:
        return sample.input

    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        kind = self._scorer_kind

        if kind == "number":
            return (1.0, True) if numbers_equal(extract_number(prediction), target) else (0.0, False)

        if kind == "mc":
            return (1.0, True) if letters_equal(extract_letter(prediction), target) else (0.0, False)

        if kind == "regex":
            if not self._regex_pattern:
                raise ValueError(f"{self.path}: scorer=regex requires a 'regex' field.")
            match = re.search(self._regex_pattern, prediction)
            if match:
                captured = match.group(1) if match.groups() else match.group(0)
                return (1.0, True) if captured.strip() == target.strip() else (0.0, False)
            return 0.0, False

        if kind == "llm-judge":
            if self._judge is None:
                self._judge = LLMJudge(model=self._judge_model)
            return self._grade_async(sample.input, prediction, target)

        # default: case-insensitive substring match of the gold in the response
        return (1.0, True) if target.strip().lower() in prediction.lower() else (0.0, False)

    async def _grade_async(self, question: str, prediction: str, target: str) -> tuple[float, bool]:
        passed, _ = await self._judge.grade(question, prediction, target)
        return (1.0, True) if passed else (0.0, False)
