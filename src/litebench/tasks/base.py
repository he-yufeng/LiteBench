from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from litebench.core.models import Sample


class Task(ABC):
    """A benchmark task: knows how to load samples, build the prompt, and score."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def load_samples(self, n: int | None = None, split: str = "test") -> Iterable[Sample]:
        """Yield samples from the dataset. ``n`` caps the count when given."""

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        """Return the user prompt for one sample."""

    @abstractmethod
    def score(self, sample: Sample, prediction: str) -> tuple[float, bool]:
        """Return (score_in_[0, 1], correct_flag)."""

    def system_prompt(self) -> str | None:
        return None
