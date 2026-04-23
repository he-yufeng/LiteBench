from __future__ import annotations

from litebench.tasks.arc import ARCTask
from litebench.tasks.base import Task
from litebench.tasks.gsm8k import GSM8KTask
from litebench.tasks.humaneval import HumanEvalTask
from litebench.tasks.math_task import MATHTask
from litebench.tasks.mmlu import MMLUTask
from litebench.tasks.truthfulqa import TruthfulQATask

_REGISTRY: dict[str, type[Task]] = {
    "humaneval": HumanEvalTask,
    "gsm8k": GSM8KTask,
    "mmlu": MMLUTask,
    "truthfulqa": TruthfulQATask,
    "math": MATHTask,
    "arc": ARCTask,
}


def get_task(name: str) -> Task:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown task: {name}. Try: {', '.join(_REGISTRY)}")
    return _REGISTRY[key]()


def list_tasks() -> list[str]:
    return list(_REGISTRY)


__all__ = ["Task", "get_task", "list_tasks"]
