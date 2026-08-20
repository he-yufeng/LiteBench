"""The unbiased pass@k estimator.

Running each task n times and reporting pass@k answers "is the model reliable"
instead of "did one sample happen to pass". The estimator is the standard
unbiased one: pass@k = 1 - C(n-c, k) / C(n, k), where c of n samples are
correct.
"""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for a task with ``c`` correct out of ``n`` samples.

    Returns 0.0 when k exceeds n (not enough samples to judge) is avoided by
    the caller clamping k first; this function assumes 1 <= k <= n.
    """
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(correct_per_task: list[int], n: int, k: int) -> float:
    """Mean pass@k across tasks, each with ``n`` samples."""
    if not correct_per_task:
        return 0.0
    return sum(pass_at_k(n, c, k) for c in correct_per_task) / len(correct_per_task)
