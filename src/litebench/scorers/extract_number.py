"""Pull the final numeric answer out of a free-form model response.

Strategy, in order:
1. ``\\boxed{...}`` (LaTeX style, often emitted by math-trained models).
2. ``#### <number>`` (GSM8K's own answer convention).
3. ``The answer is X`` / ``answer: X`` (common chain-of-thought endings).
4. Last number in the text.
"""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_GSM_MARKER = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_ANSWER_IS = re.compile(
    r"(?:final answer|the answer is|answer\s*[:=])\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _normalize(raw: str) -> str | None:
    raw = raw.strip().rstrip(".").replace(",", "").replace("$", "")
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    if val.is_integer():
        return str(int(val))
    return str(val)


def extract_number(text: str) -> str | None:
    if not text:
        return None

    m = _BOXED.search(text)
    if m:
        nums = _NUMBER.findall(m.group(1))
        if nums:
            return _normalize(nums[-1])

    m = _GSM_MARKER.search(text)
    if m:
        return _normalize(m.group(1))

    m = _ANSWER_IS.search(text)
    if m:
        return _normalize(m.group(1))

    nums = _NUMBER.findall(text)
    if nums:
        return _normalize(nums[-1])

    return None


def numbers_equal(pred: str | None, target: str) -> bool:
    if pred is None:
        return False
    ref = _normalize(target)
    return ref is not None and pred == ref
