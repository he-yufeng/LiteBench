"""Match a multiple-choice letter answer (A/B/C/D) from a model response."""

from __future__ import annotations

import re

_LETTER_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-D])\)?\s*[.)]?\s*$", re.MULTILINE),
    re.compile(r"\boption\s+\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\b([A-D])\)\s"),
    re.compile(r"\b([A-D])\b"),
]


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def letters_equal(pred: str | None, target: str) -> bool:
    return pred is not None and pred.upper() == target.upper()
