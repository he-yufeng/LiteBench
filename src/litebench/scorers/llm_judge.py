"""LLM-as-judge scorer.

Feeds (question, model_answer, reference_answer) into a grader model and parses a
yes/no verdict. Used for free-form tasks where exact match doesn't apply.

Kept intentionally simple — one prompt, one call, one regex. Nothing about chain-
of-thought grading, majority voting, or bias correction here; those belong in a
separate evaluator if someone wants them.
"""

from __future__ import annotations

import re

from litebench.llm.client import LLMClient

_DEFAULT_JUDGE_PROMPT = """\
You are grading a model's answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

Model's answer:
{answer}

Does the model's answer match the reference, accounting for different phrasings
but not for substantive differences? Reply with a single word: YES or NO.
"""

_VERDICT = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_verdict(text: str) -> bool:
    """Read a yes/no verdict from a judge reply, taking the LAST match.

    The grader is asked for a single word, but models sometimes reason before
    concluding ("the phrasings differ but overall YES"). Matching the first
    yes/no would read such a reply by an earlier mention and flip the grade, so
    take the final verdict instead. Unparseable replies grade as False.
    """
    matches = _VERDICT.findall(text)
    return bool(matches) and matches[-1].lower() == "yes"


class LLMJudge:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_template: str = _DEFAULT_JUDGE_PROMPT,
        temperature: float = 0.0,
    ):
        self.client = LLMClient(model=model, temperature=temperature, max_tokens=16)
        self.prompt_template = prompt_template

    async def grade(self, question: str, answer: str, reference: str) -> tuple[bool, str]:
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            reference=reference,
        )
        result = await self.client.complete(prompt)
        return _parse_verdict(result.text), result.text.strip()
