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
        match = _VERDICT.search(result.text)
        if match:
            verdict = match.group(1).lower() == "yes"
            return verdict, result.text.strip()
        return False, result.text.strip()
