"""Execute generated Python in a short-lived subprocess and check pass/fail.

Used for HumanEval-style "return a function, check it against the tests"
benchmarks. The subprocess boundary is for fault isolation — an infinite loop or
segfault in the model's code shouldn't take down the runner. It is **not** a
security sandbox. Do not feed untrusted model output to real hardware.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python_block(text: str) -> str:
    """Pull a python fence out of the model response, or return the raw text."""
    matches = _CODE_BLOCK.findall(text)
    if matches:
        # Prefer the last block — models sometimes explain, then give the answer.
        return matches[-1].strip()
    return text.strip()


def run_humaneval_test(prompt: str, completion: str, test: str, entry_point: str, timeout: int = 8) -> tuple[bool, str]:
    """Returns (passed, stderr_if_failed)."""
    # Don't reach for textwrap.dedent + multiline f-string here: dedent uses the
    # minimum common indent, and {prompt}/{completion} insert multi-line blocks
    # at column 0, so the first line ends up over-indented. Plain string join
    # keeps every block flush-left.
    parts = [prompt, completion, test, f"check({entry_point})", 'print("__LITEBENCH_OK__")']
    code = "\n".join(parts)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and "__LITEBENCH_OK__" in result.stdout:
            return True, ""
        return False, (result.stderr or result.stdout or "non-zero exit").strip()[:500]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:500]
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
