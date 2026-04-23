from litebench.scorers.exec_code import extract_python_block, run_humaneval_test
from litebench.scorers.extract_number import extract_number, numbers_equal
from litebench.scorers.multiple_choice import extract_letter, letters_equal


class TestExtractNumber:
    def test_answer_is_phrase(self):
        assert extract_number("Step 1... Step 2... The answer is 42") == "42"

    def test_gsm8k_marker(self):
        assert extract_number("Let me compute. #### 18") == "18"

    def test_boxed(self):
        assert extract_number(r"We have $x = 7$, so \boxed{7}.") == "7"

    def test_commas(self):
        assert extract_number("Total cost: $1,234 dollars. The answer is 1,234") == "1234"

    def test_negative(self):
        assert extract_number("The answer is -15") == "-15"

    def test_decimal_normalizes_integer(self):
        assert extract_number("Answer: 42.0") == "42"

    def test_decimal_keeps_fractional(self):
        assert extract_number("The answer is 3.14") == "3.14"

    def test_falls_back_to_last_number(self):
        assert extract_number("We get 5 then later 10 then finally 17 as the total.") == "17"

    def test_no_number(self):
        assert extract_number("I don't know.") is None

    def test_empty(self):
        assert extract_number("") is None


class TestNumbersEqual:
    def test_exact(self):
        assert numbers_equal("42", "42") is True

    def test_none_pred(self):
        assert numbers_equal(None, "42") is False

    def test_formatted_target(self):
        # target may come from dataset with commas; pred is already normalized
        assert numbers_equal("1234", "1,234") is True


class TestExtractLetter:
    def test_the_answer_is(self):
        assert extract_letter("After weighing the options, the answer is B.") == "B"

    def test_standalone_letter_on_its_own_line(self):
        assert extract_letter("Long reasoning here.\n\nC\n") == "C"

    def test_parenthesized(self):
        assert extract_letter("The answer is (D)") == "D"

    def test_option_x(self):
        assert extract_letter("I'd pick option A because...") == "A"

    def test_no_letter(self):
        assert extract_letter("I'm not sure.") is None


class TestExecCode:
    def test_extract_python_block_basic(self):
        text = "Here's the solution:\n```python\ndef foo():\n    return 1\n```"
        assert "def foo():" in extract_python_block(text)

    def test_extract_picks_last_block(self):
        text = "```python\n# first\n```\n\nActually:\n```python\n# second\n```"
        assert "second" in extract_python_block(text)

    def test_no_fence_returns_raw(self):
        assert extract_python_block("def foo(): return 1") == "def foo(): return 1"

    def test_run_humaneval_pass(self):
        prompt = "def add(a, b):\n    '''return a + b'''\n"
        completion = "def add(a, b):\n    return a + b\n"
        test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
"""
        passed, err = run_humaneval_test(prompt, completion, test, "add", timeout=5)
        assert passed is True, err

    def test_run_humaneval_fail(self):
        prompt = "def add(a, b):\n    '''return a + b'''\n"
        completion = "def add(a, b):\n    return a - b\n"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        passed, err = run_humaneval_test(prompt, completion, test, "add", timeout=5)
        assert passed is False
        assert "AssertionError" in err or "assert" in err.lower()


class TestLettersEqual:
    def test_case_insensitive(self):
        assert letters_equal("b", "B") is True

    def test_none_pred(self):
        assert letters_equal(None, "A") is False
