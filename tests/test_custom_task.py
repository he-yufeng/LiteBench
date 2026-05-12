from pathlib import Path

import pytest
import yaml

from litebench.tasks.custom import CustomTask


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_number_scorer_correct(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "name": "nums",
            "scorer": "number",
            "samples": [{"input": "1+1?", "target": "2"}],
        },
    )
    task = CustomTask(path)
    samples = list(task.load_samples())
    assert len(samples) == 1
    score, correct = task.score(samples[0], "The answer is 2.")
    assert correct is True
    assert score == 1.0


def test_mc_scorer_wrong(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "name": "mc",
            "scorer": "mc",
            "samples": [{"input": "A or B?", "target": "A"}],
        },
    )
    task = CustomTask(path)
    samples = list(task.load_samples())
    _, correct = task.score(samples[0], "The answer is B")
    assert correct is False


def test_regex_scorer(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "name": "re",
            "scorer": "regex",
            "regex": r"ID=(\w+)",
            "samples": [{"input": "give me id", "target": "abc123"}],
        },
    )
    task = CustomTask(path)
    samples = list(task.load_samples())
    _, correct_yes = task.score(samples[0], "Sure, ID=abc123 here you go.")
    _, correct_no = task.score(samples[0], "Sure, ID=xyz999 here you go.")
    assert correct_yes is True
    assert correct_no is False


def test_default_substring(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "name": "sub",
            "samples": [{"input": "capital of france?", "target": "Paris"}],
        },
    )
    task = CustomTask(path)
    samples = list(task.load_samples())
    _, yes = task.score(samples[0], "The capital is Paris, France.")
    _, no = task.score(samples[0], "The capital is London.")
    assert yes is True
    assert no is False


def test_missing_samples_raises(tmp_path):
    path = _write_yaml(tmp_path, {"name": "bad"})
    with pytest.raises(ValueError, match="samples"):
        CustomTask(path)


def test_n_limit(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "name": "five",
            "scorer": "number",
            "samples": [{"input": f"{i}?", "target": str(i)} for i in range(5)],
        },
    )
    task = CustomTask(path)
    assert len(list(task.load_samples(n=2))) == 2
    assert len(list(task.load_samples())) == 5


def test_custom_task_reads_utf8_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "samples.jsonl"
    data_path.write_text(
        '{"input": "你好，世界？", "target": "世界"}\n',
        encoding="utf-8",
    )
    path = _write_yaml(
        tmp_path,
        {
            "name": "中文任务",
            "samples_jsonl": data_path.name,
        },
    )

    task = CustomTask(path)
    samples = list(task.load_samples())

    assert task.name == "中文任务"
    assert samples[0].input == "你好，世界？"
    assert samples[0].target == "世界"
