# LiteBench

A pip-installable benchmark runner for LLMs and agents. Five minutes to your first eval.

[![PyPI](https://img.shields.io/pypi/v/litebench.svg)](https://pypi.org/project/litebench/)
[![Python](https://img.shields.io/pypi/pyversions/litebench.svg)](https://pypi.org/project/litebench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[中文文档](README_CN.md)

## What is this?

`inspect_ai` is powerful but heavy — you write Solver and Scorer classes.
`lm-evaluation-harness` is thorough but research-oriented and slow to set up.
`promptfoo` tests prompts, not full agents.

**LiteBench** sits in the middle: an opinionated CLI for app developers who want to
benchmark their model or agent on common tasks (HumanEval / GSM8K / MMLU / MATH / TruthfulQA / ARC)
without having to write a framework first.

```bash
pip install litebench

litebench list
litebench run gsm8k -m deepseek/deepseek-chat -n 50
litebench run humaneval -m gpt-5 -n 20
litebench run mmlu -m claude-sonnet-4-6 --subject computer_security -n 100
litebench run math -m kimi -n 50

# Custom YAML tasks
litebench run ./my-task.yaml -m gpt-4o-mini

# Compare models
litebench runs
litebench compare <run-id-1> <run-id-2>
```

## Features

- **6 built-in tasks** — HumanEval, GSM8K, MMLU, MATH-500, TruthfulQA, ARC-Challenge.
- **100+ model providers via [litellm](https://github.com/BerriAI/litellm)** — OpenAI, Anthropic, Gemini, DeepSeek, Kimi, Qwen, GLM, local Ollama, and more. Shortcuts built in: `-m opus`, `-m kimi`, `-m deepseek`.
- **Streaming datasets** via HuggingFace `datasets` — no manual downloads.
- **Local SQLite run history** — diff runs across models and days.
- **Async concurrency** — `--concurrency 8` default, safely parallel.
- **Custom YAML tasks** — point at a YAML or JSONL and go. Supports `number` / `mc` / `regex` / `string` / `llm-judge` scorers.
- **LLM-as-judge** — plug a grader model in for free-form tasks.

## Install

```bash
pip install litebench
```

Then set the API key for whatever provider you plan to hit:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
# etc.
```

## Usage

### Run a built-in task

```bash
litebench run gsm8k -m deepseek/deepseek-chat -n 100 --concurrency 8
```

Output:

```
           gsm8k · deepseek/deepseek-chat
 Samples       100
 Accuracy      85.0%  (85/100)
 Mean latency  3420 ms
 Tokens        prompt=22,100  completion=58,743
 Duration      57.3s
 Run ID        a51819c4
```

### Model shortcuts

The CLI accepts either a full litellm string or one of the shortcuts:

| Shortcut | Resolves to |
| --- | --- |
| `opus` | `claude-opus-4-7` |
| `sonnet` | `claude-sonnet-4-6` |
| `haiku` | `claude-haiku-4-5-20251001` |
| `gpt-5` | `gpt-5` |
| `gpt-4o` | `gpt-4o` |
| `gemini` | `gemini/gemini-2.5-pro` |
| `deepseek` | `deepseek/deepseek-chat` |
| `kimi` | `openrouter/moonshotai/kimi-k2.6` |
| `qwen` | `openrouter/qwen/qwen3.5-max` |
| `glm` | `openrouter/zhipu/glm-5` |

### Custom YAML task

Create `my-task.yaml`:

```yaml
name: sql-questions
description: Ask for a SQL query, grade with a pattern.
scorer: regex
regex: "SELECT\\s+.*FROM\\s+users"
system_prompt: |
  Return only a SQL query, nothing else.
samples:
  - input: "Get every user's email."
    target: "SELECT email FROM users"
  - input: "Get active users."
    target: "SELECT * FROM users WHERE active = TRUE"
```

Then run it:

```bash
litebench run my-task.yaml -m gpt-4o-mini
```

Supported scorers: `number` / `mc` / `regex` / `string` (default: substring match) / `llm-judge`.

For `llm-judge`, add `judge_model: gpt-4o-mini` (or any litellm-supported model).

You can also load samples from JSONL instead of inline:

```yaml
name: my-task
scorer: string
samples_jsonl: ./data.jsonl
```

### Compare runs

```bash
$ litebench runs
                                Recent runs
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Run      ┃ Task  ┃ Model       ┃ Samples ┃ Accuracy ┃ When             ┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 10ab7654 │ gsm8k │ gpt-4o      │     100 │    89.0% │ 2026-04-23 17:38 │
│ 86d845e0 │ gsm8k │ gpt-4o-mini │     100 │    80.0% │ 2026-04-23 17:37 │
└──────────┴───────┴─────────────┴─────────┴──────────┴──────────────────┘

$ litebench compare 10ab7654 86d845e0
                              Comparing 2 runs
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Model       ┃ Task  ┃ N   ┃ Accuracy ┃ Mean latency ┃ Tokens (p/c)  ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ gpt-4o      │ gsm8k │ 100 │    89.0% │       3710ms │  8,700 / 23.9k│
│ gpt-4o-mini │ gsm8k │ 100 │    80.0% │       4230ms │  8,700 / 22.3k│
└─────────────┴───────┴─────┴──────────┴──────────────┴───────────────┘
```

## Built-in tasks

| Task | Description | Dataset |
| --- | --- | --- |
| `humaneval` | Code completion, executed against hidden tests | `openai_humaneval` |
| `gsm8k` | Grade-school word problems | `gsm8k` (main, test) |
| `mmlu` | 57-subject multiple choice; use `--subject` | `cais/mmlu` |
| `math` | Competition-level math, answer in `\boxed{…}` | `HuggingFaceH4/MATH-500` |
| `truthfulqa` | MC1 single-correct multiple choice | `truthful_qa` (multiple_choice) |
| `arc` | AI2 science exam; `--arc-easy` for Easy split | `allenai/ai2_arc` (Challenge) |

## Roadmap

- ✅ Phase 1 — MVP CLI, 3 tasks, SQLite history
- ✅ Phase 2 — 6 tasks, YAML custom, LLM judge, 31 regression tests
- ⏳ Phase 3 — Agent mode (tool-use eval via litellm function calling)
- ⏳ Phase 4 — Web dashboard (FastAPI + React, `litebench serve`)

## Contributing

Issues and PRs welcome. `pytest tests/` should stay green.

## License

MIT
