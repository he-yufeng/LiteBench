<div align="center">

<img src="docs/banner.png" alt="LiteBench — benchmark runner for LLMs and agents" width="100%">

[![PyPI](https://img.shields.io/pypi/v/litebench.svg)](https://pypi.org/project/litebench/)
[![Python](https://img.shields.io/pypi/pyversions/litebench.svg)](https://pypi.org/project/litebench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[**Quick Start**](#usage) · [**Built-in Tasks**](#built-in-tasks) · [**Custom Tasks**](#custom-tasks) · [中文](README_CN.md)

</div>

<p align="center"><img src="docs/demo.png" alt="litebench run gsm8k" width="580"></p>

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
litebench run gsm8k -m openrouter/qwen/qwen3-235b-a22b-thinking-2507 --system-prompt "Give only the final answer. No explanation. No steps. One line." -n 50
litebench run humaneval -m gpt-5 -n 20
litebench run mmlu -m claude-sonnet-4-6 --subject computer_security -n 100
litebench run math -m kimi -n 50

# Custom YAML tasks
litebench run ./my-task.yaml -m gpt-4o-mini

# Compare models
litebench runs
litebench compare <run-id-1> <run-id-2>
litebench export <run-id> -o run.json
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

Export a saved run, including every sample, for reports or offline analysis:

```bash
litebench export 10ab7654 -o gsm8k-gpt4o.json
litebench export 10ab7654 --format jsonl -o gsm8k-gpt4o.jsonl
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

## Agent mode

Pass a task that exposes tools and LiteBench runs a full multi-turn rollout
instead of a single chat:

```bash
litebench run gsm8k-agent -m gpt-5 -n 50
```

The built-in `gsm8k-agent` task gives the model a `calculator` tool and a
`final_answer` tool, then scores whichever number it submits. The recorded
per-sample trace (tool name, arguments, result) is kept in the SQLite history
and can be dumped with `--json-out`:

```
gsm8k-agent-0 | correct=True | steps=3 | final="18"
  → calculator({'expression': '16 - 3 - 4'}) = 9
  → calculator({'expression': '9 * 2'}) = 18
  → final_answer({'answer': '18'}) = 18
```

Custom agent tasks are a Python subclass (`AgentTask`) — see `src/litebench/tasks/gsm8k_agent.py`.

## Web dashboard

```bash
pip install 'litebench[web]'
litebench serve
# → open http://127.0.0.1:8600
```

Three tabs:
- **Runs** — every run you've saved, clickable for full sample-by-sample breakdown (including per-sample agent tool traces).
- **Compare** — accuracy heatmap across (task × model), shows the latest run per pair.
- **Tasks** — the built-in task registry.

Pure single-file HTML + vanilla JS — no React, no build step, works offline.

## Roadmap

**Shipped:** the CLI, six built-in benchmarks (HumanEval, GSM8K, MMLU, MATH-500, plus YAML-defined custom tasks), an LLM-as-judge mode, agent/tool-use evaluation via litellm function calling, a SQLite run history, and a `litebench serve` web dashboard — all under a regression suite that stays green.

**Planned:**

- **A pass@k sampler** — run each task n times and report pass@1 / pass@k, so a model's reliability is visible, not just whether one sample happened to pass.
- **Resumable runs** — checkpoint partway through a long benchmark and continue, instead of paying for the whole sweep again after an interruption.
- **More built-in tasks** — a code-repair task and a tool-use task from real traces, since the runner already supports both shapes and only the curated dataset is missing.
- **Cost-aware comparison** — sort the leaderboard by accuracy-per-dollar, not just accuracy, using the token data each run already records.

## Contributing

Issues and PRs welcome. `pytest tests/` should stay green.

## Related Projects

LiteBench is part of how I benchmark and watch LLM systems. A few related tools:

- **[CoreCoder](https://github.com/he-yufeng/CoreCoder)** — want to understand how a coding agent really works? Read the whole ~1k-line engine end to end, not a black box.
- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — dropped into an unfamiliar codebase? It gives you a guided wiki and a where-to-start reading path, a self-hostable DeepWiki alternative.
- **[AgentProbe](https://github.com/he-yufeng/AgentProbe)** — catch when your LLM agent silently changes behavior: snapshot tests for agents, run in pytest.
- **[agentcikit](https://github.com/he-yufeng/agentcikit)** — the CI safety layer for LLM agents: replay runs, fence tool calls, and triage failures before they ship.

## License

MIT
