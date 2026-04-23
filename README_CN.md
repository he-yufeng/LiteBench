# LiteBench

一个 `pip install` 就能用的 LLM / Agent 评测工具。5 分钟跑通第一次 benchmark。

[![PyPI](https://img.shields.io/pypi/v/litebench.svg)](https://pypi.org/project/litebench/)
[![Python](https://img.shields.io/pypi/pyversions/litebench.svg)](https://pypi.org/project/litebench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md)

## 解决什么问题

`inspect_ai` 功能强但重 —— 每个任务要自己写 Solver / Scorer。
`lm-evaluation-harness` 全但偏研究，环境配置复杂。
`promptfoo` 是给 prompt 做对比的，不是给 agent 跑 benchmark。

**LiteBench** 卡在中间这层：给应用层开发者一个轻量 CLI，在自己的模型或 agent 上跑 HumanEval / GSM8K / MMLU / MATH / TruthfulQA / ARC 这些常见任务，不用先搭框架。

```bash
pip install litebench

litebench list
litebench run gsm8k -m deepseek/deepseek-chat -n 50
litebench run humaneval -m kimi -n 20
litebench run mmlu -m claude-sonnet-4-6 --subject computer_security -n 100
litebench run math -m gpt-5 -n 50

# 自定义 YAML 任务
litebench run ./my-task.yaml -m gpt-4o-mini

# 对比不同模型
litebench runs
litebench compare <run-id-1> <run-id-2>
```

## 特性

- **6 个内置任务** —— HumanEval、GSM8K、MMLU、MATH-500、TruthfulQA、ARC-Challenge。
- **100+ 模型** —— 基于 [litellm](https://github.com/BerriAI/litellm)，OpenAI / Anthropic / Gemini / DeepSeek / Kimi / Qwen / GLM / 本地 Ollama 都支持。内置别名简写：`-m opus`、`-m kimi`、`-m deepseek`。
- **流式加载数据集** —— 通过 HuggingFace `datasets` 直接流式拉取,不用手动下载。
- **本地 SQLite 历史** —— 跨模型、跨天的 run 都存下来,方便 diff。
- **并发请求** —— 默认 `--concurrency 8`,异步安全。
- **自定义 YAML 任务** —— 写个 YAML 或 JSONL 就能跑,内置 `number` / `mc` / `regex` / `string` / `llm-judge` 五种 scorer。
- **LLM 打分** —— 接一个 grader model 当评委,处理 free-form 回答。

## 安装

```bash
pip install litebench
```

设置 API key:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
# 按你要用的 provider 来
```

## 用法

### 跑内置任务

```bash
litebench run gsm8k -m deepseek/deepseek-chat -n 100 --concurrency 8
```

输出:

```
           gsm8k · deepseek/deepseek-chat
 Samples       100
 Accuracy      85.0%  (85/100)
 Mean latency  3420 ms
 Tokens        prompt=22,100  completion=58,743
 Duration      57.3s
 Run ID        a51819c4
```

### 模型简写

CLI 接受 litellm 完整字符串或以下简写:

| 简写 | 实际 model |
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

### 自定义 YAML 任务

新建 `my-task.yaml`:

```yaml
name: sql-questions
description: 考 SQL,用正则打分
scorer: regex
regex: "SELECT\\s+.*FROM\\s+users"
system_prompt: |
  只返回 SQL 查询,不要解释
samples:
  - input: "查询所有用户的 email"
    target: "SELECT email FROM users"
  - input: "查询活跃用户"
    target: "SELECT * FROM users WHERE active = TRUE"
```

跑:

```bash
litebench run my-task.yaml -m gpt-4o-mini
```

Scorer 可选: `number` / `mc` / `regex` / `string` (默认,子串匹配) / `llm-judge`。

用 `llm-judge` 时加 `judge_model: gpt-4o-mini` 指定打分模型。

也可以用 JSONL 存样本,不写在 YAML 里:

```yaml
name: my-task
scorer: string
samples_jsonl: ./data.jsonl
```

### 对比 run

```bash
$ litebench runs
$ litebench compare <run-id-1> <run-id-2>
```

## 内置任务一览

| 任务 | 描述 | 数据集 |
| --- | --- | --- |
| `humaneval` | 代码补全,跑隐藏测试 | `openai_humaneval` |
| `gsm8k` | 小学数学应用题 | `gsm8k` (main, test) |
| `mmlu` | 57 学科选择题, `--subject` 过滤 | `cais/mmlu` |
| `math` | 竞赛数学,答案在 `\boxed{…}` | `HuggingFaceH4/MATH-500` |
| `truthfulqa` | MC1 单选 | `truthful_qa` (multiple_choice) |
| `arc` | AI2 科学考试,`--arc-easy` 切换 | `allenai/ai2_arc` |

## 路线图

- ✅ Phase 1 — MVP CLI、3 任务、SQLite 历史
- ✅ Phase 2 — 6 任务、YAML 自定义、LLM judge、31 个回归单测
- ⏳ Phase 3 — Agent 模式 (litellm function calling 的 tool-use eval)
- ⏳ Phase 4 — Web 面板 (FastAPI + React, `litebench serve`)

## 贡献

欢迎 Issue / PR。`pytest tests/` 必须全绿。

## License

MIT
