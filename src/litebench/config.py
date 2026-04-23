from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".litebench"
CONFIG_PATH = CONFIG_DIR / "config.json"
DB_PATH = CONFIG_DIR / "runs.sqlite"

# Shortcut aliases so you can say `--model kimi` instead of the full litellm string.
# litellm-style provider prefix is always explicit so the user isn't surprised by
# which provider gets hit.
MODEL_ALIASES: dict[str, str] = {
    "gpt-5": "gpt-5",
    "gpt-4o": "gpt-4o",
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "gemini": "gemini/gemini-2.5-pro",
    "deepseek": "deepseek/deepseek-chat",
    "kimi": "openrouter/moonshotai/kimi-k2.6",
    "qwen": "openrouter/qwen/qwen3.5-max",
    "glm": "openrouter/zhipu/glm-5",
}


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key(provider: str) -> str | None:
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    cfg = load_config().get("api_keys", {})
    return cfg.get(provider)
