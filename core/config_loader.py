"""Загрузка и валидация config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "gigachat": {
        "scope": "GIGACHAT_API_CORP",
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_backoff_base": 1.0,
        "verify_ssl": False,
    },
    "architectures": {
        "zero_shot": {"enabled": True, "temperature": 0.0, "max_tokens": 500},
        "cot": {"enabled": True, "temperature": 0.3, "max_tokens": 1500},
        "self_consistency": {
            "enabled": True,
            "n_runs": 5,
            "temperature": 0.7,
            "max_tokens": 1500,
            "aggregation_temperature": 0.0,
        },
    },
    "judge": {
        "temperature": 0.0,
        "max_tokens": 400,
        "include_sop_context": True,
    },
    "execution": {
        "max_workers": 3,
        "checkpoint_every_n_questions": 20,
        "continue_on_error": True,
        "random_seed": 42,
    },
    "logging": {"level": "INFO", "log_to_file": True, "log_dir": "./logs"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    """Читаем YAML + подмешиваем дефолты."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)

    gc = cfg["gigachat"]
    if not gc.get("model"):
        raise ValueError("config.gigachat.model не задан")
    if not gc.get("scope"):
        raise ValueError("config.gigachat.scope не задан")

    return cfg
