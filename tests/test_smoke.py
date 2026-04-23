"""Smoke-тест всего пайплайна с замоканным GigaChat-клиентом.

Цель: убедиться, что интеграция модулей (архитектуры, судья, Excel I/O, агрегатор)
работает без доступа к реальному API.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import core.architectures  # noqa: F401
from core.gigachat_client import LLMResponse
from core.runner import run


def _mock_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 1000, **_):
    """Очень простой мок: детектит тип вызова по промпту и возвращает подходящий ответ."""
    # Судья
    if "Оцени заново" in prompt or "ОЦЕНКА (JSON)" in prompt:
        return LLMResponse(
            content='{"score": 1, "comment": "Суть совпадает с эталоном."}',
            prompt_tokens=100, completion_tokens=20, total_tokens=120,
            latency_sec=0.05, raw={},
        )
    # SC-агрегатор
    if "selected_answer" in prompt and "ПЯТЬ ОТВЕТОВ МОДЕЛИ" in prompt:
        return LLMResponse(
            content='{"selected_answer": "Агрегированный ответ модели.", "selected_index": 1, "consensus_reason": "Большинство"}',
            prompt_tokens=300, completion_tokens=30, total_tokens=330,
            latency_sec=0.07, raw={},
        )
    # CoT
    if "<размышление>" in prompt or "Шаг 1" in prompt:
        return LLMResponse(
            content="<размышление>Шаг 1: читаю СОП.\nШаг 5: вывод.</размышление>\n<ответ>Да, согласование требуется.</ответ>",
            prompt_tokens=500, completion_tokens=40, total_tokens=540,
            latency_sec=0.1, raw={},
        )
    # Zero-shot
    return LLMResponse(
        content="Да, согласование с отделом рисков требуется.",
        prompt_tokens=400, completion_tokens=15, total_tokens=415,
        latency_sec=0.08, raw={},
    )


def test_full_pipeline_smoke(tmp_path):
    # Подготовка
    sop_path = tmp_path / "sop.md"
    sop_path.write_text("Тестовый СОП", encoding="utf-8")

    questions_path = tmp_path / "questions.xlsx"
    pd.DataFrame([
        {"sop_id": "S1", "question": "Требуется ли согласование?",
         "question_type": "binary", "reference_answer": "Да"},
        {"sop_id": "S1", "question": "Что произойдёт при просрочке?",
         "question_type": "open", "reference_answer": "Начисляются пени."},
    ]).to_excel(questions_path, sheet_name="Questions", index=False)

    output_path = tmp_path / "out.xlsx"

    config = {
        "gigachat": {
            "model": "GigaChat-2-Max",
            "scope": "GIGACHAT_API_CORP",
            "credentials": "mock-credentials",
            "verify_ssl": False,
            "timeout_seconds": 5,
            "max_retries": 1,
            "retry_backoff_base": 0.1,
        },
        "architectures": {
            "zero_shot": {"enabled": True, "temperature": 0.0, "max_tokens": 100},
            "cot": {"enabled": True, "temperature": 0.3, "max_tokens": 500},
            "self_consistency": {
                "enabled": True, "n_runs": 3, "temperature": 0.7,
                "max_tokens": 500, "aggregation_temperature": 0.0,
            },
        },
        "judge": {"temperature": 0.0, "max_tokens": 200, "include_sop_context": False},
        "execution": {
            "max_workers": 2,
            "checkpoint_every_n_questions": 100,
            "continue_on_error": True,
        },
    }

    with patch("core.gigachat_client.GigaChatClient.complete", side_effect=_mock_llm):
        summary = run(sop_path, questions_path, output_path, config)

    # Проверки
    assert output_path.exists(), "Выходной Excel не создан"

    # Summary dict
    assert "zero_shot" in summary
    assert "cot" in summary
    assert "self_consistency" in summary
    for arch in summary.values():
        assert arch["accuracy"] == 1.0, f"Ожидалась accuracy=1.0 при замоканных 'верных' ответах: {arch}"
        assert arch["evaluated"] == 2

    # Проверка структуры Excel — по листу на архитектуру + Metrics
    xl = pd.ExcelFile(output_path)
    assert set(xl.sheet_names) == {"ZeroShot", "CoT", "SelfConsistency", "Metrics"}

    expected_cols = {"question", "reference_answer", "llm_answer", "verdict"}
    for sheet in ["ZeroShot", "CoT", "SelfConsistency"]:
        df = pd.read_excel(output_path, sheet_name=sheet)
        assert set(df.columns) == expected_cols, f"{sheet}: колонки {df.columns}"
        assert len(df) == 2
        binary_rows = df[df["reference_answer"].str.strip().str.lower().str.startswith("да")]
        for v in binary_rows["verdict"]:
            assert v in ("правильно", "неправильно"), f"{sheet}: {v!r}"

    metrics = pd.read_excel(output_path, sheet_name="Metrics")
    # Базовый режим (detailed=False по умолчанию): только 4 колонки
    assert list(metrics.columns) == ["architecture", "total", "correct", "accuracy"]
    assert len(metrics) == 3
    for _, row in metrics.iterrows():
        assert row["accuracy"] == 1.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_full_pipeline_smoke(Path(td))
        print("✓ Smoke-тест пройден")
