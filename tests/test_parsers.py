"""Тесты парсеров CoT-тегов и нормализаторов ответов."""

from __future__ import annotations

import pytest

from core.models import (
    QuestionType,
    extract_cot_answer,
    normalize_binary_answer,
    normalize_for_voting,
)


class TestExtractCotAnswer:
    def test_standard_xml_tags(self):
        raw = """<размышление>
        Шаг 1: читаю СОП.
        Шаг 5: вывод.
        </размышление>
        <ответ>
        Да, согласование требуется при сумме свыше 500 тыс.
        </ответ>"""
        answer, reasoning = extract_cot_answer(raw)
        assert answer == "Да, согласование требуется при сумме свыше 500 тыс."
        assert reasoning is not None
        assert "Шаг 1" in reasoning

    def test_tags_case_insensitive(self):
        raw = "<РАЗМЫШЛЕНИЕ>abc</РАЗМЫШЛЕНИЕ><Ответ>Нет</Ответ>"
        answer, reasoning = extract_cot_answer(raw)
        assert answer == "Нет"
        assert reasoning == "abc"

    def test_no_tags_fallback_to_marker(self):
        raw = "Я размышляю... Финальный ответ: Да, требуется."
        answer, reasoning = extract_cot_answer(raw)
        assert answer == "Да, требуется."
        assert reasoning is None

    def test_no_tags_no_marker(self):
        raw = "Просто ответ без структуры"
        answer, reasoning = extract_cot_answer(raw)
        assert answer == "Просто ответ без структуры"
        assert reasoning is None

    def test_multiline_answer(self):
        raw = "<ответ>Строка 1.\nСтрока 2.\nСтрока 3.</ответ>"
        answer, _ = extract_cot_answer(raw)
        assert "Строка 1." in answer and "Строка 3." in answer


class TestNormalizeBinary:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Да, требуется", "да"),
            ("нет, не нужно", "нет"),
            ("  Да.", "да"),
            ("«Нет»", "нет"),
            ("Требуется согласование", "да"),
            ("Не требуется", "нет"),
            ("", None),
            ("Возможно", None),
            ("Правильно", "да"),
            ("Неправильно", "нет"),
        ],
    )
    def test_cases(self, text, expected):
        assert normalize_binary_answer(text) == expected


class TestNormalizeForVoting:
    def test_binary_maps_to_da_net(self):
        assert normalize_for_voting("Да, требуется", QuestionType.BINARY) == "да"
        assert normalize_for_voting("Нет, не нужно", QuestionType.BINARY) == "нет"
        assert normalize_for_voting("я не знаю", QuestionType.BINARY) == "n/a"

    def test_multiple_lowercase_truncate(self):
        text = "Вариант А: согласование с руководителем отдела операционного центра"
        norm = normalize_for_voting(text, QuestionType.MULTIPLE)
        assert norm == norm.lower()
        assert len(norm) <= 80

    def test_open_handles_empty(self):
        assert normalize_for_voting("", QuestionType.OPEN) == "n/a"

    def test_whitespace_collapsed(self):
        raw = "Начисление    пени\n\n  за просрочку"
        norm = normalize_for_voting(raw, QuestionType.OPEN)
        assert "  " not in norm
