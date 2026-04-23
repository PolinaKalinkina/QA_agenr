"""Тесты majority voting в SelfConsistency и JSON-парсера в judge."""

from __future__ import annotations

from core.architectures.self_consistency import (
    SelfConsistencyArchitecture,
    _extract_json_object,
)
from core.judge import _coerce_verdict, _extract_json
from core.models import QuestionType


class TestMajorityVote:
    def test_binary_all_same(self):
        votes = ["Да, требуется", "Да, однозначно", "Да", "Да, в любом случае", "Да"]
        result = SelfConsistencyArchitecture._majority_vote(votes, QuestionType.BINARY)
        assert result.lower().startswith("да")

    def test_binary_majority_wins(self):
        votes = ["Да", "Да", "Да", "Нет", "Нет"]
        result = SelfConsistencyArchitecture._majority_vote(votes, QuestionType.BINARY)
        assert result.lower().startswith("да")

    def test_binary_all_na_returns_first(self):
        votes = ["непонятно", "возможно", "может быть", "не уверен", "трудно сказать"]
        result = SelfConsistencyArchitecture._majority_vote(votes, QuestionType.BINARY)
        # Все нормализуются в "n/a", возвращается первый оригинальный
        assert result == "непонятно"

    def test_multiple_choice_wins(self):
        votes = [
            "Вариант А",
            "Вариант Б",
            "Вариант А",
            "Вариант А",
            "Вариант В",
        ]
        result = SelfConsistencyArchitecture._majority_vote(votes, QuestionType.MULTIPLE)
        assert result == "Вариант А"


class TestJsonExtractors:
    def test_plain_json(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = "```json\n{\"score\": 1, \"comment\": \"ok\"}\n```"
        result = _extract_json_object(text)
        assert result == {"score": 1, "comment": "ok"}

    def test_json_with_surrounding_text(self):
        text = "Вот мой ответ: {\"score\": 0, \"comment\": \"неверно\"} надеюсь помог"
        result = _extract_json_object(text)
        assert result == {"score": 0, "comment": "неверно"}

    def test_broken_json_returns_none(self):
        assert _extract_json_object("не json вообще") is None

    def test_judge_extract_greedy_braces(self):
        # Случай с вложенным объектом
        text = 'junk {"score": 1, "comment": "всё {ok}"} more junk'
        result = _extract_json(text)
        assert result is not None
        assert result.get("score") == 1


class TestCoerceVerdict:
    def test_int_score(self):
        v = _coerce_verdict({"score": 1, "comment": "ок"})
        assert v is not None and v.score == 1

    def test_string_score(self):
        v = _coerce_verdict({"score": "0", "comment": "нет"})
        assert v is not None and v.score == 0

    def test_ru_string_score(self):
        v = _coerce_verdict({"score": "верно", "comment": "ok"})
        assert v is not None and v.score == 1

    def test_invalid_score(self):
        assert _coerce_verdict({"score": 5, "comment": "x"}) is None
        assert _coerce_verdict({"score": "maybe"}) is None

    def test_missing_comment(self):
        v = _coerce_verdict({"score": 1})
        assert v is not None and v.comment == "(без комментария)"
