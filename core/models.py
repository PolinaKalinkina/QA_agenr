"""Модели данных и вспомогательные утилиты для SOP Validator."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    OPEN = "open"
    BINARY = "binary"
    MULTIPLE = "multiple"


class InputQuestion(BaseModel):
    """Строка входного Excel-файла."""

    sop_id: str
    question: str
    question_type: QuestionType
    reference_answer: str
    options: str = ""

    @field_validator("question", "reference_answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Поле не может быть пустым")
        return v.strip()

    @property
    def is_closed(self) -> bool:
        return self.question_type in (QuestionType.BINARY, QuestionType.MULTIPLE)


class ArchitectureName(str, Enum):
    ZERO_SHOT = "zeroshot"
    COT = "cot"
    SELF_CONSISTENCY = "sc"


class ArchitectureResult(BaseModel):
    """Результат прогона одной архитектуры на одном вопросе."""

    architecture: ArchitectureName
    answer: str
    reasoning: str | None = None
    all_votes: list[str] = Field(default_factory=list)  # для SC
    latency_sec: float = 0.0
    total_tokens: int = 0
    error: str | None = None


class JudgeVerdict(BaseModel):
    score: Literal[0, 1] | None
    comment: str
    judge_failed: bool = False
    total_tokens: int = 0
    latency_sec: float = 0.0


class QuestionRunResult(BaseModel):
    """Полный результат обработки одного вопроса всеми архитектурами."""

    input: InputQuestion
    results: dict[ArchitectureName, ArchitectureResult] = Field(default_factory=dict)
    verdicts: dict[ArchitectureName, JudgeVerdict] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Нормализаторы и извлечение структурированных фрагментов
# ---------------------------------------------------------------------------


_COT_ANSWER_RE = re.compile(r"<ответ>\s*(.*?)\s*</ответ>", re.DOTALL | re.IGNORECASE)
_COT_REASONING_RE = re.compile(r"<размышление>\s*(.*?)\s*</размышление>", re.DOTALL | re.IGNORECASE)


def extract_cot_answer(raw: str) -> tuple[str, str | None]:
    """Извлекает финальный ответ и рассуждение из CoT-вывода.

    Возвращает (answer, reasoning_or_none).
    Если тегов нет — считаем, что весь ответ это и есть финал, reasoning = None.
    """
    answer_match = _COT_ANSWER_RE.search(raw)
    reasoning_match = _COT_REASONING_RE.search(raw)

    if answer_match:
        return answer_match.group(1).strip(), (
            reasoning_match.group(1).strip() if reasoning_match else None
        )

    # Fallback: модель могла пропустить теги. Пытаемся обрезать по "Ответ:" или "Финальный ответ:".
    for marker in ("Финальный ответ:", "Ответ:", "ОТВЕТ:"):
        if marker in raw:
            return raw.split(marker, 1)[1].strip(), None

    return raw.strip(), None


def normalize_binary_answer(text: str) -> str | None:
    """Возвращает 'да' / 'нет' / None по первому значимому слову ответа."""
    if not text:
        return None
    stripped = text.strip().lower()
    # Убираем ведущие пунктуацию и кавычки
    stripped = re.sub(r"^[^а-яa-z]+", "", stripped)
    if stripped.startswith("да"):
        return "да"
    if stripped.startswith("нет"):
        return "нет"
    # Типичные варианты
    if re.match(r"(требуется|необходимо|обязательн|верн|правильн)", stripped):
        return "да"
    if re.match(r"(не\s+требуется|не\s+нужно|не\s+обязательн|неверн|неправильн)", stripped):
        return "нет"
    return None


def normalize_for_voting(text: str, question_type: QuestionType) -> str:
    """Нормализация ответа для majority voting в self-consistency.

    - binary: 'да' / 'нет' / 'n/a'
    - multiple: первые 80 символов, приведённые к lower + squashed whitespace
    - open: первые 120 символов lowercase (для voting open всё равно уходит в LLM-agg)
    """
    if not text:
        return "n/a"
    if question_type == QuestionType.BINARY:
        return normalize_binary_answer(text) or "n/a"

    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if question_type == QuestionType.MULTIPLE:
        return cleaned[:80]
    return cleaned[:120]
