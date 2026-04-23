"""LLM-as-judge: бинарная оценка (0/1) + комментарий.

Двухпроходная логика:
1. Основной промпт (judge.txt) → пытаемся распарсить JSON.
2. Если не распарсилось — fallback-промпт (judge_fallback.txt) → ещё одна попытка.
3. Если и это провалилось — score=None, judge_failed=True.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from core.gigachat_client import GigaChatClient, GigaChatError
from core.models import InputQuestion, JudgeVerdict


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_MAX_SOP_CONTEXT_CHARS = 6000  # чтобы не раздувать промпт судьи


def _load(name: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def _extract_json(text: str) -> dict | None:
    """Робастное извлечение JSON-объекта из ответа модели."""
    if not text:
        return None
    cleaned = text.strip()
    # Снимаем markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # Попытка 1: весь текст это JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Попытка 2: первый {...} внутри
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Попытка 3: жадный захват от первой { до последней }
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(cleaned[first : last + 1])
        except json.JSONDecodeError:
            pass

    return None


def _coerce_verdict(parsed: dict) -> JudgeVerdict | None:
    """Преобразовать распаршенный dict в JudgeVerdict с валидацией."""
    if not isinstance(parsed, dict):
        return None
    score_raw = parsed.get("score")
    comment = str(parsed.get("comment", "")).strip()
    # score может прийти как int, str, bool
    if isinstance(score_raw, bool):
        score = 1 if score_raw else 0
    elif isinstance(score_raw, (int, float)):
        score = int(score_raw)
    elif isinstance(score_raw, str):
        s = score_raw.strip().lower()
        if s in ("1", "true", "верно", "yes", "да"):
            score = 1
        elif s in ("0", "false", "неверно", "no", "нет"):
            score = 0
        else:
            return None
    else:
        return None

    if score not in (0, 1):
        return None
    return JudgeVerdict(score=score, comment=comment or "(без комментария)")


class Judge:
    """LLM-as-judge для оценки ответов модели."""

    def __init__(self, client: GigaChatClient, judge_config: dict) -> None:
        self.client = client
        self.config = judge_config
        self._main_template = _load("judge")
        self._fallback_template = _load("judge_fallback")

    def evaluate(
        self,
        question: InputQuestion,
        model_answer: str,
        sop_text: str | None = None,
    ) -> JudgeVerdict:
        """Оценить ответ модели. Всегда возвращает JudgeVerdict (при сбое — judge_failed=True)."""
        if not model_answer or not model_answer.strip():
            return JudgeVerdict(
                score=0, comment="Ответ модели пуст или отсутствует.", judge_failed=False
            )

        # Для binary/multiple контекст СОП только мешает — судья начинает оценивать
        # обоснование, а не сам ответ. СОП даём только для open-вопросов.
        include_sop = self.config.get("include_sop_context", True) and not question.is_closed
        sop_block = ""
        if include_sop and sop_text:
            truncated = sop_text[:_MAX_SOP_CONTEXT_CHARS]
            ellipsis = "…" if len(sop_text) > _MAX_SOP_CONTEXT_CHARS else ""
            sop_block = f"Контекст СОП (для справки):\n{truncated}{ellipsis}\n"

        prompt = self._main_template.format(
            question_type=question.question_type.value,
            question=question.question,
            reference_answer=question.reference_answer,
            model_answer=model_answer,
            sop_context_block=sop_block,
        )

        total_tokens = 0
        total_latency = 0.0

        verdict, t, l = self._try_once(prompt)
        total_tokens += t
        total_latency += l
        if verdict is None:
            fallback_prompt = self._fallback_template.format(
                question_type=question.question_type.value,
                question=question.question,
                reference_answer=question.reference_answer,
                model_answer=model_answer,
            )
            verdict, t, l = self._try_once(fallback_prompt)
            total_tokens += t
            total_latency += l

        if verdict is not None:
            comment = "" if question.is_closed else verdict.comment
            return JudgeVerdict(
                score=verdict.score,
                comment=comment,
                judge_failed=False,
                total_tokens=total_tokens,
                latency_sec=total_latency,
            )

        logger.error(
            f"Judge failed twice on question: {question.question[:80]!r}. "
            "Score будет None."
        )
        return JudgeVerdict(
            score=None,
            comment="Не удалось распарсить JSON-ответ судьи после двух попыток.",
            judge_failed=True,
            total_tokens=total_tokens,
            latency_sec=total_latency,
        )

    def _try_once(self, prompt: str) -> tuple[JudgeVerdict | None, int, float]:
        try:
            response = self.client.complete(
                prompt=prompt,
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 400),
            )
        except GigaChatError as exc:
            logger.warning(f"Judge call failed: {exc}")
            return None, 0, 0.0

        parsed = _extract_json(response.content)
        if parsed is None:
            return None, response.total_tokens, response.latency_sec
        return _coerce_verdict(parsed), response.total_tokens, response.latency_sec
