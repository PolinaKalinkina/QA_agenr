"""Self-Consistency: N прогонов CoT + агрегация (голосование или LLM-выбор)."""

from __future__ import annotations

import json
import re
from collections import Counter

from loguru import logger

from core.architectures.base import (
    FORMAT_INSTRUCTIONS,
    ArchitectureRegistry,
    BaseArchitecture,
    build_options_block,
    load_prompt,
)
from core.gigachat_client import GigaChatError
from core.models import (
    ArchitectureName,
    ArchitectureResult,
    InputQuestion,
    QuestionType,
    extract_cot_answer,
    normalize_for_voting,
)


@ArchitectureRegistry.register(ArchitectureName.SELF_CONSISTENCY)
class SelfConsistencyArchitecture(BaseArchitecture):
    """N прогонов CoT + majority vote (binary/multiple) или LLM-агрегация (open)."""

    def __init__(self, client, arch_config):
        super().__init__(client, arch_config)
        self._cot_template = load_prompt("cot")
        self._aggregator_template = load_prompt("sc_aggregator")

    def answer(self, question: InputQuestion, sop_text: str) -> ArchitectureResult:
        n_runs = int(self.config.get("n_runs", 5))
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1500)
        agg_temp = self.config.get("aggregation_temperature", 0.0)

        prompt = self._cot_template.format(
            sop_text=sop_text,
            question=question.question,
            question_type=question.question_type.value,
            format_instructions=FORMAT_INSTRUCTIONS[question.question_type],
            options_block=build_options_block(question),
        )

        votes: list[str] = []
        total_latency = 0.0
        total_tokens = 0
        errors: list[str] = []

        for i in range(n_runs):
            try:
                resp = self.client.complete(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer_text, _ = extract_cot_answer(resp.content)
                votes.append(answer_text.strip())
                total_latency += resp.latency_sec
                total_tokens += resp.total_tokens
            except GigaChatError as exc:
                msg = f"run {i + 1}/{n_runs}: {exc}"
                errors.append(msg)
                logger.warning(f"SC {msg}")

        if not votes:
            return ArchitectureResult(
                architecture=self.name,
                answer="",
                all_votes=[],
                latency_sec=total_latency,
                total_tokens=total_tokens,
                error=f"Все {n_runs} прогонов упали: " + "; ".join(errors),
            )

        # Агрегация
        if question.question_type in (QuestionType.BINARY, QuestionType.MULTIPLE):
            final_answer = self._majority_vote(votes, question.question_type)
        else:  # open
            try:
                final_answer, agg_latency, agg_tokens = self._llm_aggregate(
                    question.question, votes, agg_temp
                )
                total_latency += agg_latency
                total_tokens += agg_tokens
            except GigaChatError as exc:
                logger.warning(f"SC aggregator failed, fallback to longest answer: {exc}")
                final_answer = max(votes, key=len)

        return ArchitectureResult(
            architecture=self.name,
            answer=final_answer,
            all_votes=votes,
            latency_sec=total_latency,
            total_tokens=total_tokens,
            error="; ".join(errors) if errors else None,
        )

    @staticmethod
    def _majority_vote(votes: list[str], qtype: QuestionType) -> str:
        """Выбираем самый частый нормализованный ответ; возвращаем первый оригинал из этого кластера."""
        normalized = [normalize_for_voting(v, qtype) for v in votes]
        counter = Counter(normalized)
        winner_norm, _ = counter.most_common(1)[0]

        # Возвращаем первый оригинальный ответ, чья нормализация совпала с победителем.
        for original, norm in zip(votes, normalized):
            if norm == winner_norm:
                return original
        return votes[0]

    def _llm_aggregate(
        self, question_text: str, votes: list[str], temperature: float
    ) -> tuple[str, float, int]:
        """LLM-агрегация для open-вопросов. Возвращает (answer, latency, tokens)."""
        # Дополняем до 5, если прогонов меньше (чтобы промпт совпадал по структуре).
        padded = list(votes) + ["(прогон не выполнился)"] * (5 - len(votes))

        # Используем str.format_map с двойными фигурными скобками в промпте для JSON-примера.
        prompt = self._aggregator_template.format(
            question=question_text,
            answer_1=padded[0],
            answer_2=padded[1],
            answer_3=padded[2],
            answer_4=padded[3],
            answer_5=padded[4],
        )

        resp = self.client.complete(
            prompt=prompt, temperature=temperature, max_tokens=1000
        )
        parsed = _extract_json_object(resp.content)
        if parsed is None or "selected_answer" not in parsed:
            # Fallback: самый длинный ответ
            logger.warning("SC aggregator JSON parse failed, falling back to longest vote")
            return max(votes, key=len), resp.latency_sec, resp.total_tokens

        return str(parsed["selected_answer"]).strip(), resp.latency_sec, resp.total_tokens


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    """Извлекаем первый JSON-объект из строки. Устойчиво к обёрткам ```json ... ```."""
    if not text:
        return None
    cleaned = text.strip()
    # Снимаем markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
