"""Chain-of-Thought архитектура: пошаговое рассуждение через промпт."""

from __future__ import annotations

from loguru import logger

from core.architectures.base import (
    FORMAT_INSTRUCTIONS,
    ArchitectureRegistry,
    BaseArchitecture,
    build_options_block,
    load_prompt,
)
from core.gigachat_client import GigaChatError
from core.models import ArchitectureName, ArchitectureResult, InputQuestion, extract_cot_answer


@ArchitectureRegistry.register(ArchitectureName.COT)
class ChainOfThoughtArchitecture(BaseArchitecture):
    """CoT через явный промпт с XML-тегами <размышление>/<ответ>."""

    def __init__(self, client, arch_config):
        super().__init__(client, arch_config)
        self._template = load_prompt("cot")

    def answer(self, question: InputQuestion, sop_text: str) -> ArchitectureResult:
        prompt = self._template.format(
            sop_text=sop_text,
            question=question.question,
            question_type=question.question_type.value,
            format_instructions=FORMAT_INSTRUCTIONS[question.question_type],
            options_block=build_options_block(question),
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 1500),
            )
            answer_text, reasoning = extract_cot_answer(response.content)
            return ArchitectureResult(
                architecture=self.name,
                answer=answer_text,
                reasoning=reasoning,
                latency_sec=response.latency_sec,
                total_tokens=response.total_tokens,
            )
        except GigaChatError as exc:
            logger.error(f"CoT failed on question: {question.question[:60]}: {exc}")
            return ArchitectureResult(
                architecture=self.name,
                answer="",
                error=str(exc),
            )
