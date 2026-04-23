"""Базовая архитектура и реестр."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

from core.gigachat_client import GigaChatClient
from core.models import ArchitectureName, ArchitectureResult, InputQuestion, QuestionType


# Инструкции по формату ответа для разных типов вопросов (подставляются в промпты).
FORMAT_INSTRUCTIONS: dict[QuestionType, str] = {
    QuestionType.OPEN: (
        "Ответ — свободная формулировка на русском языке, 1–4 предложения. "
        "Без списков, без markdown-оформления."
    ),
    QuestionType.BINARY: (
        "Ответ должен начинаться с одного слова «Да» или «Нет», "
        "после которого через точку допустимо короткое пояснение в 1 предложение."
    ),
    QuestionType.MULTIPLE: (
        "Ниже перечислены варианты ответа с буквами А/Б/В/Г/Д. "
        "В качестве ответа укажи букву и текст варианта дословно (например, «В) Нет, периодичность 60 минут»). "
        "Больше ничего не добавляй — ни пояснений, ни нумерации, ни markdown."
    ),
}


def build_options_block(question: InputQuestion) -> str:
    """Собрать блок с вариантами ответа для подстановки в промпт."""
    if question.options and question.options.strip():
        return f"\nВарианты ответа:\n{question.options.strip()}\n"
    return ""


def load_prompt(name: str) -> str:
    """Загрузить промпт из ./prompts/<name>.txt относительно корня пакета."""
    root = Path(__file__).resolve().parents[2]
    path = root / "prompts" / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


class BaseArchitecture(ABC):
    """Абстрактная архитектура промптинга."""

    name: ArchitectureName

    def __init__(self, client: GigaChatClient, arch_config: dict) -> None:
        self.client = client
        self.config = arch_config

    @abstractmethod
    def answer(self, question: InputQuestion, sop_text: str) -> ArchitectureResult:
        """Сгенерировать ответ на вопрос по СОП."""


class ArchitectureRegistry:
    """Реестр архитектур — для простого добавления новых."""

    _registry: dict[ArchitectureName, type[BaseArchitecture]] = {}

    @classmethod
    def register(cls, name: ArchitectureName):
        def decorator(klass: type[BaseArchitecture]) -> type[BaseArchitecture]:
            klass.name = name
            cls._registry[name] = klass
            logger.debug(f"Registered architecture: {name.value}")
            return klass

        return decorator

    @classmethod
    def build_enabled(
        cls,
        all_config: dict,
        client: GigaChatClient,
        filter_names: set[ArchitectureName] | None = None,
    ) -> list[BaseArchitecture]:
        """Создать инстансы всех включённых (enabled=true) архитектур.

        Если filter_names задан, оставляем только пересечение.
        """
        result: list[BaseArchitecture] = []
        mapping = {
            ArchitectureName.ZERO_SHOT: "zero_shot",
            ArchitectureName.COT: "cot",
            ArchitectureName.SELF_CONSISTENCY: "self_consistency",
        }
        for arch_name, config_key in mapping.items():
            arch_cfg = all_config.get(config_key, {})
            if not arch_cfg.get("enabled", True):
                continue
            if filter_names and arch_name not in filter_names:
                continue
            klass = cls._registry.get(arch_name)
            if klass is None:
                logger.warning(f"Architecture {arch_name.value} is enabled but not registered")
                continue
            result.append(klass(client, arch_cfg))
        return result
