"""Пакет архитектур. Импорты гарантируют регистрацию в ArchitectureRegistry."""

from core.architectures.base import ArchitectureRegistry, BaseArchitecture
from core.architectures.cot import ChainOfThoughtArchitecture
from core.architectures.self_consistency import SelfConsistencyArchitecture
from core.architectures.zero_shot import ZeroShotArchitecture

__all__ = [
    "ArchitectureRegistry",
    "BaseArchitecture",
    "ChainOfThoughtArchitecture",
    "SelfConsistencyArchitecture",
    "ZeroShotArchitecture",
]
