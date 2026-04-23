"""Построение агрегатных метрик по результатам прогона."""

from __future__ import annotations

import math

import pandas as pd

from core.models import ArchitectureName, QuestionRunResult, QuestionType


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI для биномиальной пропорции. Для n=0 → (0, 0)."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


_ARCH_LABELS: dict[ArchitectureName, str] = {
    ArchitectureName.ZERO_SHOT: "zero_shot",
    ArchitectureName.COT: "cot",
    ArchitectureName.SELF_CONSISTENCY: "self_consistency",
}


def build_summary(
    results: list[QuestionRunResult],
    enabled_archs: list[ArchitectureName],
    detailed: bool = False,
) -> pd.DataFrame:
    """Сводка по архитектурам.

    detailed=False → 4 колонки: architecture, всего, правильно, accuracy
    detailed=True  → добавляются Wilson CI, tokens, latency, per-type accuracy
    """
    total = len(results)
    rows: list[dict] = []

    for arch in enabled_archs:
        correct = evaluated = judge_failed = 0
        arch_tokens = judge_tokens = 0
        arch_latency = judge_latency = 0.0
        by_type: dict[QuestionType, dict] = {t: {"correct": 0, "evaluated": 0} for t in QuestionType}

        for r in results:
            arch_res = r.results.get(arch)
            verdict = r.verdicts.get(arch)

            if arch_res is not None:
                arch_tokens += arch_res.total_tokens
                arch_latency += arch_res.latency_sec

            if verdict is None:
                continue
            judge_tokens += verdict.total_tokens
            judge_latency += verdict.latency_sec
            if verdict.judge_failed or verdict.score is None:
                judge_failed += 1
                continue

            evaluated += 1
            by_type[r.input.question_type]["evaluated"] += 1
            if verdict.score == 1:
                correct += 1
                by_type[r.input.question_type]["correct"] += 1

        row: dict = {
            "architecture": _ARCH_LABELS[arch],
            "total": total,
            "correct": correct,
            "accuracy": _safe_div(correct, evaluated),
        }
        if detailed:
            ci_low, ci_high = wilson_ci(correct, evaluated)
            total_tokens = arch_tokens + judge_tokens
            total_latency = arch_latency + judge_latency
            row["evaluated"] = evaluated
            row["ci95_low"] = ci_low
            row["ci95_high"] = ci_high
            for qtype in QuestionType:
                n = by_type[qtype]["evaluated"]
                if n > 0:
                    row[f"accuracy_{qtype.value}"] = _safe_div(by_type[qtype]["correct"], n)
                    row[f"n_{qtype.value}"] = n
            row["judge_failed"] = judge_failed
            row["arch_tokens"] = arch_tokens
            row["judge_tokens"] = judge_tokens
            row["total_tokens"] = total_tokens
            row["avg_latency_sec"] = round(total_latency / total, 3) if total else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def _safe_div(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0
