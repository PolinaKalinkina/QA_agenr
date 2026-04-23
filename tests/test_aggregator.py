"""Тесты build_summary (базовый + detailed) + Wilson CI."""

from __future__ import annotations

from core.aggregator import build_summary, wilson_ci
from core.models import (
    ArchitectureName,
    ArchitectureResult,
    InputQuestion,
    JudgeVerdict,
    QuestionRunResult,
    QuestionType,
)


def _make_result(qtext, qtype, ref, scores):
    qr = QuestionRunResult(
        input=InputQuestion(
            sop_id="S1", question=qtext, question_type=qtype, reference_answer=ref
        )
    )
    for arch, score in scores.items():
        qr.results[arch] = ArchitectureResult(
            architecture=arch, answer="ans", latency_sec=0.5, total_tokens=100,
        )
        qr.verdicts[arch] = JudgeVerdict(
            score=score, comment="test", judge_failed=(score is None),
        )
    return qr


class TestSummaryBasic:
    def test_all_correct(self):
        archs = [ArchitectureName.ZERO_SHOT, ArchitectureName.COT]
        results = [
            _make_result("q1", QuestionType.BINARY, "Да",
                {ArchitectureName.ZERO_SHOT: 1, ArchitectureName.COT: 1}),
            _make_result("q2", QuestionType.OPEN, "ref",
                {ArchitectureName.ZERO_SHOT: 1, ArchitectureName.COT: 1}),
        ]
        df = build_summary(results, archs)
        assert list(df.columns) == ["architecture", "total", "correct", "accuracy"]
        assert len(df) == 2
        zs = df[df["architecture"] == "zero_shot"].iloc[0]
        assert zs["accuracy"] == 1.0
        assert zs["correct"] == 2
        assert zs["total"] == 2

    def test_mixed(self):
        archs = [ArchitectureName.ZERO_SHOT]
        results = [
            _make_result("q1", QuestionType.BINARY, "Да", {ArchitectureName.ZERO_SHOT: 1}),
            _make_result("q2", QuestionType.BINARY, "Нет", {ArchitectureName.ZERO_SHOT: 0}),
            _make_result("q3", QuestionType.OPEN, "x", {ArchitectureName.ZERO_SHOT: 1}),
        ]
        df = build_summary(results, archs)
        row = df.iloc[0]
        assert row["correct"] == 2
        assert abs(row["accuracy"] - 2 / 3) < 1e-4


class TestSummaryDetailed:
    def test_includes_ci_and_tokens(self):
        archs = [ArchitectureName.ZERO_SHOT]
        results = [
            _make_result("q1", QuestionType.BINARY, "Да", {ArchitectureName.ZERO_SHOT: 1}),
            _make_result("q2", QuestionType.OPEN, "x", {ArchitectureName.ZERO_SHOT: 0}),
        ]
        df = build_summary(results, archs, detailed=True)
        row = df.iloc[0]
        assert {"ci95_low", "ci95_high", "arch_tokens", "judge_tokens",
                "total_tokens", "avg_latency_sec", "evaluated"} <= set(df.columns)
        assert 0.0 <= row["ci95_low"] <= row["accuracy"] <= row["ci95_high"] <= 1.0

    def test_per_type_accuracy_only_present_types(self):
        archs = [ArchitectureName.ZERO_SHOT]
        # Только binary в датасете
        results = [
            _make_result("q1", QuestionType.BINARY, "Да", {ArchitectureName.ZERO_SHOT: 1}),
            _make_result("q2", QuestionType.BINARY, "Нет", {ArchitectureName.ZERO_SHOT: 0}),
        ]
        df = build_summary(results, archs, detailed=True)
        assert "accuracy_binary" in df.columns
        assert "n_binary" in df.columns
        # Для пустых типов колонки не добавляются
        assert "accuracy_open" not in df.columns
        assert "accuracy_multiple" not in df.columns

    def test_judge_failed_counted(self):
        archs = [ArchitectureName.ZERO_SHOT]
        results = [
            _make_result("q1", QuestionType.OPEN, "x", {ArchitectureName.ZERO_SHOT: 1}),
            _make_result("q2", QuestionType.OPEN, "x", {ArchitectureName.ZERO_SHOT: None}),
        ]
        df = build_summary(results, archs, detailed=True)
        row = df.iloc[0]
        assert row["correct"] == 1
        assert row["evaluated"] == 1
        assert row["judge_failed"] == 1


class TestWilson:
    def test_full_success(self):
        low, high = wilson_ci(30, 30)
        assert high == 1.0
        assert low > 0.85

    def test_contains_point(self):
        for c, n in [(1, 10), (5, 30), (80, 100)]:
            p = c / n
            low, high = wilson_ci(c, n)
            assert low - 1e-6 <= p <= high + 1e-6

    def test_zero_of_zero(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)
