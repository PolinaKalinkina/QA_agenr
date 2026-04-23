"""Оркестрация: чтение входов, прогон архитектур, судейство, агрегация, запись."""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import core.architectures  # noqa: F401 — импорт ради регистрации архитектур
from core.aggregator import build_summary
from core.architectures.base import ArchitectureRegistry, BaseArchitecture
from core.excel_io import (
    CheckpointWriter,
    read_questions,
    write_results,
)
from core.gigachat_client import GigaChatClient
from core.judge import Judge
from core.models import (
    ArchitectureName,
    InputQuestion,
    QuestionRunResult,
)


def _process_one_question(
    question: InputQuestion,
    sop_text: str,
    architectures: list[BaseArchitecture],
    judge: Judge,
) -> QuestionRunResult:
    """Прогнать один вопрос через все архитектуры и судью (последовательно внутри потока)."""
    result = QuestionRunResult(input=question)
    for arch in architectures:
        try:
            arch_res = arch.answer(question, sop_text)
        except Exception as exc:  # защитная сетка
            logger.exception(f"Unhandled error in {arch.name.value}: {exc}")
            continue
        result.results[arch.name] = arch_res

        # Судья оценивает только если есть непустой ответ
        if arch_res.answer:
            verdict = judge.evaluate(question, arch_res.answer, sop_text)
        else:
            from core.models import JudgeVerdict

            verdict = JudgeVerdict(
                score=0, comment=f"Нет ответа (ошибка: {arch_res.error})", judge_failed=False
            )
        result.verdicts[arch.name] = verdict
    return result


def run(
    sop_path: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    config: dict,
    filter_archs: set[ArchitectureName] | None = None,
    detailed: bool = False,
) -> dict:
    """Полный пайплайн. Возвращает short-summary как dict (для CLI-печати)."""
    sop_path = Path(sop_path)
    questions_path = Path(questions_path)
    output_path = Path(output_path)

    # 1. СОП
    if not sop_path.exists():
        raise FileNotFoundError(f"SOP file not found: {sop_path}")
    sop_text = sop_path.read_text(encoding="utf-8").strip()
    if not sop_text:
        raise ValueError(f"SOP file is empty: {sop_path}")
    logger.info(f"Загружен СОП: {sop_path.name} ({len(sop_text)} символов)")

    # 2. Вопросы
    questions = read_questions(questions_path)

    # 3. Клиент + архитектуры + судья
    gc_cfg = config["gigachat"]
    credentials = gc_cfg.get("credentials") or os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS не задан. "
            "Скопируйте .env.example -> .env и впишите credentials."
        )
    client = GigaChatClient(
        credentials=credentials,
        model=gc_cfg["model"],
        scope=gc_cfg.get("scope", "GIGACHAT_API_CORP"),
        verify_ssl=gc_cfg.get("verify_ssl", False),
        timeout_seconds=gc_cfg.get("timeout_seconds", 30),
        max_retries=gc_cfg.get("max_retries", 3),
        retry_backoff_base=gc_cfg.get("retry_backoff_base", 1.0),
    )

    architectures = ArchitectureRegistry.build_enabled(
        config["architectures"], client, filter_names=filter_archs
    )
    if not architectures:
        raise RuntimeError("Ни одной архитектуры не включено — нечего запускать")

    enabled_archs = [a.name for a in architectures]
    logger.info(f"Активные архитектуры: {[a.value for a in enabled_archs]}")

    judge = Judge(client, config["judge"])

    # 4. Параллельная обработка
    exec_cfg = config["execution"]
    max_workers = int(exec_cfg.get("max_workers", 3))
    checkpoint_every = int(exec_cfg.get("checkpoint_every_n_questions", 20))
    continue_on_error = bool(exec_cfg.get("continue_on_error", True))
    seed = exec_cfg.get("random_seed")
    if seed is not None:
        random.seed(int(seed))
        logger.info(f"random_seed={seed} зафиксирован")

    checkpoint = CheckpointWriter(output_path, enabled_archs, checkpoint_every)
    results_by_input: dict[int, QuestionRunResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sop") as executor:
        futures = {
            executor.submit(
                _process_one_question, q, sop_text, architectures, judge
            ): idx
            for idx, q in enumerate(questions)
        }

        with tqdm(total=len(futures), desc="Вопросы", unit="q") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results_by_input[idx] = result
                    checkpoint.add(result)
                except Exception as exc:
                    if continue_on_error:
                        logger.exception(
                            f"Вопрос #{idx} упал, продолжаю: {exc}"
                        )
                    else:
                        raise
                pbar.update(1)

    # Восстанавливаем порядок входа
    all_results = [results_by_input[i] for i in sorted(results_by_input)]

    # 5. Сохранение финального результата
    checkpoint.flush()
    summary_df = build_summary(all_results, enabled_archs, detailed=detailed)
    # Для консольной сводки всегда считаем полные метрики
    detailed_summary = (
        summary_df if detailed else build_summary(all_results, enabled_archs, detailed=True)
    )
    write_results(output_path, all_results, enabled_archs, metrics_df=summary_df)
    client.close()

    # 6. Краткая сводка для CLI
    short = {}
    for _, row in detailed_summary.iterrows():
        short[row["architecture"]] = {
            "accuracy": row["accuracy"],
            "evaluated": row["evaluated"],
            "correct": row["correct"],
            "judge_failed": row["judge_failed"],
        }
    return short
