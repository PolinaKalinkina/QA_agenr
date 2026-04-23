"""CLI entry point для SOP Validator.

Пример:
    python main.py \\
        --sop ./sop.txt \\
        --questions ./questions.xlsx \\
        --output ./results/sop_results.xlsx \\
        --config ./config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from core.config_loader import load_config
from core.models import ArchitectureName
from core.runner import run


_ARCH_ALIASES = {
    "zero_shot": ArchitectureName.ZERO_SHOT,
    "zeroshot": ArchitectureName.ZERO_SHOT,
    "cot": ArchitectureName.COT,
    "self_consistency": ArchitectureName.SELF_CONSISTENCY,
    "sc": ArchitectureName.SELF_CONSISTENCY,
}


def _parse_architectures(arg: str | None) -> set[ArchitectureName] | None:
    if not arg:
        return None
    names = {a.strip().lower() for a in arg.split(",") if a.strip()}
    result = set()
    for name in names:
        if name not in _ARCH_ALIASES:
            raise argparse.ArgumentTypeError(
                f"Unknown architecture '{name}'. "
                f"Expected one of: {sorted(_ARCH_ALIASES)}"
            )
        result.add(_ARCH_ALIASES[name])
    return result


def _configure_logging(cfg_logging: dict, run_id: str) -> None:
    logger.remove()
    level = cfg_logging.get("level", "INFO")
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if cfg_logging.get("log_to_file", True):
        log_dir = Path(cfg_logging.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"run_{run_id}.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            encoding="utf-8",
        )
        logger.info(f"Детальный лог: {log_file}")


def _default_output_path(questions_path: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = questions_path.stem
    return questions_path.parent / f"{stem}_results_{run_id}.xlsx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SOP Validator: прогон вопросов по СОП через GigaChat в 3 архитектурах.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sop", required=True, help="Путь к .txt/.md файлу СОП")
    parser.add_argument(
        "--questions", required=True, help="Путь к Excel-файлу с вопросами"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь к выходному Excel (по умолчанию: <questions>_results_<timestamp>.xlsx)",
    )
    parser.add_argument(
        "--config", default="./config.yaml", help="Путь к config.yaml"
    )
    parser.add_argument(
        "--architectures",
        default=None,
        help="Список архитектур через запятую: zero_shot,cot,sc (по умолчанию — все включённые в config)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Добавить в Metrics развёрнутые DS-метрики (Wilson CI, токены, latency, per-type accuracy)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Загрузка конфига
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"[FATAL] Ошибка конфига: {exc}", file=sys.stderr)
        return 2

    _configure_logging(config.get("logging", {}), run_id)

    questions_path = Path(args.questions)
    output_path = Path(args.output) if args.output else _default_output_path(questions_path)

    try:
        filter_archs = _parse_architectures(args.architectures)
    except argparse.ArgumentTypeError as exc:
        logger.error(str(exc))
        return 2

    logger.info("=" * 60)
    logger.info("SOP Validator — старт")
    logger.info(f"SOP:       {args.sop}")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Output:    {output_path}")
    logger.info(f"Config:    {args.config}")
    logger.info("=" * 60)

    started = time.perf_counter()
    try:
        summary = run(
            sop_path=args.sop,
            questions_path=args.questions,
            output_path=output_path,
            config=config,
            filter_archs=filter_archs,
            detailed=args.detailed,
        )
    except Exception as exc:
        logger.exception(f"Прогон упал: {exc}")
        return 1

    elapsed = time.perf_counter() - started
    logger.info("=" * 60)
    logger.info(f"Готово за {elapsed:.1f} сек")
    logger.info("Accuracy по архитектурам:")
    for arch, stats in summary.items():
        logger.info(
            f"  {arch:18s} acc={stats['accuracy']:.2%} "
            f"({stats['correct']}/{stats['evaluated']}), judge_failed={stats['judge_failed']}"
        )
    logger.info(f"Результат: {output_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
